/* qmesh_afd_peer: the P7-side AFD-TP3 transport as a CPython extension.
 *
 * The benchmark binary (qmesh_afd_edge) owns the wire protocol and the
 * measurements. This module exposes the same protocol one layer at a time so a
 * serving runtime can drive it per MoE layer:
 *
 *   peer = _qmesh_afd.Peer("mlx5_0", 5, "127.0.0.1", 47101, layers=40,
 *                          rows=1, iters=0)
 *   peer.post(L, x_dev_ptr, rout_dev_ptr)   # routing + payload + in-flag
 *   peer.wait(L)                            # out-flag of this layer
 *   wr = peer.read(L)                       # RDMA READ into the slab out-slot
 *   peer.collect(wr)                        # completion of that READ
 *   mv = peer.out_mv(L)                     # rows*K*2 bytes, zero copy
 *
 * Design notes, all of them measured on the two edges:
 *  - The payload slab, the flag plane and the routing plane are pinned host
 *    memory (cudaHostAllocMapped). ibv_reg_mr cannot register cudaMalloc memory
 *    here (no nvidia-peermem: EFAULT on P7 and on GB10) and dmabuf export fails
 *    on GB10, so pinned host is the one MR type both NICs accept.
 *  - The routing WRITE is posted unsignaled ahead of the payload on the same RC
 *    QP, so the worker cannot observe a flag for routing it has not applied.
 *  - Out-flag polling uses 4 B RDMA READs on a second QP, so the poll is never
 *    queued behind an in-flight payload READ.
 *  - Blocking spans release the GIL: the poll loops are microseconds apart from
 *    each other but the caller may have other threads.
 */

#include <Python.h>

#include <infiniband/verbs.h>

#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>
#include <emmintrin.h>

#define DIE(msg) do { fprintf(stderr, "%s: %s (errno=%d)\n", msg, strerror(errno), errno); exit(1); } while (0)
#define CHECK_CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); exit(2); } } while (0)

/* ---- wire layout, identical to qmesh_afd_edge.cpp and the GX worker ---- */
static constexpr int      NL        = 40;
static constexpr int      K_HIDDEN  = 5120;
/* Row capacity of the wire; a post carries its own valid-row count in the
 * routing header, so decode sends 1 row and prefill a whole chunk.
 *
 * One in-slot and one out-slot serve every layer, which keeps the registered
 * region at 2 slots (336 MiB at 4096 rows) instead of 2*NL slots (13 GiB). The
 * reuse is safe because the client is serialized per layer: it posts layer L+1
 * only after collect(L) observed L's out-flag, and the worker raises that flag
 * after it has read L's payload and routing. The flag plane stays per layer.
 */
static constexpr int      R_MAX     = 4096;
static constexpr size_t   SLOT      = (size_t)K_HIDDEN * 2 * R_MAX;  /* 41,943,040 */
static constexpr size_t   IN_OFF    = 0;
static constexpr size_t   OUT_OFF   = SLOT;
static constexpr size_t   SLAB      = 2 * SLOT;
static constexpr size_t   FSTRIDE   = 64;
static constexpr size_t   FIN_OFF   = 0;
static constexpr size_t   FOUT_OFF  = (size_t)NL * FSTRIDE;
static constexpr size_t   FLAGS     = (size_t)2 * NL * FSTRIDE;

/* Layer 0's flag stride: word 0 is its in-flag, word 1 the wire-engine epoch,
 * word 2 the worker's quiescent-epoch ACK. The ACK is how a newly connected
 * client learns that a worker left over from a previous client has stopped
 * reading the slab and the routing plane, without fabricating a round. */
static constexpr size_t   EPOCH_OFF = 4;
static constexpr size_t   ACK_OFF   = 8;

/* Payload: e4m3 activations, one byte per element, one scale in the routing
 * header (hdr[2] = 1/s as float bits, hdr[3] = 1 marking the fp8 payload). The
 * expert kernels quantize these rows to nvfp4 with a per-tensor scale before
 * the GEMM, so an fp8 payload costs one rounding step on top of the ~4 bits
 * fp4 keeps anyway, and it halves the bytes of every layer in both directions
 * of the post. The out-slot stays bf16: that is the partial the attention host
 * sums. SLOT keeps its bf16 size so the slab, the edge and the worker's mmap
 * are unchanged; the in-slot simply carries n*K bytes of it. */
static constexpr int      IN_ELEM   = 1;
static constexpr size_t   IN_ROW    = (size_t)K_HIDDEN * IN_ELEM;
static constexpr int      FMT_FP8   = 1;    /* hdr[3] */
static inline uint32_t in_bytes(int n) { return (uint32_t)((size_t)n * IN_ROW); }

static constexpr int      TOPK_ROUT = 6;
static constexpr size_t   ROUT_ROW  = (size_t)TOPK_ROUT * 4 * 2;
static constexpr size_t   ROUT_HDR  = 16;   /* int32 seq, int32 n, float 1/s, int32 fmt */
static constexpr size_t   ROUT_SLOT = ROUT_HDR + (size_t)R_MAX * ROUT_ROW;
static constexpr size_t   ROUT      = ROUT_SLOT;

static inline size_t in_slot(int)       { return IN_OFF; }
static inline size_t out_slot(int)      { return OUT_OFF; }
static inline size_t in_flag(int L)  { return FIN_OFF + (size_t)L * FSTRIDE; }
static inline size_t out_flag(int L) { return FOUT_OFF + (size_t)L * FSTRIDE; }
static inline size_t rout_slot(int)     { return 0; }

struct ConnMeta {
  uint32_t lid, qpn, psn, rkey, size, mem_mode, gid_idx, mtu;
  uint8_t  gid[16];
  uint64_t vaddr;
  uint32_t flag_rkey, layers, iters;
  uint64_t flag_vaddr;
  uint32_t qpn2, psn2;
  uint32_t rout_rkey;
  uint64_t rout_vaddr;
};

static uint8_t g_peer_gid[16];
/* Each peer's post/wait/read release the GIL, so the Python driver runs the
 * peers' wire phases on separate threads. The ids must stay unique across
 * peers: a duplicate would let one thread's CQ poll consume the other's
 * completion and then wait for a wr_id that has already been seen. */
static std::atomic<uint64_t> g_wr_id{1};
static uint64_t take_wr_id() { return g_wr_id.fetch_add(1, std::memory_order_relaxed); }

static int mgmt_connect(const char* ip, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) DIE("socket");
  int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) DIE("inet_pton");
  for (int i = 0; i < 600; i++) {
    if (connect(fd, (sockaddr*)&a, sizeof(a)) == 0) return fd;
    usleep(100000);
  }
  DIE("connect timeout");
  return -1;
}

static void xfer_all(int fd, void* buf, size_t n, int send_mode) {
  uint8_t* p = (uint8_t*)buf; size_t left = n;
  while (left) {
    ssize_t r = send_mode ? send(fd, p, left, 0) : recv(fd, p, left, MSG_WAITALL);
    if (r <= 0) DIE(send_mode ? "send" : "recv");
    p += r; left -= (size_t)r;
  }
}

/* ---- one peer ---- */
struct Peer {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_qp* qp = nullptr;
  ibv_qp* qp2 = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  ibv_mr* fmr = nullptr;
  ibv_mr* rmr = nullptr;
  ibv_port_attr port_attr{};
  int port = 1, gid_idx = 3;

  void* buf = nullptr;      /* host VA of the payload slab */
  void* flags = nullptr;
  void* rbuf = nullptr;     /* routing plane */
  uint64_t base = 0, fbase = 0, rbase = 0;

  ConnMeta remote{};
  int layers = NL, rows = 1;
  uint32_t bytes = 0;
  int32_t seq = 0;
  cudaStream_t st = nullptr;
  std::set<uint64_t> done;
  long long polls_total = 0;
  long long layer_polls = 0;
  /* Serve mode is a relay that returns when the client's management socket
   * closes. Hold it for the life of the peer so the peer's lifetime, not the
   * handshake's, decides when the edge releases the port. */
  int mfd = -1;
  int open(const char* dev_name, int gid_idx_, const char* mgmt_ip, int mgmt_port,
           int layers_, int rows_, int iters_);
  void begin_round() { seq++; }
  cudaStream_t cs = nullptr;   /* caller's stream; staged copies run on it */
  void stage_and_post(int L, uint64_t x_ptr, uint64_t rout_ptr, int n, float inv_scale);
  /* timeout_ms: 0 waits forever (the old caller's behaviour); >0 gives up
   * after that many milliseconds and returns -1 without touching the slab. */
  int wait_layer(int L, long long timeout_ms = 0);
  uint64_t read_layer(int L, int n);
  void collect(uint64_t wr);
  void set_stream(uint64_t s) { cs = (cudaStream_t)(uintptr_t)s; }
  void close();
};

static void poll_cq_tracked(Peer& p, uint64_t want);

static void open_device(Peer& p, const char* dev_name) {
  int n = 0; ibv_device** list = ibv_get_device_list(&n);
  if (!list || n <= 0) DIE("ibv_get_device_list");
  ibv_device* dev = nullptr;
  for (int i = 0; i < n; i++)
    if (strcmp(ibv_get_device_name(list[i]), dev_name) == 0) dev = list[i];
  if (!dev) {
    fprintf(stderr, "IB device %s not found; available:", dev_name);
    for (int i = 0; i < n; i++) fprintf(stderr, " %s", ibv_get_device_name(list[i]));
    fprintf(stderr, "\n");
    exit(3);
  }
  p.ctx = ibv_open_device(dev);
  if (!p.ctx) DIE("ibv_open_device");
  if (ibv_query_port(p.ctx, p.port, &p.port_attr)) DIE("ibv_query_port");
  p.pd = ibv_alloc_pd(p.ctx); if (!p.pd) DIE("ibv_alloc_pd");
  p.cq = ibv_create_cq(p.ctx, 8192, nullptr, nullptr, 0); if (!p.cq) DIE("ibv_create_cq");
  ibv_free_device_list(list);
}

static void create_one_qp(Peer& p, ibv_qp** out) {
  ibv_qp_init_attr attr{};
  attr.send_cq = p.cq; attr.recv_cq = p.cq;
  attr.cap.max_send_wr = 512; attr.cap.max_recv_wr = 64;
  attr.cap.max_send_sge = 1;  attr.cap.max_recv_sge = 1;
  attr.cap.max_inline_data = 64;
  attr.qp_type = IBV_QPT_RC;
  *out = ibv_create_qp(p.pd, &attr);
  if (!*out) { attr.cap.max_inline_data = 0; *out = ibv_create_qp(p.pd, &attr); }
  if (!*out) DIE("ibv_create_qp");
}

static void alloc_and_reg(Peer& p) {
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaHostAlloc(&p.buf, SLAB, cudaHostAllocMapped));
  CHECK_CUDA(cudaHostAlloc(&p.flags, FLAGS, cudaHostAllocMapped));
  CHECK_CUDA(cudaHostAlloc(&p.rbuf, ROUT, cudaHostAllocMapped));
  /* Session epoch: a worker started before this process must be able to tell
   * stale flags from live ones, so bump the counter in the spare word of layer
   * 0's flag stride after zeroing everything. */
  int32_t* sess = (int32_t*)((char*)p.flags + 4);
  const int32_t epoch = *sess + 1;
  memset(p.buf, 0, SLAB);
  memset(p.flags, 0, FLAGS);
  memset(p.rbuf, 0, ROUT);
  *sess = epoch;
  p.base = (uint64_t)(uintptr_t)p.buf;
  p.fbase = (uint64_t)(uintptr_t)p.flags;
  p.rbase = (uint64_t)(uintptr_t)p.rbuf;
  p.mr = ibv_reg_mr(p.pd, p.buf, SLAB, access);
  if (!p.mr) DIE("ibv_reg_mr(slab)");
  p.fmr = ibv_reg_mr(p.pd, p.flags, FLAGS, access);
  if (!p.fmr) DIE("ibv_reg_mr(flags)");
  p.rmr = ibv_reg_mr(p.pd, p.rbuf, ROUT, access);
  if (!p.rmr) DIE("ibv_reg_mr(rout)");
}

static void qp_to_init(Peer& p, ibv_qp* qp) {
  ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = p.port;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    DIE("qp init");
}

static void qp_to_rtr(Peer& p, ibv_qp* qp, uint32_t dqpn, uint32_t dpsn, uint8_t dlid) {
  ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTR;
  a.path_mtu = p.port_attr.active_mtu;
  a.dest_qp_num = dqpn; a.rq_psn = dpsn;
  a.ah_attr.dlid = dlid; a.ah_attr.sl = 0; a.ah_attr.src_path_bits = 0;
  a.ah_attr.port_num = (uint8_t)p.port;
  a.ah_attr.is_global = 1;
  memcpy(a.ah_attr.grh.dgid.raw, g_peer_gid, 16);
  a.ah_attr.grh.sgid_index = (uint8_t)p.gid_idx;
  a.ah_attr.grh.hop_limit = 1;
  a.max_dest_rd_atomic = 4; a.min_rnr_timer = 12;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    DIE("qp rtr");
}

static void qp_to_rts(Peer& p, ibv_qp* qp, uint32_t psn) {
  ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTS;
  a.sq_psn = psn; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7;
  a.max_rd_atomic = 4;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                             IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
    DIE("qp rts");
}

int Peer::open(const char* dev_name, int gid_idx_, const char* mgmt_ip, int mgmt_port,
               int layers_, int rows_, int iters_) {
  gid_idx = gid_idx_;
  layers = layers_;
  rows = rows_;
  if (layers < 1 || layers > NL) { fprintf(stderr, "layers 1..%d\n", NL); return -1; }
  if (rows < 1 || rows > R_MAX) { fprintf(stderr, "rows 1..%d\n", R_MAX); return -1; }
  bytes = (uint32_t)rows * K_HIDDEN * 2;

  open_device(*this, dev_name);
  create_one_qp(*this, &qp);
  create_one_qp(*this, &qp2);
  alloc_and_reg(*this);
  CHECK_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  qp_to_init(*this, qp);
  qp_to_init(*this, qp2);

  union ibv_gid lg{};
  if (ibv_query_gid(ctx, port, gid_idx, &lg)) DIE("query_gid");
  const uint32_t my_psn = (uint32_t)(lrand48() & 0xffffff);
  const uint32_t my_psn2 = (uint32_t)(lrand48() & 0xffffff);

  ConnMeta local{};
  local.lid = port_attr.lid; local.qpn = qp->qp_num; local.psn = my_psn;
  local.qpn2 = qp2->qp_num; local.psn2 = my_psn2;
  local.rkey = mr->rkey; local.vaddr = base;
  memcpy(local.gid, lg.raw, 16);
  local.size = (uint32_t)SLAB; local.gid_idx = (uint32_t)gid_idx;
  local.mtu = (uint32_t)port_attr.active_mtu;
  local.flag_rkey = fmr->rkey; local.flag_vaddr = fbase;
  local.rout_rkey = rmr->rkey; local.rout_vaddr = rbase;
  local.layers = (uint32_t)layers; local.iters = (uint32_t)iters_;

  const int mfd = mgmt_connect(mgmt_ip, mgmt_port);
  xfer_all(mfd, &remote, sizeof(remote), 0);
  xfer_all(mfd, &local, sizeof(local), 1);
  if (remote.size != SLAB) { fprintf(stderr, "slab mismatch\n"); return -1; }
  if (remote.layers != (uint32_t)layers) {
    fprintf(stderr, "peer layers=%u want %d\n", remote.layers, layers);
    return -1;
  }
  if (remote.iters != (uint32_t)iters_)
    fprintf(stderr, "note: peer iters=%u, client iters=%d (schedule is client-driven)\n",
            remote.iters, iters_);
  memcpy(g_peer_gid, remote.gid, 16);
  qp_to_rtr(*this, qp, remote.qpn, remote.psn, (uint8_t)remote.lid);
  qp_to_rts(*this, qp, my_psn);
  qp_to_rtr(*this, qp2, remote.qpn2, remote.psn2, (uint8_t)remote.lid);
  qp_to_rts(*this, qp2, my_psn2);
  /* The peer sends its byte only after its QPs are in RTS. Exchanging here,
   * before the first RDMA READ below, is what makes the handshake safe: the
   * server cannot still be in INIT when this client's flag read arrives. */
  {
    char b = 1;
    xfer_all(mfd, &b, 1, 1);
    xfer_all(mfd, &b, 1, 0);
  }
  /* Connect handshake.
   *
   * A worker left over from a dead client may be parked in a layer wait whose
   * in-flag the dead client already raised, or in a device spin on a flag that
   * the new client is about to observe. Raising in-flags to a made-up sequence
   * (the old resume) freed such a worker by letting it serve a round nobody
   * posted: it read the new client's routing and slab and published out-flags
   * for a round the new client had not begun. Instead, claim the wire and wait
   * until the worker says it is no longer reading round state:
   *
   *   1. Read the worker's flag plane.
   *   2. Clear every in/out flag and keep the epoch and ACK (the worker never
   *      bumps the epoch, so this only hides the dead round's flags).
   *   3. Bump the epoch. A worker between rounds or inside any layer wait
   *      notices at its next check, synchronizes its device work, and aborts
   *      the old round without reading routing or payload.
   *   4. Read the ACK word until it carries the new epoch. Only then clear the
   *      plane again and start at seq 0, so the first post is round 1.
   *
   * The clears and the bump are RDMA WRITEs on one RC QP, so the worker cannot
   * see the bump before the clears. It re-checks the epoch after a flag becomes
   * visible, which closes the remaining window between its check and the flag
   * load. The ACK wait is bounded and fails open() rather than posting into a
   * worker that never became quiescent. */
  {
    int32_t* pflags = (int32_t*)flags;
    ibv_send_wr* bad = nullptr;

    /* 1. Read the whole remote flag plane into the registered local mirror. */
    uint64_t rid = take_wr_id();
    ibv_sge sge{}; sge.addr = fbase; sge.length = (uint32_t)FLAGS; sge.lkey = fmr->lkey;
    ibv_send_wr rw{};
    rw.wr_id = rid; rw.sg_list = &sge; rw.num_sge = 1;
    rw.opcode = IBV_WR_RDMA_READ; rw.send_flags = IBV_SEND_SIGNALED;
    rw.wr.rdma.remote_addr = remote.flag_vaddr;
    rw.wr.rdma.rkey = remote.flag_rkey;
    if (ibv_post_send(qp2, &rw, &bad)) DIE("post READ flags");
    poll_cq_tracked(*this, rid);

    const int32_t e_old = pflags[EPOCH_OFF / 4];
    const int32_t ack_old = pflags[ACK_OFF / 4];
    /* Epochs are non-negative and increase; a wrapped or corrupt value falls
     * back to 1 rather than 0, which is the "no epoch yet" value. */
    const int32_t e_new = (e_old >= 0 && e_old < 0x7fffffff) ? e_old + 1 : 1;

    /* 2. Clear the in/out flags. Preserve the epoch (a zero would look like a
     *    new engine to a worker between rounds) and the old ACK (harmless: the
     *    wait below wants e_new). */
    memset(pflags, 0, FLAGS);
    pflags[EPOCH_OFF / 4] = e_old;
    pflags[ACK_OFF / 4] = ack_old;
    rid = take_wr_id();
    sge.addr = fbase; sge.length = (uint32_t)FLAGS;
    ibv_send_wr cw{};
    cw.wr_id = rid; cw.sg_list = &sge; cw.num_sge = 1;
    cw.opcode = IBV_WR_RDMA_WRITE; cw.send_flags = IBV_SEND_SIGNALED;
    cw.wr.rdma.remote_addr = remote.flag_vaddr;
    cw.wr.rdma.rkey = remote.flag_rkey;
    if (ibv_post_send(qp, &cw, &bad)) DIE("post WRITE flags (clear)");
    poll_cq_tracked(*this, rid);

    /* 3. Claim the wire. The bump is ordered after the clear on this QP. */
    pflags[EPOCH_OFF / 4] = e_new;
    __sync_synchronize();
    rid = take_wr_id();
    ibv_sge es{}; es.addr = fbase + EPOCH_OFF; es.length = 4; es.lkey = fmr->lkey;
    ibv_send_wr ew{};
    ew.wr_id = rid; ew.sg_list = &es; ew.num_sge = 1;
    ew.opcode = IBV_WR_RDMA_WRITE; ew.send_flags = IBV_SEND_SIGNALED;
    ew.wr.rdma.remote_addr = remote.flag_vaddr + EPOCH_OFF;
    ew.wr.rdma.rkey = remote.flag_rkey;
    if (ibv_post_send(qp, &ew, &bad)) DIE("post WRITE epoch");
    poll_cq_tracked(*this, rid);

    /* 4. Wait for the worker to acknowledge that it is quiescent on e_new. A
     *    stale ACK from an older epoch cannot satisfy this because the check is
     *    equality against a monotonically increasing value. */
    const char* ack_env = getenv("QMESH_AFD_ACK_TIMEOUT_MS");
    long long timeout_ms = ack_env ? atoll(ack_env) : 30000;
    if (timeout_ms < 1000) timeout_ms = 1000;
    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long waited_ms = 0;
    for (;;) {
      rid = take_wr_id();
      ibv_sge as{}; as.addr = fbase + ACK_OFF; as.length = 4; as.lkey = fmr->lkey;
      ibv_send_wr aw{};
      aw.wr_id = rid; aw.sg_list = &as; aw.num_sge = 1;
      aw.opcode = IBV_WR_RDMA_READ; aw.send_flags = IBV_SEND_SIGNALED;
      aw.wr.rdma.remote_addr = remote.flag_vaddr + ACK_OFF;
      aw.wr.rdma.rkey = remote.flag_rkey;
      if (ibv_post_send(qp2, &aw, &bad)) DIE("post READ ack");
      poll_cq_tracked(*this, rid);
      if (pflags[ACK_OFF / 4] == e_new) break;
      clock_gettime(CLOCK_MONOTONIC, &t1);
      waited_ms = (long long)((t1.tv_sec - t0.tv_sec) * 1000
                              + (t1.tv_nsec - t0.tv_nsec) / 1000000);
      if (waited_ms >= timeout_ms) {
        fprintf(stderr, "AFD: epoch %d -> %d: no quiescent ACK after %lld ms "
                        "(worker still reading?); not posting\n",
                e_old, e_new, waited_ms);
        ::close(mfd);
        return -1;
      }
      usleep(1000);
    }

    /* 5. The worker is between rounds. Erase any out-flag the aborted round
     *    published before it synchronized, keep the claimed epoch, and start
     *    the first round at 1. */
    memset(pflags, 0, FLAGS);
    pflags[EPOCH_OFF / 4] = e_new;
    pflags[ACK_OFF / 4] = e_new;
    rid = take_wr_id();
    sge.addr = fbase; sge.length = (uint32_t)FLAGS;
    ibv_send_wr fw{};
    fw.wr_id = rid; fw.sg_list = &sge; fw.num_sge = 1;
    fw.opcode = IBV_WR_RDMA_WRITE; fw.send_flags = IBV_SEND_SIGNALED;
    fw.wr.rdma.remote_addr = remote.flag_vaddr;
    fw.wr.rdma.rkey = remote.flag_rkey;
    if (ibv_post_send(qp, &fw, &bad)) DIE("post WRITE flags (quiescent)");
    poll_cq_tracked(*this, rid);
    seq = 0;
    fprintf(stderr, "AFD: epoch %d -> %d, quiescent ACK after %lld ms; "
                    "first round seq 1\n", e_old, e_new, waited_ms);
  }
  this->mfd = mfd;
  return 0;
}

static void poll_cq_tracked(Peer& p, uint64_t want) {
  if (p.done.erase(want)) return;
  timespec t0{}, t1{};
  clock_gettime(CLOCK_MONOTONIC, &t0);
  long long n = 0;
  for (;;) {
    ibv_wc wc{};
    int k = ibv_poll_cq(p.cq, 1, &wc);
    if (k < 0) DIE("poll_cq");
    if (k == 0) {
      /* A completion that never arrives used to spin here forever, which is
       * indistinguishable from a peer that has not answered yet. Report the QP
       * states every five seconds of silence and stop rather than hang. */
      if ((++n & 0xFFFFF) == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double dt = (double)(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (dt > 5.0) {
          ibv_qp_attr qa{}; ibv_qp_init_attr qi{};
          int qs = -1, qs2 = -1;
          if (!ibv_query_qp(p.qp, &qa, IBV_QP_STATE, &qi)) qs = qa.qp_state;
          if (!ibv_query_qp(p.qp2, &qa, IBV_QP_STATE, &qi)) qs2 = qa.qp_state;
          fprintf(stderr, "AFD: no completion after %.0fs (%lld polls, want=%lu, "
                          "stashed=%zu, qp=%d qp2=%d, seq=%d)\n",
                  dt, n + p.polls_total, (unsigned long)want, p.done.size(), qs, qs2, p.seq);
          if (dt > 60.0) {
            fprintf(stderr, "AFD: abandoning the completion queue\n");
            exit(12);
          }
          n = 0;
          clock_gettime(CLOCK_MONOTONIC, &t0);
        }
      }
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "wc status=%s opcode=%d wr_id=%lu want=%lu\n",
              ibv_wc_status_str(wc.status), wc.opcode, (unsigned long)wc.wr_id,
              (unsigned long)want);
      exit(9);
    }
    if (wc.wr_id == want) return;
    p.done.insert(wc.wr_id);
  }
}

void Peer::stage_and_post(int L, uint64_t x_ptr, uint64_t rout_ptr, int n, float inv_scale) {
  if (seq < 1) { fprintf(stderr, "post before begin_round\n"); exit(4); }
  if (L < 0 || L >= layers) { fprintf(stderr, "layer %d out of range\n", L); exit(4); }
  if (n < 1 || n > rows) { fprintf(stderr, "post rows %d outside 1..%d\n", n, rows); exit(4); }
  /* One sequence per round of `layers` posts, not per layer: the GX worker
   * serves a round as 40 sequential layers under a single sequence number, and
   * each layer's in-flag carries it. Caller bumps once per step. */
  const int32_t s = seq;
  const size_t xbytes = in_bytes(n);
  /* Activation (n*K fp8) and routing (16 B header + n*48) land in the registered
   * slab and routing plane; the NIC reads them from there. Both are produced on
   * the caller's CUDA stream, so the copies have to run on that stream: copying
   * from a tensor whose producer kernel is still queued elsewhere puts the
   * previous layer on the wire. Without a caller stream, drain the device
   * first. */
  cudaStream_t cp = cs;
  if (!cp) { CHECK_CUDA(cudaDeviceSynchronize()); cp = st; }
  CHECK_CUDA(cudaMemcpyAsync((void*)(uintptr_t)(base + in_slot(L)), (const void*)(uintptr_t)x_ptr,
                             xbytes, cudaMemcpyDeviceToHost, cp));
  if (rout_ptr)
    CHECK_CUDA(cudaMemcpyAsync((void*)(uintptr_t)(rbase + rout_slot(L) + ROUT_HDR),
                               (const void*)(uintptr_t)rout_ptr, (size_t)n * ROUT_ROW,
                               cudaMemcpyDeviceToHost, cp));
  CHECK_CUDA(cudaStreamSynchronize(cp));
  /* The header states how many rows follow and in which format; the worker
   * reads it at the head of the routing slot before it touches the payload. */
  if (rout_ptr) {
    int32_t* hdr = (int32_t*)(uintptr_t)(rbase + rout_slot(L));
    float inv = inv_scale;
    int32_t inv_bits;
    memcpy(&inv_bits, &inv, sizeof(inv_bits));
    hdr[0] = s;
    hdr[1] = n;
    hdr[2] = inv_bits;
    hdr[3] = FMT_FP8;
    __sync_synchronize();
  }

  /* Routing first, unsignaled, same QP: RC ordering puts it ahead of the
   * payload and the flag, so the worker never applies stale routing. */
  if (rout_ptr) {
    ibv_sge sge{}; sge.addr = rbase + rout_slot(L);
    sge.length = (uint32_t)(ROUT_HDR + (size_t)n * ROUT_ROW);
    sge.lkey = rmr->lkey;
    ibv_send_wr wr{}; wr.wr_id = 0; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = 0;
    wr.wr.rdma.remote_addr = remote.rout_vaddr + rout_slot(L);
    wr.wr.rdma.rkey = remote.rout_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp, &wr, &bad)) DIE("post WRITE rout");
  }
  uint64_t pid = 0;
  {
    /* A single 40 MiB RDMA WRITE reaches only ~6.3 GB/s on this RC pair while
     * the RDMA READ of the same buffer reaches 11.6 GB/s, i.e. line rate of
     * the 100 GbE port. Splitting the payload into 4 MiB WRs, unsignalled
     * except the last, keeps several host-memory fetches in flight instead of
     * letting one huge descriptor chain serialise them. */
    const uint32_t kChunk = 4u << 20;
    const int nchunk = (int)((xbytes + kChunk - 1) / kChunk);
    if (nchunk > 16) DIE("post WRITE payload: too many chunks");
    ibv_send_wr wrs[16];
    ibv_sge sges[16];
    pid = take_wr_id();
    for (int i = 0; i < nchunk; i++) {
      const size_t off = (size_t)i * kChunk;
      const uint32_t len = (uint32_t)(xbytes - off < kChunk ? xbytes - off : kChunk);
      sges[i] = ibv_sge{};
      sges[i].addr = base + in_slot(L) + off;
      sges[i].length = len;
      sges[i].lkey = mr->lkey;
      wrs[i] = ibv_send_wr{};
      wrs[i].wr_id = (i + 1 == nchunk) ? pid : 0;
      wrs[i].sg_list = &sges[i];
      wrs[i].num_sge = 1;
      wrs[i].opcode = IBV_WR_RDMA_WRITE;
      wrs[i].send_flags = (i + 1 == nchunk) ? IBV_SEND_SIGNALED : 0;
      wrs[i].wr.rdma.remote_addr = remote.vaddr + in_slot(L) + off;
      wrs[i].wr.rdma.rkey = remote.rkey;
      wrs[i].next = (i + 1 < nchunk) ? &wrs[i + 1] : nullptr;
    }
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp, &wrs[0], &bad)) DIE("post WRITE payload");
  }
  {
    int32_t* pflags = (int32_t*)flags;
    pflags[in_flag(L) / 4] = s;
    __sync_synchronize();
    ibv_sge sge{}; sge.addr = fbase + in_flag(L); sge.length = 4; sge.lkey = fmr->lkey;
    ibv_send_wr wr{}; wr.wr_id = 0; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = 0;
    wr.wr.rdma.remote_addr = remote.flag_vaddr + in_flag(L);
    wr.wr.rdma.rkey = remote.flag_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp, &wr, &bad)) DIE("post WRITE flag");
  }
  /* Routing and flag are unsignaled; the payload's completion is the one thing
   * worth waiting for, and it also proves the NIC consumed the SQ. */
  poll_cq_tracked(*this, pid);
}

int Peer::wait_layer(int L, long long timeout_ms) {
  int32_t* pflags = (int32_t*)flags;
  const int32_t s = seq;
  int polls = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const uint64_t id = take_wr_id();
    ibv_sge sge{}; sge.addr = fbase + out_flag(L); sge.length = 4; sge.lkey = fmr->lkey;
    ibv_send_wr wr{}; wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote.flag_vaddr + out_flag(L);
    wr.wr.rdma.rkey = remote.flag_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp2, &wr, &bad)) DIE("post READ flag");
    poll_cq_tracked(*this, id);
    polls++;
    if (pflags[out_flag(L) / 4] >= s) break;
    if (timeout_ms > 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() >= timeout_ms) {
      polls_total += polls;
      layer_polls = polls;
      return -1;
    }
  }
  polls_total += polls;
  layer_polls = polls;
  return polls;
}

uint64_t Peer::read_layer(int L, int n) {
  if (n < 1 || n > rows) { fprintf(stderr, "read rows %d outside 1..%d\n", n, rows); exit(4); }
  const uint64_t id = take_wr_id();
  ibv_sge sge{}; sge.addr = base + out_slot(L);
  sge.length = (uint32_t)((size_t)n * K_HIDDEN * 2); sge.lkey = mr->lkey;
  ibv_send_wr wr{}; wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote.vaddr + out_slot(L);
  wr.wr.rdma.rkey = remote.rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(qp, &wr, &bad)) DIE("post READ out");
  return id;
}

void Peer::collect(uint64_t wr) { poll_cq_tracked(*this, wr); }

void Peer::close() {
  if (mfd >= 0) { char b = 1; xfer_all(mfd, &b, 1, 1); ::close(mfd); mfd = -1; }
  if (st) { cudaStreamDestroy(st); st = nullptr; }
  if (qp) { ibv_destroy_qp(qp); qp = nullptr; }
  if (qp2) { ibv_destroy_qp(qp2); qp2 = nullptr; }
  if (mr) { ibv_dereg_mr(mr); mr = nullptr; }
  if (fmr) { ibv_dereg_mr(fmr); fmr = nullptr; }
  if (rmr) { ibv_dereg_mr(rmr); rmr = nullptr; }
  if (buf) { cudaFreeHost(buf); buf = nullptr; }
  if (flags) { cudaFreeHost(flags); flags = nullptr; }
  if (rbuf) { cudaFreeHost(rbuf); rbuf = nullptr; }
  if (cq) { ibv_destroy_cq(cq); cq = nullptr; }
  if (pd) { ibv_dealloc_pd(pd); pd = nullptr; }
  if (ctx) { ibv_close_device(ctx); ctx = nullptr; }
}


/* ---- Python binding ---- */
typedef struct { PyObject_HEAD Peer* p; } PeerObj;

static PyObject* PeerObj_new(PyTypeObject* type, PyObject*, PyObject*) {
  PeerObj* self = (PeerObj*)type->tp_alloc(type, 0);
  if (self) self->p = new Peer();
  return (PyObject*)self;
}

static void PeerObj_dealloc(PeerObj* self) {
  if (self->p) { self->p->close(); delete self->p; self->p = nullptr; }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PeerObj_init(PeerObj* self, PyObject* args, PyObject* kwds) {
  static const char* kw[] = {"dev", "gid_idx", "mgmt_ip", "mgmt_port",
                             "layers", "rows", "iters", nullptr};
  const char* dev = nullptr; int gid_idx = 3; const char* mgmt_ip = nullptr;
  int mgmt_port = 47101, layers = NL, rows = 1, iters = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sis|iiii", (char**)kw, &dev, &gid_idx,
                                   &mgmt_ip, &mgmt_port, &layers, &rows, &iters))
    return -1;
  int rc = 0;
  Py_BEGIN_ALLOW_THREADS
  rc = self->p->open(dev, gid_idx, mgmt_ip, mgmt_port, layers, rows, iters);
  Py_END_ALLOW_THREADS
  if (rc) { PyErr_SetString(PyExc_RuntimeError, "peer open failed"); return -1; }
  return 0;
}

static PyObject* PeerObj_post(PeerObj* self, PyObject* args) {
  int L; unsigned long long x_ptr = 0, rout_ptr = 0; int n = -1; float inv = 1.0f;
  if (!PyArg_ParseTuple(args, "iK|Kif", &L, &x_ptr, &rout_ptr, &n, &inv)) return nullptr;
  if (n < 0) n = self->p->rows;
  Py_BEGIN_ALLOW_THREADS
  self->p->stage_and_post(L, x_ptr, rout_ptr, n, inv);
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject* PeerObj_wait(PeerObj* self, PyObject* args) {
  int L;
  if (!PyArg_ParseTuple(args, "i", &L)) return nullptr;
  int polls = 0;
  Py_BEGIN_ALLOW_THREADS
  polls = self->p->wait_layer(L);
  Py_END_ALLOW_THREADS
  return PyLong_FromLong(polls);
}

static PyObject* PeerObj_read(PeerObj* self, PyObject* args) {
  int L; int n = -1;
  if (!PyArg_ParseTuple(args, "i|i", &L, &n)) return nullptr;
  if (n < 0) n = self->p->rows;
  uint64_t id = 0;
  Py_BEGIN_ALLOW_THREADS
  id = self->p->read_layer(L, n);
  Py_END_ALLOW_THREADS
  return PyLong_FromUnsignedLongLong(id);
}

static PyObject* PeerObj_collect(PeerObj* self, PyObject* args) {
  unsigned long long wr;
  if (!PyArg_ParseTuple(args, "K", &wr)) return nullptr;
  Py_BEGIN_ALLOW_THREADS
  self->p->collect(wr);
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject* PeerObj_begin_round(PeerObj* self, PyObject*) {
  self->p->begin_round();
  return PyLong_FromLong(self->p->seq);
}

static PyObject* PeerObj_set_stream(PeerObj* self, PyObject* args) {
  unsigned long long stream = 0;
  if (!PyArg_ParseTuple(args, "K", &stream)) return nullptr;
  self->p->set_stream(stream);
  Py_RETURN_NONE;
}


/* Zero-copy view of a region of the registered slab. Host memory, so the caller
 * can wrap it with numpy and copy to the device. */
static PyObject* region_mv(Peer* p, size_t off, size_t nbytes) {
  return PyMemoryView_FromMemory((char*)(uintptr_t)(p->base + off), (Py_ssize_t)nbytes,
                                 PyBUF_WRITE);
}

static PyObject* PeerObj_out_mv(PeerObj* self, PyObject* args) {
  int L;
  if (!PyArg_ParseTuple(args, "i", &L)) return nullptr;
  return region_mv(self->p, out_slot(L), self->p->bytes);
}

static PyObject* PeerObj_in_mv(PeerObj* self, PyObject* args) {
  int L;
  if (!PyArg_ParseTuple(args, "i", &L)) return nullptr;
  return region_mv(self->p, in_slot(L), self->p->bytes);
}

static PyObject* PeerObj_rout_mv(PeerObj* self, PyObject* args) {
  int L;
  if (!PyArg_ParseTuple(args, "i", &L)) return nullptr;
  return PyMemoryView_FromMemory((char*)(uintptr_t)(self->p->rbase + rout_slot(L)),
                                 (Py_ssize_t)(ROUT_HDR + (size_t)self->p->rows * ROUT_ROW), PyBUF_WRITE);
}

static PyObject* PeerObj_close_py(PeerObj* self, PyObject*) {
  Py_BEGIN_ALLOW_THREADS
  self->p->close();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject* PeerObj_get_seq(PeerObj* self, void*) {
  return PyLong_FromLong(self->p->seq);
}
static PyObject* PeerObj_get_polls(PeerObj* self, void*) {
  return PyLong_FromLongLong(self->p->polls_total);
}
static PyObject* PeerObj_get_layer_polls(PeerObj* self, void*) {
  return PyLong_FromLongLong(self->p->layer_polls);
}

static PyGetSetDef PeerObj_getset[] = {
  {(char*)"seq", (getter)PeerObj_get_seq, nullptr, (char*)"last posted sequence number", nullptr},
  {(char*)"polls", (getter)PeerObj_get_polls, nullptr, (char*)"cumulative out-flag polls", nullptr},
  {(char*)"layer_polls", (getter)PeerObj_get_layer_polls, nullptr,
   (char*)"polls spent on the last wait()", nullptr},
  {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyMethodDef PeerObj_methods[] = {
  {"begin_round", (PyCFunction)PeerObj_begin_round, METH_NOARGS,
   "begin_round(): advance the round sequence; every post() until the next call\n"
   "carries it as its in-flag. One call per decode step."},
  {"post", (PyCFunction)PeerObj_post, METH_VARARGS,
   "post(layer, x_dev_ptr[, rout_dev_ptr[, n[, inv_scale]]]): stage the first n\n"
   "rows of x (n*K e4m3), the interleaved routing (n*48 bytes) and the payload\n"
   "scale into the registered regions, write {seq, n, 1/inv_scale, fmt=fp8} into\n"
   "the layer's routing header, and post routing, payload and in-flag. n defaults\n"
   "to the peer's row capacity; the caller scales x by 1/inv_scale before it\n"
   "casts, so the worker multiplies the rows back by inv_scale."},
  {"wait", (PyCFunction)PeerObj_wait, METH_VARARGS,
   "wait(layer): poll the peer's out-flag for the last posted sequence; returns polls."},
  {"read", (PyCFunction)PeerObj_read, METH_VARARGS,
   "read(layer[, n]): post an RDMA READ of the first n out-rows; returns a wr id.\n"
   "n defaults to the peer's row capacity."},
  {"collect", (PyCFunction)PeerObj_collect, METH_VARARGS,
   "collect(wr): wait for the completion of a read()."},
  {"out_mv", (PyCFunction)PeerObj_out_mv, METH_VARARGS, "out_mv(layer): view of the out-slot."},
  {"in_mv", (PyCFunction)PeerObj_in_mv, METH_VARARGS, "in_mv(layer): view of the in-slot."},
  {"rout_mv", (PyCFunction)PeerObj_rout_mv, METH_VARARGS, "rout_mv(layer): view of the routing slot."},
  {"set_stream", (PyCFunction)PeerObj_set_stream, METH_VARARGS,
   "set_stream(cuda_stream_ptr): run the post() staging copies on that stream, so they\n"
   "are ordered behind the kernel that produces the activation."},
  {"close", (PyCFunction)PeerObj_close_py, METH_NOARGS, "close(): tear down verbs objects."},
  {nullptr, nullptr, 0, nullptr},
};

static PyTypeObject PeerType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  .tp_name = "_qmesh_afd.Peer",
  .tp_basicsize = sizeof(PeerObj),
  .tp_dealloc = (destructor)PeerObj_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "One AFD-TP3 transport peer (one GX rank).",
  .tp_methods = PeerObj_methods,
  .tp_getset = PeerObj_getset,
  .tp_init = (initproc)PeerObj_init,
  .tp_new = PeerObj_new,
};

/* Device-driven seam: kernels, proxies, Seam object and binding. These are
 * textual includes so one nvcc command builds the whole extension. */
#include "qmesh_afd_seam.cu"
#include "qmesh_afd_seam_host.inc"
#include "qmesh_afd_seam_api.inc"
#include "qmesh_afd_seam_py.inc"

static PyMethodDef module_methods[] = {
  {"quant_test", (PyCFunction)seam_quant_test, METH_VARARGS,
   "quant_test(x, x_stride, n, out, scale_ptr[, stream]): amax+scale+e4m3 pack, no peers."},
  {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef moddef = {
  PyModuleDef_HEAD_INIT, "_qmesh_afd", "AFD-TP3 transport for the P7 runtime.", -1, module_methods,
};

PyMODINIT_FUNC PyInit__qmesh_afd(void) {
  if (PyType_Ready(&PeerType) < 0) return nullptr;
  PyObject* m = PyModule_Create(&moddef);
  if (!m) return nullptr;
  Py_INCREF(&PeerType);
  if (PyModule_AddObject(m, "Peer", (PyObject*)&PeerType) < 0) {
    Py_DECREF(&PeerType); Py_DECREF(m);
    return nullptr;
  }
  PyModule_AddIntConstant(m, "LAYERS", NL);
  PyModule_AddIntConstant(m, "K_HIDDEN", K_HIDDEN);
  PyModule_AddIntConstant(m, "TOPK", TOPK_ROUT);
  PyModule_AddIntConstant(m, "ROUT_ROW", (long)ROUT_ROW);
  PyModule_AddIntConstant(m, "SLAB", (long)SLAB);
  if (PyType_Ready(&SeamType) < 0) return nullptr;
  Py_INCREF(&SeamType);
  if (PyModule_AddObject(m, "Seam", (PyObject*)&SeamType) < 0) {
    Py_DECREF(&SeamType); Py_DECREF(m);
    return nullptr;
  }
  return m;
}
