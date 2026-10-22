/* qmesh_afd_edge.cpp
 *
 * Request/response RDMA transport for the AFD-TP3 expert workers.
 *
 * One-sided from the client's (P7) point of view. The GX server runs no host
 * code on the hot path: one kernel per layer spins on an in-flag that the NIC
 * writes into pinned host memory, copies the payload, then raises an out-flag
 * that the client pulls with a 4-byte RDMA READ.
 *
 *   client, layer L, seq:  WRITE payload -> server slot      (unsignaled)
 *                          WRITE in_flag[L] = seq            (unsignaled, RC-ordered after)
 *   server, layer L, seq:  mock_expert kernel: spin in_flag[L] >= seq,
 *                          compute, last block sets out_flag[L] = seq
 *   client, layer L, seq:  poll out_flag[L] with 4-byte RDMA READ into its own
 *                          pinned flag region, then READ out[L]
 *
 * Two QPs. The 4-byte flag READs ride a second RC QP: on one QP an RC READ is
 * ordered behind the in-flight 61 KiB payload READ, which raised the flag wait
 * from 5.2 to 16.9 us. Both QPs share one CQ; wr_ids are globally unique and
 * poll_cq_tracked remembers early completions, because a payload READ that
 * completes during a flag wait must not lose its CQE.
 *
 * In the pipelined round the client keeps one payload READ in flight across the
 * next layer's flag wait, so the READ's wire time is hidden behind compute.
 * Measured R=6, 40 layers, 61,440 B (per layer, no compute / +275 us mock):
 *   edge 20 P7<->GX2 16.96 / 287.3   edge 01 P7<->GX1 14.95 / 287.0
 *   edge 12 GX1<->GX2 14.22 / 287.2; payload READ hidden (read_us 0.13).
 *
 * Memory: the whole slab is mapped pinned host memory. ibv_reg_mr on cudaMalloc
 * memory fails here (no nvidia-peermem client: EFAULT on P7 and GB10) and
 * dmabuf export fails on GB10, so pinned host is the only registrable backing;
 * on GB10 host and device memory are the same DRAM, and on P7 the kernels read
 * and write it through the mapped device pointer.
 *
 * Target: <= 40 us serialized round trip at R=6 (61,440 B).
 * Build identical to qmesh_verbs_cuda_pingpong (scripts/build_local.sh):
 *   nvcc -O2 -std=c++17 -x cu <this> -o bin/qmesh_afd_edge.$(uname -m) \
 *        -libverbs -lcuda -lcudart
 *
 * Usage (pass the RDMA device and management IP explicitly; loopback shown):
 *   server: ./qmesh_afd_edge --role server --dev mlx5_0 --gid-idx 3 \
 *             --mgmt-ip 127.0.0.1 --mgmt-port 47100 --layers 40 --rows 6 --iters 20
 *   client: ./qmesh_afd_edge --role client --dev mlx5_0 --gid-idx 3 \
 *             --mgmt-ip 127.0.0.1 --mgmt-port 47100 --layers 40 --rows 6 --iters 20
 */
#include <algorithm>
#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <set>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(2); \
  } \
} while (0)

#define CHECK_CU(call) do { \
  CUresult _e = (call); \
  if (_e != CUDA_SUCCESS) { \
    const char* s=nullptr; cuGetErrorString(_e,&s); \
    fprintf(stderr, "CU error %s:%d: %s\n", __FILE__, __LINE__, s?s:"?"); \
    exit(2); \
  } \
} while (0)

#define DIE(msg) do { fprintf(stderr, "FATAL: %s (errno=%d %s)\n", msg, errno, strerror(errno)); exit(1); } while (0)

/* ---- payload slab (device memory), identical layout on both sides ---- */
static constexpr int    NL        = 40;
static constexpr int    K_HIDDEN  = 5120;
/* Row capacity, mirrored as R_MAX in qmesh_afd_peer.cpp and WIRE_ROWS in the
 * GX worker. The post's own valid-row count rides in the routing header, so
 * this only sizes the slots.
 *
 * One in-slot and one out-slot serve every layer: the client posts layer L+1
 * only after it saw L's out-flag, and this side raises that flag after the
 * worker consumed L's payload, so a single buffer pair is enough and keeps the
 * shm region at 2 slots instead of 2*NL (which no longer fits beside the
 * resident expert arena at 4096 rows). The flag plane stays per layer. */
static constexpr int    R_MAX     = 4096;
static constexpr size_t SLOT      = (size_t)R_MAX * K_HIDDEN * 2;   /* 41,943,040 B */
static constexpr size_t IN_OFF    = 0;
static constexpr size_t OUT_OFF   = SLOT;
static constexpr size_t SLAB      = 2 * SLOT;

/* ---- flag region (pinned host, registered), same offsets on both sides ---- */
static constexpr size_t FSTRIDE   = 64;
static constexpr size_t FIN_OFF   = 0;
static constexpr size_t FOUT_OFF  = NL * FSTRIDE;
static constexpr size_t FLAGS     = 2 * NL * FSTRIDE;

static inline size_t in_slot(int)     { return IN_OFF; }
static inline size_t out_slot(int)    { return OUT_OFF; }
static inline size_t in_flag(int L)  { return FIN_OFF + (size_t)L * FSTRIDE; }
static inline size_t out_flag(int L) { return FOUT_OFF + (size_t)L * FSTRIDE; }

/* ---- routing plane (pinned host, registered), same offsets on both sides ----
 * One 16 B header (int32 seq, int32 n, 8 B pad) then ids[n][TOPK] int32 and
 * tw[n][TOPK] f32, so header + rows*48 B is one contiguous transfer, reused by
 * every layer under the same serialization as the payload pair.
 * The router lives on the attention host, so the ids and the top-k weights must
 * cross the wire with the activation they apply to. */
static constexpr int    TOPK_ROUT = 6;
static constexpr size_t ROUT_ROW  = (size_t)TOPK_ROUT * 4 * 2;      /* ids + tw */
static constexpr size_t ROUT_HDR  = 16;
static constexpr size_t ROUT_SLOT = ROUT_HDR + (size_t)R_MAX * ROUT_ROW;
static constexpr size_t ROUT      = ROUT_SLOT;
static inline size_t rout_slot(int) { return 0; }

/* sequence bases: the server pre-enqueues the whole schedule, so both sides
 * must derive it identically from (layers, iters). */
static constexpr uint32_t SEQ_SER  = 100;
static constexpr uint32_t SEQ_PIPE = 1000;

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

/* Set once from the peer's ConnMeta; both QPs (data and flag plane) share the
 * same address vector. */
static uint8_t g_peer_gid[16];
static double now_s() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int mgmt_listen(const char* ip, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) DIE("socket");
  int one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) DIE("inet_pton");
  if (bind(fd, (sockaddr*)&a, sizeof(a)) < 0) DIE("bind");
  if (listen(fd, 1) < 0) DIE("listen");
  fprintf(stderr, "mgmt listening on %s:%d\n", ip, port);
  int c = accept(fd, nullptr, nullptr);
  if (c < 0) DIE("accept");
  close(fd);
  return c;
}

static int mgmt_connect(const char* ip, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) DIE("socket");
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) DIE("inet_pton");
  for (int i = 0; i < 300; i++) {
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

struct Ctx {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_qp* qp = nullptr;        /* data plane: payload writes and reads */
  ibv_cq* cq = nullptr;        /* shared by both QPs; wr_ids are globally unique */
  ibv_qp* qp2 = nullptr;       /* flag plane: 4 B out-flag reads, never queued behind a payload */
  ibv_mr* mr = nullptr;        /* pinned host payload slab */
  ibv_mr* fmr = nullptr;       /* pinned host flags */
  ibv_mr* rmr = nullptr;       /* pinned host routing plane */
  ibv_port_attr port_attr{};
  int port = 1, gid_idx = 3;
  void* buf = nullptr;         /* host VA of slab */
  void* dbuf = nullptr;        /* device-visible VA of slab (UVA: == buf) */
  void* flags = nullptr;
  void* rbuf = nullptr;        /* host VA of routing plane */
  uint64_t base = 0, fbase = 0, rbase = 0;
};
static void open_device(Ctx& c, const char* dev_name) {
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
  c.ctx = ibv_open_device(dev);
  if (!c.ctx) DIE("ibv_open_device");
  if (ibv_query_port(c.ctx, c.port, &c.port_attr)) DIE("ibv_query_port");
  fprintf(stderr, "dev=%s state=%d mtu=%d lid=%u\n", dev_name, c.port_attr.state,
          c.port_attr.active_mtu, c.port_attr.lid);
  c.pd = ibv_alloc_pd(c.ctx); if (!c.pd) DIE("ibv_alloc_pd");
  c.cq = ibv_create_cq(c.ctx, 8192, nullptr, nullptr, 0); if (!c.cq) DIE("ibv_create_cq");
  ibv_free_device_list(list);
}

static void create_one_qp(Ctx& c, ibv_qp** out, const char* tag) {
  ibv_qp_init_attr attr{};
  attr.send_cq = c.cq; attr.recv_cq = c.cq;
  attr.cap.max_send_wr = 512; attr.cap.max_recv_wr = 64;
  attr.cap.max_send_sge = 1;  attr.cap.max_recv_sge = 1;
  attr.cap.max_inline_data = 64;
  attr.qp_type = IBV_QPT_RC;
  *out = ibv_create_qp(c.pd, &attr);
  if (!*out) { attr.cap.max_inline_data = 0; *out = ibv_create_qp(c.pd, &attr); }
  if (!*out) DIE("ibv_create_qp");
  fprintf(stderr, "qpn[%s]=%u max_inline=%u\n", tag, (*out)->qp_num, attr.cap.max_inline_data);
}

static void create_qps(Ctx& c) {
  create_one_qp(c, &c.qp, "data");
  create_one_qp(c, &c.qp2, "flag");
}

/* Serve mode: the payload slab and the flag region live in /dev/shm so a
 * separate compute process (the Python worker) can drive them. Only the C++
 * side talks verbs; the worker reads the in-flags and raises the out-flags in
 * the same pages, which ibv_reg_mr pins anyway. */
static char g_shm[256] = {0};
static int g_serve = 0;

static void* shm_map(const char* suffix, size_t bytes) {
  char path[300];
  snprintf(path, sizeof(path), "%s%s", g_shm, suffix);
  int fd = open(path, O_RDWR | O_CREAT, 0666);
  if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(5); }
  if (ftruncate(fd, (off_t)bytes)) { fprintf(stderr, "ftruncate %s\n", path); exit(5); }
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) { fprintf(stderr, "mmap %s: %s\n", path, strerror(errno)); exit(5); }
  close(fd);
  fprintf(stderr, "shm %s %.2fMiB va=0x%lx\n", path, bytes / 1048576.0, (unsigned long)(uintptr_t)p);
  return p;
}

static void alloc_and_reg(Ctx& c) {
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (g_shm[0]) {
    c.buf = shm_map(".slab", SLAB);
    c.flags = shm_map(".flags", FLAGS);
    c.rbuf = shm_map(".rout", ROUT);
  } else {
    CHECK_CUDA(cudaSetDevice(0));
    /* The payload slab lives in mapped pinned host memory.
     *
     * cudaMalloc memory cannot be pinned by ibv_reg_mr here (no nvidia-peermem
     * peer-memory client: EFAULT on P7 and on GB10), and dmabuf export fails too
     * (GB10: CUDA_ERROR_INVALID_VALUE). Pinned host is the one MR type both edges
     * accept, and on GB10 host and device memory are the same DRAM, so the
     * zero-copy device pointer is what the kernels use. */
    CHECK_CUDA(cudaHostAlloc(&c.buf, SLAB, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&c.flags, FLAGS, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&c.rbuf, ROUT, cudaHostAllocMapped));
  }
  /* Session epoch: the worker may open the region before this process starts,
   * so bump a counter in the unused word of layer 0's flag stride. The worker
   * snapshots it at startup and ignores everything until it changes.
   *
   * Payload, routing and flags are all scratch for the new epoch. Clearing the
   * flag plane here (including the worker's quiescent-ACK word) means a worker
   * cannot mistake a dead round's in-flag for the new epoch's work, and a client
   * cannot mistake an old ACK for the worker having stopped on its epoch. The
   * epoch and ACK words are explicitly excluded from the clear: zeroing them
   * would expose a transient epoch 0, and a client starting in that window would
   * derive a lower epoch (1) than the worker's last one. The rest of the plane is
   * cleared, then ACK is zeroed, fenced, and only then is the new epoch
   * published: a worker that observes the new epoch also observes the cleared
   * flags. (The client's connect handshake performs the same clearing with its
   * own epoch bump; see qmesh_afd_peer.cpp.) */
  int32_t* sess = (int32_t*)((char*)c.flags + 4);
  int32_t* ack = (int32_t*)((char*)c.flags + 8);
  const int32_t e_old = *sess;
  const int32_t epoch = (e_old >= 0 && e_old < 0x7fffffff) ? e_old + 1 : 1;
  memset(c.buf, 0, SLAB);
  memset(c.rbuf, 0, ROUT);
  memset(c.flags, 0, 4);                         /* layer-0 in flag */
  memset((char*)c.flags + 12, 0, FLAGS - 12);    /* everything after epoch/ACK */
  *ack = 0;
  __sync_synchronize();
  *sess = epoch;
  if (!g_serve) CHECK_CUDA(cudaHostGetDevicePointer(&c.dbuf, c.buf, 0));
  double t0 = now_s();
  c.base = (uint64_t)(uintptr_t)c.buf;
  c.fbase = (uint64_t)(uintptr_t)c.flags;
  c.rbase = (uint64_t)(uintptr_t)c.rbuf;
  c.mr = ibv_reg_mr(c.pd, c.buf, SLAB, access);
  if (!c.mr) { fprintf(stderr, "ibv_reg_mr(slab) errno=%d %s\n", errno, strerror(errno)); exit(5); }
  c.fmr = ibv_reg_mr(c.pd, c.flags, FLAGS, access);
  if (!c.fmr) { fprintf(stderr, "ibv_reg_mr(flags) errno=%d %s\n", errno, strerror(errno)); exit(5); }
  c.rmr = ibv_reg_mr(c.pd, c.rbuf, ROUT, access);
  if (!c.rmr) { fprintf(stderr, "ibv_reg_mr(rout) errno=%d %s\n", errno, strerror(errno)); exit(5); }
  fprintf(stderr, "reg slab=%.2fMiB va=0x%lx iova=0x%lx lkey=%u rkey=%u | flags=%zuB rkey=%u | "
                  "rout=%zuB rkey=%u | reg_ms=%.2f\n",
          SLAB / 1048576.0, (unsigned long)c.base, (unsigned long)c.mr->addr, c.mr->lkey,
          c.mr->rkey, FLAGS, c.fmr->rkey, ROUT, c.rmr->rkey, (now_s() - t0) * 1e3);
}

static void qp_to_init(Ctx& c, ibv_qp* qp) {
  ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = c.port;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    DIE("modify INIT");
}

static void qp_to_rtr(Ctx& c, ibv_qp* qp, uint32_t dqpn, uint32_t dpsn, uint8_t dlid) {
  ibv_qp_attr a{}; memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR; a.path_mtu = c.port_attr.active_mtu;
  a.dest_qp_num = dqpn; a.rq_psn = dpsn;
  a.max_dest_rd_atomic = 4; a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1; a.ah_attr.dlid = dlid; a.ah_attr.sl = 0;
  a.ah_attr.src_path_bits = 0; a.ah_attr.port_num = c.port;
  memcpy(&a.ah_attr.grh.dgid, g_peer_gid, 16);
  a.ah_attr.grh.sgid_index = c.gid_idx;
  a.ah_attr.grh.hop_limit = 16; a.ah_attr.grh.traffic_class = 0;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    DIE("modify RTR");
}

static void qp_to_rts(Ctx& c, ibv_qp* qp, uint32_t psn) {
  (void)c;
  ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTS; a.sq_psn = psn;
  a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.max_rd_atomic = 4;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
    DIE("modify RTS");
}

/* Every signaled post gets a unique id from this counter; a completion is only
 * consumed when its id matches. Signaled sends whose CQE is never reaped would
 * otherwise be returned by the next poll, making a later read look complete
 * before its data arrived (measured: a 61 KiB RDMA READ "in 0.28 us"). */
static uint64_t g_wr_id = 1;
static uint64_t take_wr_id() { return g_wr_id++; }

static void poll_cq_id(Ctx& c, uint64_t wr_id, const char* tag) {
  ibv_wc wc{};
  for (;;) {
    int k = ibv_poll_cq(c.cq, 1, &wc);
    if (k < 0) DIE("poll_cq");
    if (k == 0) continue;
    if (wc.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "%s wc status=%s opcode=%d vendor=0x%x wr_id=%lu want=%lu\n", tag,
              ibv_wc_status_str(wc.status), wc.opcode, wc.vendor_err,
              (unsigned long)wc.wr_id, (unsigned long)wr_id);
      exit(9);
    }
    if (wc.wr_id == wr_id) return;
  }
}

/* Same, but remembers completions instead of discarding them. Completions are in
 * SQ order, so a payload READ posted earlier can complete while we are waiting
 * for a later flag READ; that CQE must be kept for its own waiter. `done` holds
 * ids that arrived early and is emptied as each waiter claims its own. */
static void poll_cq_tracked(Ctx& c, uint64_t want, std::set<uint64_t>& done, const char* tag) {
  if (done.erase(want)) return;
  for (;;) {
    ibv_wc wc{};
    int k = ibv_poll_cq(c.cq, 1, &wc);
    if (k < 0) DIE("poll_cq");
    if (k == 0) continue;
    if (wc.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "%s wc status=%s opcode=%d vendor=0x%x wr_id=%lu want=%lu\n", tag,
              ibv_wc_status_str(wc.status), wc.opcode, wc.vendor_err,
              (unsigned long)wc.wr_id, (unsigned long)want);
      exit(9);
    }
    if (wc.wr_id == want) return;
    done.insert(wc.wr_id);
  }
}

static uint64_t post_write_from_dev(Ctx& c, size_t local_off, uint64_t raddr, uint32_t rkey,
                                    uint32_t bytes, int signaled) {
  uint64_t id = signaled ? take_wr_id() : 0;
  ibv_sge sge{}; sge.addr = c.base + local_off; sge.length = bytes; sge.lkey = c.mr->lkey;
  ibv_send_wr wr{}; memset(&wr, 0, sizeof(wr));
  wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(c.qp, &wr, &bad)) { fprintf(stderr, "post WRITE errno=%d %s\n", errno, strerror(errno)); exit(10); }
  return id;
}

static uint64_t post_write_flag(Ctx& c, size_t local_off, uint64_t raddr, uint32_t rkey,
                                uint32_t bytes, int signaled) {
  uint64_t id = signaled ? take_wr_id() : 0;
  ibv_sge sge{}; sge.addr = c.fbase + local_off; sge.length = bytes; sge.lkey = c.fmr->lkey;
  ibv_send_wr wr{}; memset(&wr, 0, sizeof(wr));
  wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(c.qp, &wr, &bad)) { fprintf(stderr, "post WRITE flag errno=%d %s\n", errno, strerror(errno)); exit(10); }
  return id;
}

static void post_write_rout(Ctx& c, size_t local_off, uint64_t raddr, uint32_t rkey,
                            uint32_t bytes) {
  ibv_sge sge{}; sge.addr = c.rbase + local_off; sge.length = bytes; sge.lkey = c.rmr->lkey;
  ibv_send_wr wr{}; memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = 0;   /* unsignaled: ordered ahead of the payload on the same QP */
  wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(c.qp, &wr, &bad)) { fprintf(stderr, "post WRITE rout errno=%d %s\n", errno, strerror(errno)); exit(10); }
}

static uint64_t post_read_to_dev(Ctx& c, size_t local_off, uint64_t raddr, uint32_t rkey,
                                 uint32_t bytes) {
  uint64_t id = take_wr_id();
  ibv_sge sge{}; sge.addr = c.base + local_off; sge.length = bytes; sge.lkey = c.mr->lkey;
  ibv_send_wr wr{}; memset(&wr, 0, sizeof(wr));
  wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(c.qp, &wr, &bad)) { fprintf(stderr, "post READ errno=%d %s\n", errno, strerror(errno)); exit(10); }
  return id;
}

static uint64_t post_read_flag(Ctx& c, size_t local_off, uint64_t raddr, uint32_t rkey) {
  uint64_t id = take_wr_id();
  /* Flag plane: a 4 B READ on its own QP is never queued behind an in-flight
   * 61 KiB payload READ, so the out-flag poll keeps full latency while the
   * payload read is still on the wire. */
  ibv_sge sge{}; sge.addr = c.fbase + local_off; sge.length = 4; sge.lkey = c.fmr->lkey;
  ibv_send_wr wr{}; memset(&wr, 0, sizeof(wr));
  wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(c.qp2, &wr, &bad)) { fprintf(stderr, "post READ flag errno=%d %s\n", errno, strerror(errno)); exit(10); }
  return id;
}

/* Expert-worker stand-in: spin on the NIC-written in-flag (pinned host memory),
 * copy the payload, then raise the out-flag once every block has finished.
 *
 * The wait has to live inside the kernel. cuStreamWaitValue32 on host memory
 * costs ~30 us per node here (measured: 40 layers took 1.28 ms with nothing to
 * copy), which swamps the 5-13 us wire time; a kernel-side spin sees the flag in
 * ~1 us.
 *
 * Completion uses a device-memory counter, not a host one: system-scope atomics
 * on pinned host memory serialize at ~1 us each, so 40 blocks took ~40 us per
 * layer (measured: 40.0 us wait per layer with transport at 40.7 us). Device
 * atomics cost ~100 ns; only the last block crosses to host, with one system
 * fence and one 4 B store of the out-flag.
 * The counter is never reset: `expect` is the cumulative block count of this
 * kernel in the whole schedule, so no round trip is needed to clear it.
 *
 * `spin` delays the copy by that many clock cycles for every block, modelling
 * expert compute. */
__global__ void mock_expert(const uint2* __restrict__ in, uint2* __restrict__ out,
                            volatile int32_t* in_flag, volatile int32_t* out_flag,
                            int32_t* done, int n8, int seq, int expect, long long spin) {
  if (threadIdx.x == 0) { while (*in_flag < seq) {} }
  __syncthreads();
  if (spin > 0) {
    long long t0 = clock64();
    while (clock64() - t0 < spin) {}
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n8) out[i] = in[i];
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    int prev = atomicAdd(done, 1);
    if (prev + 1 == expect) {
      __threadfence_system();
      *out_flag = seq;
    }
  }
}

static void launch_mock(cudaStream_t st, uint64_t base, int L, uint32_t bytes, long long spin,
                        volatile int32_t* in_flag, volatile int32_t* out_flag, int32_t* done,
                        int seq, int expect) {
  int n8 = (int)(bytes / 8);
  int threads = 256, blocks = (n8 + threads - 1) / threads;
  mock_expert<<<blocks, threads, 0, st>>>(
      (const uint2*)(uintptr_t)(base + in_slot(L)),
      (uint2*)(uintptr_t)(base + out_slot(L)), in_flag, out_flag, done, n8, seq, expect, spin);
}

static void fill_pattern(uint8_t* h, size_t n, uint32_t seed) {
  for (size_t i = 0; i < n; i++) h[i] = (uint8_t)((i * 131u + seed) & 0xff);
}

/* SM clock calibration: spin a known number of cycles. */
__global__ void clock_spin(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}

int main(int argc, char** argv) {
  const char* role = "server";
  const char* dev = nullptr;
  const char* mgmt_ip = nullptr;
  int mgmt_port = 47100, gid_idx = 3, port = 1;
  int layers = NL, rows = 6, iters = 20;
  double mock_us = 0.0;
  int verify = 1;

  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* n) { if (i + 1 >= argc) { fprintf(stderr, "missing %s\n", n); exit(1);} return argv[++i]; };
    if      (!strcmp(argv[i], "--role"))      role = need("role");
    else if (!strcmp(argv[i], "--dev"))       dev = need("dev");
    else if (!strcmp(argv[i], "--mgmt-ip"))   mgmt_ip = need("ip");
    else if (!strcmp(argv[i], "--mgmt-port")) mgmt_port = atoi(need("port"));
    else if (!strcmp(argv[i], "--gid-idx"))   gid_idx = atoi(need("gid"));
    else if (!strcmp(argv[i], "--port"))      port = atoi(need("port"));
    else if (!strcmp(argv[i], "--layers"))    layers = atoi(need("layers"));
    else if (!strcmp(argv[i], "--rows"))      rows = atoi(need("rows"));
    else if (!strcmp(argv[i], "--iters"))     iters = atoi(need("iters"));
    else if (!strcmp(argv[i], "--mock-us"))   mock_us = atof(need("us"));
    else if (!strcmp(argv[i], "--shm"))       snprintf(g_shm, sizeof(g_shm), "%s", need("prefix"));
    else if (!strcmp(argv[i], "--serve"))     g_serve = 1;
    else if (!strcmp(argv[i], "--no-verify")) verify = 0;
    else { fprintf(stderr, "unknown arg %s\n", argv[i]); exit(1); }
  }
  if (!dev) { fprintf(stderr, "--dev is required\n"); exit(1); }
  if (!mgmt_ip) { fprintf(stderr, "--mgmt-ip is required\n"); exit(1); }
  if (layers < 1 || layers > NL) { fprintf(stderr, "layers 1..%d\n", NL); exit(1); }
  if (rows < 1 || rows > R_MAX) { fprintf(stderr, "rows 1..%d\n", R_MAX); exit(1); }
  if (iters < 0 || (iters < 1 && !g_serve)) { fprintf(stderr, "iters >= 1 (or 0 with --serve)\n"); exit(1); }
  const uint32_t bytes = (uint32_t)rows * K_HIDDEN * 2;
  const bool is_client = !strcmp(role, "client");

  fprintf(stderr, "=== qmesh_afd_edge === role=%s dev=%s gid=%d mgmt=%s:%d layers=%d rows=%d "
                  "bytes=%u mock_us=%.1f iters=%d\n",
          role, dev, gid_idx, mgmt_ip, mgmt_port, layers, rows, bytes, mock_us, iters);

  Ctx c; c.port = port; c.gid_idx = gid_idx;
  double t0 = now_s();
  open_device(c, dev);
  create_qps(c);
  alloc_and_reg(c);
  qp_to_init(c, c.qp);
  qp_to_init(c, c.qp2);

  union ibv_gid lg{};
  if (ibv_query_gid(c.ctx, c.port, c.gid_idx, &lg)) DIE("query_gid");

  uint32_t my_psn = (uint32_t)(lrand48() & 0xffffff);
  uint32_t my_psn2 = (uint32_t)(lrand48() & 0xffffff);
  ConnMeta local{}; local.lid = c.port_attr.lid; local.qpn = c.qp->qp_num; local.psn = my_psn;
  local.qpn2 = c.qp2->qp_num; local.psn2 = my_psn2;
  local.rkey = c.mr->rkey;
  local.vaddr = (uint64_t)(uintptr_t)c.mr->addr;
  memcpy(local.gid, lg.raw, 16);
  local.size = (uint32_t)SLAB; local.gid_idx = (uint32_t)gid_idx;
  local.mtu = (uint32_t)c.port_attr.active_mtu;
  local.flag_rkey = c.fmr->rkey; local.flag_vaddr = c.fbase;
  local.rout_rkey = c.rmr->rkey; local.rout_vaddr = (uint64_t)(uintptr_t)c.rmr->addr;
  local.layers = (uint32_t)layers; local.iters = (uint32_t)iters;

  int mfd = is_client ? mgmt_connect(mgmt_ip, mgmt_port) : mgmt_listen(mgmt_ip, mgmt_port);
  ConnMeta remote{};
  if (is_client) { xfer_all(mfd, &remote, sizeof(remote), 0); xfer_all(mfd, &local, sizeof(local), 1); }
  else           { xfer_all(mfd, &local, sizeof(local), 1);  xfer_all(mfd, &remote, sizeof(remote), 0); }
  fprintf(stderr, "peer qpn=%u/%u rkey=%u base=0x%lx flag_rkey=%u flag_base=0x%lx "
                  "rout_rkey=%u rout_base=0x%lx\n",
          remote.qpn, remote.qpn2, remote.rkey, (unsigned long)remote.vaddr, remote.flag_rkey,
          (unsigned long)remote.flag_vaddr, remote.rout_rkey, (unsigned long)remote.rout_vaddr);
  if (remote.size != SLAB) { fprintf(stderr, "slab mismatch\n"); exit(11); }
  /* Layers must match (the slot table is shared); iters only has to match for
   * the benchmark's fixed schedule. Serve mode is a relay, so a runtime client
   * brings its own sequence and may use --iters 0 there. */
  if (remote.layers != (uint32_t)layers ||
      (!g_serve && remote.iters != (uint32_t)iters)) {
    fprintf(stderr, "schedule mismatch peer layers=%u iters=%u\n", remote.layers, remote.iters);
    exit(11);
  }
  memcpy(g_peer_gid, remote.gid, 16);
  qp_to_rtr(c, c.qp, remote.qpn, remote.psn, (uint8_t)remote.lid);
  qp_to_rts(c, c.qp, my_psn);
  qp_to_rtr(c, c.qp2, remote.qpn2, remote.psn2, (uint8_t)remote.lid);
  qp_to_rts(c, c.qp2, my_psn2);
  char b = 1; xfer_all(mfd, &b, 1, 1); xfer_all(mfd, &b, 1, 0);

  /* Serve mode: the compute process owns the flags. Stay up until the client
   * finishes its schedule, then return. */
  if (g_serve) {
    fprintf(stderr, "SERVE READY shm=%s slab=0x%lx flags=0x%lx rout=0x%lx\n", g_shm,
            (unsigned long)c.base, (unsigned long)c.fbase, (unsigned long)c.rbase);
    fflush(stderr);
    xfer_all(mfd, &b, 1, 0);
    fprintf(stderr, "serve: client done\n");
    return 0;
  }

  /* CUDA 13 dropped cudaDeviceProp::clockRate, so measure the SM clock directly:
   * the mock kernel spins a known number of clock64() cycles. */
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  clock_spin<<<1, 1>>>(1000000LL);
  CHECK_CUDA(cudaDeviceSynchronize());
  double ct0 = now_s();
  clock_spin<<<1, 1>>>(100000000LL);
  CHECK_CUDA(cudaDeviceSynchronize());
  const double clock_ghz = 100e6 / (now_s() - ct0) / 1e9;
  const long long spin = (long long)(mock_us * clock_ghz * 1e3);
  fprintf(stderr, "gpu=%s sm_%d%d clock=%.2fGHz spin_cycles=%lld\n",
          prop.name, prop.major, prop.minor, clock_ghz, spin);

  cudaStream_t st;
  CHECK_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

  int32_t* pflags = (int32_t*)c.flags;

  if (!is_client) {
    /* Pre-enqueue the entire schedule: serialized rounds, then the pipeline round.
     * One node per layer: mock_expert spins on the in-flag itself and raises the
     * out-flag after its last block, so there is no per-layer graph wait node. */
    int n8 = (int)(bytes / 8);
    int blocks = (n8 + 255) / 256;
    int32_t* done = nullptr;
    CHECK_CUDA(cudaMalloc(&done, 4));
    CHECK_CUDA(cudaMemsetAsync(done, 0, 4, st));
    uint64_t dbuf = (uint64_t)(uintptr_t)c.dbuf;
    double te0 = now_s();
    long long idx = 0;
    for (int r = 0; r < iters; r++)
      for (int L = 0; L < layers; L++) {
        uint32_t seq = SEQ_SER + (uint32_t)r;
        launch_mock(st, dbuf, L, bytes, spin, (int32_t*)((char*)c.flags + in_flag(L)),
                    (int32_t*)((char*)c.flags + out_flag(L)), done, (int)seq,
                    (int)(++idx * blocks));
      }
    for (int L = 0; L < layers; L++)
      launch_mock(st, dbuf, L, bytes, spin, (int32_t*)((char*)c.flags + in_flag(L)),
                  (int32_t*)((char*)c.flags + out_flag(L)), done, SEQ_PIPE,
                  (int)(++idx * blocks));
    fprintf(stderr, "schedule enqueued: %d nodes in %.1f us (%.2f us/layer)\n",
            (iters + 1) * layers, (now_s() - te0) * 1e6,
            (now_s() - te0) * 1e6 / ((iters + 1) * layers));
    CHECK_CUDA(cudaStreamSynchronize(st));
    fprintf(stderr, "server: schedule drained after %.2f ms (includes client time)\n", (now_s() - te0) * 1e3);
    xfer_all(mfd, &b, 1, 0);
  } else {
    std::vector<uint8_t> pat(bytes);
    fill_pattern(pat.data(), bytes, 0xABCDEFu);
    for (int L = 0; L < layers; L++)
      CHECK_CUDA(cudaMemcpy((uint8_t*)c.buf + in_slot(L), pat.data(), bytes, cudaMemcpyHostToDevice));

    /* Routing plane: ids/weights per row, synthetic and deterministic in
     * (layer,row) so the worker can compare what it received against what was
     * sent. The real router on the attention host fills the same layout. */
    auto fill_rout = [&](int L) {
      int32_t* ids = (int32_t*)((uint8_t*)c.rbuf + rout_slot(L));
      float* rtw = (float*)(ids + rows * TOPK_ROUT);
      for (int r = 0; r < rows; r++)
        for (int k = 0; k < TOPK_ROUT; k++) {
          ids[r * TOPK_ROUT + k] = (int32_t)((L * 13 + r * 7 + k * 31) % 384);
          rtw[r * TOPK_ROUT + k] = 1.0f / TOPK_ROUT;
        }
    };
    auto send_payload = [&](int L, int signaled) -> uint64_t {
      fill_rout(L);
      /* Same QP as the payload and the flag, so RC ordering puts this ahead of
       * both: the worker never sees a flag before the routing it belongs to. */
      post_write_rout(c, rout_slot(L), remote.rout_vaddr + rout_slot(L), remote.rout_rkey,
                      (uint32_t)(rows * ROUT_ROW));
      return post_write_from_dev(c, in_slot(L), remote.vaddr + in_slot(L), remote.rkey, bytes, signaled);
    };
    /* signaled=0 for the pipelined round: nobody waits on those completions, and
     * an unawaited id would sit in the tracker until the process exits. Delivery
     * does not depend on signaling. */
    auto send_flag = [&](int L, uint32_t seq, int signaled) -> uint64_t {
      pflags[in_flag(L) / 4] = (int32_t)seq;
      __sync_synchronize();
      return post_write_flag(c, in_flag(L), remote.flag_vaddr + in_flag(L), remote.flag_rkey, 4,
                             signaled);
    };
    /* Poll the peer's out-flag with 4 B RDMA READs on the flag QP. `done` holds
     * completions that arrived early, so an in-flight payload READ can complete
     * during a flag wait without its CQE being lost. */
    std::set<uint64_t> done;
    auto wait_out = [&](int L, uint32_t seq) -> int {
      int polls = 0;
      for (;;) {
        poll_cq_tracked(c, post_read_flag(c, out_flag(L), remote.flag_vaddr + out_flag(L),
                                          remote.flag_rkey), done, "read-flag");
        polls++;
        if (pflags[out_flag(L) / 4] >= (int32_t)seq) return polls;
      }
    };
    auto read_out = [&](int L) {
      uint64_t id = post_read_to_dev(c, out_slot(L), remote.vaddr + out_slot(L), remote.rkey, bytes);
      poll_cq_tracked(c, id, done, "read-out");
    };

    /* A. one-way write latency (payload only, no flag) */
    {
      for (int i = 0; i < 5; i++) poll_cq_id(c, send_payload(layers - 1, 1), "warm");
      const int N = 20; double acc = 0;
      for (int i = 0; i < N; i++) {
        double a = now_s();
        uint64_t id = send_payload(layers - 1, 1);
        poll_cq_id(c, id, "write");
        acc += now_s() - a;
      }
      double us = acc / N * 1e6;
      printf("RESULT oneway_write bytes=%u us=%.2f gbps=%.1f\n", bytes, us, bytes * 8.0 / (us * 1e-6) / 1e9);
      fflush(stdout);
    }

    /* B. serialized round trip: send one layer, wait for its flag, read it back */
    {
      std::vector<double> lat, pollus;
      long npolls = 0;
      for (int r = 0; r < iters; r++) {
        const uint32_t seq = SEQ_SER + (uint32_t)r;
        for (int L = 0; L < layers; L++) {
          double a = now_s();
          send_payload(L, 0);
          poll_cq_id(c, send_flag(L, seq, 1), "send");
          double w0 = now_s();
          int polls = wait_out(L, seq);
          double w1 = now_s();
          npolls += polls;
          read_out(L);
          lat.push_back(now_s() - a);
          pollus.push_back((w1 - w0) * 1e6);
        }
      }
      std::sort(lat.begin(), lat.end());
      double sum = 0; for (double v : lat) sum += v;
      printf("RESULT serialized_rt layers=%d iters=%d n=%zu mean_us=%.2f p50=%.2f p95=%.2f max=%.2f "
             "polls_per_layer=%.2f\n",
             layers, iters, lat.size(), sum / lat.size() * 1e6, lat[lat.size() / 2] * 1e6,
             lat[(size_t)(lat.size() * 0.95)] * 1e6, lat.back() * 1e6,
             (double)npolls / lat.size());
      fflush(stdout);
    }

    /* C. pipeline: post all 40 layers, then collect in order. One 61 KiB payload
     * READ stays in flight while the next layer's out-flag is awaited; without
     * that overlap the READ adds its full wire time on top of the compute
     * (measured: +12.2 us over a 275 us mock). */
    {
      std::vector<double> collect;
      double tw = 0, tr = 0;
      double a = now_s();
      double post_end = 0;
      for (int L = 0; L < layers; L++) { send_payload(L, 0); send_flag(L, SEQ_PIPE, 0); }
      post_end = now_s();
      uint64_t inflight = 0;
      for (int L = 0; L < layers; L++) {
        double c0 = now_s();
        wait_out(L, SEQ_PIPE);
        double c1 = now_s();
        if (inflight) { poll_cq_tracked(c, inflight, done, "read-out"); inflight = 0; }
        double c1b = now_s();
        inflight = post_read_to_dev(c, out_slot(L), remote.vaddr + out_slot(L), remote.rkey, bytes);
        double c2 = now_s();
        tw += c1 - c0; tr += (c1b - c1) + (c2 - c1b);
        collect.push_back(c2 - c0);
      }
      poll_cq_tracked(c, inflight, done, "read-out");
      double tot = now_s() - a;
      double cs = 0; for (double v : collect) cs += v;
      printf("RESULT pipeline layers=%d total_ms=%.3f per_layer_us=%.2f collect_us=%.2f "
             "transport_us=%.2f post_us=%.2f wait_us=%.2f read_us=%.2f\n",
             layers, tot * 1e3, tot * 1e6 / layers, cs * 1e6 / layers,
             std::max(0.0, tot * 1e6 / layers - mock_us), (post_end - a) * 1e6,
             tw * 1e6 / layers, tr * 1e6 / layers);
      fflush(stdout);
    }

    {
      long long isum = 0; double wsum = 0;
      for (int L = 0; L < layers; L++) {
        const int32_t* ids = (const int32_t*)((const uint8_t*)c.rbuf + rout_slot(L));
        const float* rtw = (const float*)(ids + rows * TOPK_ROUT);
        for (int i = 0; i < rows * TOPK_ROUT; i++) { isum += ids[i]; wsum += rtw[i]; }
      }
      printf("RESULT rout_sent layers=%d rows=%d n=%d isum=%lld wsum=%.6f\n", layers, rows,
             layers * rows * TOPK_ROUT, isum, wsum);
      fflush(stdout);
    }
    if (verify) {
      std::vector<uint8_t> got(bytes);
      CHECK_CUDA(cudaMemcpy(got.data(), (uint8_t*)c.buf + out_slot(layers - 1), bytes,
                            cudaMemcpyDeviceToHost));
      printf("%s verify bytes=%u\n", got == pat ? "OK" : "FAIL", bytes);
      fflush(stdout);
    }
    xfer_all(mfd, &b, 1, 1);
  }

  CHECK_CUDA(cudaStreamDestroy(st));
  if (c.mr) ibv_dereg_mr(c.mr);
  if (c.fmr) ibv_dereg_mr(c.fmr);
  if (c.qp) ibv_destroy_qp(c.qp);
  if (c.cq) ibv_destroy_cq(c.cq);
  if (c.pd) ibv_dealloc_pd(c.pd);
  if (c.ctx) ibv_close_device(c.ctx);
  cudaFreeHost(c.buf);
  cudaFreeHost(c.flags);
  close(mfd);
  return 0;
}
