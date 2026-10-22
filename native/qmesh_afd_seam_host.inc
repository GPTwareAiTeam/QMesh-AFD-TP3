/* Host half of the seam: one proxy thread per peer and the Seam object.
 * Included by qmesh_afd_peer.cpp after qmesh_afd_seam.cu. */

/* One payload + routing + in-flag post for a layer the device already staged.
 * Same RC QP ordering as Peer::stage_and_post: routing, payload chunks, flag. */
static void seam_wire_post(Peer& p, int L, int n, int32_t s, bool has_rout) {
  const size_t xbytes = in_bytes(n);
  if (has_rout) {
    char* rb = (char*)(uintptr_t)p.rbase;
    ((int32_t*)rb)[0] = s;   /* hdr[0]: seq; device wrote n, inv, fmt */
    __atomic_thread_fence(__ATOMIC_RELEASE);
    ibv_sge sge{}; sge.addr = p.rbase + rout_slot(L);
    sge.length = (uint32_t)(ROUT_HDR + (size_t)n * ROUT_ROW);
    sge.lkey = p.rmr->lkey;
    ibv_send_wr wr{}; wr.wr_id = 0; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = 0;
    wr.wr.rdma.remote_addr = p.remote.rout_vaddr + rout_slot(L);
    wr.wr.rdma.rkey = p.remote.rout_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(p.qp, &wr, &bad)) DIE("seam post WRITE rout");
  }
  const uint32_t kChunk = 4u << 20;
  const int nchunk = (int)((xbytes + kChunk - 1) / kChunk);
  if (nchunk < 1 || nchunk > 16) DIE("seam post WRITE payload: chunk count");
  ibv_send_wr wrs[16];
  ibv_sge sges[16];
  const uint64_t pid = take_wr_id();
  for (int i = 0; i < nchunk; i++) {
    const size_t off = (size_t)i * kChunk;
    const uint32_t len = (uint32_t)(xbytes - off < kChunk ? xbytes - off : kChunk);
    sges[i] = ibv_sge{};
    sges[i].addr = p.base + in_slot(L) + off;
    sges[i].length = len;
    sges[i].lkey = p.mr->lkey;
    wrs[i] = ibv_send_wr{};
    wrs[i].wr_id = (i + 1 == nchunk) ? pid : 0;
    wrs[i].sg_list = &sges[i];
    wrs[i].num_sge = 1;
    wrs[i].opcode = IBV_WR_RDMA_WRITE;
    wrs[i].send_flags = (i + 1 == nchunk) ? IBV_SEND_SIGNALED : 0;
    wrs[i].wr.rdma.remote_addr = p.remote.vaddr + in_slot(L) + off;
    wrs[i].wr.rdma.rkey = p.remote.rkey;
    wrs[i].next = (i + 1 < nchunk) ? &wrs[i + 1] : nullptr;
  }
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(p.qp, &wrs[0], &bad)) DIE("seam post WRITE payload");
  int32_t* pflags = (int32_t*)p.flags;
  __atomic_store_n(&pflags[in_flag(L) / 4], s, __ATOMIC_RELEASE);
  ibv_sge fs{}; fs.addr = p.fbase + in_flag(L); fs.length = 4; fs.lkey = p.fmr->lkey;
  ibv_send_wr fw{}; fw.wr_id = 0; fw.sg_list = &fs; fw.num_sge = 1;
  fw.opcode = IBV_WR_RDMA_WRITE; fw.send_flags = 0;
  fw.wr.rdma.remote_addr = p.remote.flag_vaddr + in_flag(L);
  fw.wr.rdma.rkey = p.remote.flag_rkey;
  if (ibv_post_send(p.qp, &fw, &bad)) DIE("seam post WRITE flag");
  poll_cq_tracked(p, pid);
}

/* Per-peer proxy: owns the peer's QP pair and CQ while the Seam is open. */
struct SeamPeer {
  Peer* p = nullptr;
  SeamDoorbell* db = nullptr;     /* host view of the mapped doorbell */
  uint64_t* rx = nullptr;         /* host view of the mapped rx word */
  std::thread th;
};

struct SeamObj {
  int np = 0, layers = NL, rows = 1, device = 0;
  SeamPeer peers[SEAM_MAXP];
  PyObject* peer_objs[SEAM_MAXP] = {nullptr, nullptr, nullptr, nullptr};   /* keep Peer objects alive */
  SeamDev dev{};                   /* device-visible pointers, by value into kernels */
  SeamStatus* st_host = nullptr;   /* mapped status, host view */
  SeamStatus* st_dev = nullptr;
  uint32_t* amax = nullptr;        /* device */
  float* scale = nullptr;          /* device */
  uint64_t* ticket = nullptr;      /* device */
  std::atomic<bool> stop{false};
};

/* Per-layer deadline for the peer's out-flag. A live GX host answers in well
 * under a millisecond, so this fires only when a worker is gone or wedged. It is
 * a fail-stop: the round is never completed, rx is never published, and the seam
 * stays failed (status.failed) until it is closed. */
static constexpr long long SEAM_WAIT_MS = 2000;

static inline double seam_now_ms() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void seam_proxy_main(SeamObj* S, int idx) {
  SeamPeer& sp = S->peers[idx];
  Peer& p = *sp.p;
  uint64_t last = 0;
  const bool trace = getenv("QMESH_AFD_TRACE") != nullptr;
  for (;;) {
    uint64_t t;
    const double t_loop = seam_now_ms();
    for (;;) {
      t = __atomic_load_n(&sp.db->ticket, __ATOMIC_ACQUIRE);
      if (t != last) break;
      if (S->stop.load(std::memory_order_relaxed)) return;
      _mm_pause();
    }
    const double t_door = seam_now_ms();
    const int L = sp.db->layer;
    const int n = sp.db->n;
    last = t;
    if (L < 0 || L >= S->layers || n < 1 || n > S->rows) {
      fprintf(stderr, "seam: bad doorbell layer=%d n=%d\n", L, n);
      exit(4);
    }
    /* Round boundary: layer 0 opens a new sequence, as begin_round() does on
     * the old path, so a round is 40 layers under one seq. */
    if (L == 0) p.seq++;
    p.layer_polls = 0;
    seam_wire_post(p, L, n, p.seq, true);
    const double t_post = seam_now_ms();
    const int waited = p.wait_layer(L, SEAM_WAIT_MS);
    const double t_flag = seam_now_ms();
    if (waited < 0) {
      /* No out-flag inside the deadline: fail-stop. Nothing is read and no rx
       * word is published, so the collect can never combine a stale slab and
       * no later post can overwrite a payload a slow worker is still eating. */
      fprintf(stderr, "seam[%d] TIMEOUT round %d layer %d n %d: doorbell lag %.3f ms, "
                      "post %.3f ms, flag wait %.3f ms (rx=%llu last=%llu)\n",
              idx, p.seq, L, n, t_door - t_loop, t_post - t_door,
              t_flag - t_post, (unsigned long long)*sp.rx, (unsigned long long)last);
      __atomic_fetch_add(&S->st_host->timeouts, 1ull, __ATOMIC_RELAXED);
      __atomic_store_n(&S->st_host->failed, 1ull, __ATOMIC_RELEASE);
      while (!S->stop.load(std::memory_order_relaxed)) _mm_pause();
      return;
    }
    const uint64_t wr = p.read_layer(L, n);
    p.collect(wr);
    const double t_rx = seam_now_ms();
    __atomic_store_n(sp.rx, t, __ATOMIC_RELEASE);
    if (trace && (L == 0 || L == S->layers - 1 || n > 1))
      fprintf(stderr, "seam[%d] round %d layer %d n %d: doorbell lag %.3f ms, "
                      "post %.3f ms, flag wait %.3f ms, read+collect %.3f ms\n",
              idx, p.seq, L, n, t_door - t_loop, t_post - t_door,
              t_flag - t_post, t_rx - t_flag);
  }
}

static void seam_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "seam: %s: %s\n", what, cudaGetErrorString(e));
    exit(2);
  }
}

static constexpr int SEAM_THREADS = 256;
static constexpr int SEAM_MAX_BLOCKS = 1024;

static int seam_blocks(size_t work) {
  size_t b = (work + SEAM_THREADS - 1) / SEAM_THREADS;
  if (b < 1) b = 1;
  return (int)(b > SEAM_MAX_BLOCKS ? SEAM_MAX_BLOCKS : b);
}
