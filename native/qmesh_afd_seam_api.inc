/* Seam: the device-driven front end over connected Peer objects.
 * Included by qmesh_afd_peer.cpp after the Peer binding. */

/* Single-post / stream contract (eager and captured graphs): a Seam owns one
 * amax, fin, scale, ticket and one in/out slot per peer. A post and its collect,
 * and each post relative to the previous post's publish, must be ordered on the
 * GPU: use the same stream, or insert explicit stream/event ordering. The
 * per-call stream picks where a launch is recorded; it does not order two
 * different unsynchronized streams. Concurrent posts, overlapping graph
 * replays, and an abandoned post without its collect are unsupported. No host
 * pending guard is kept: graph replay runs without host calls, so host state
 * cannot see it, and a host memset cannot repair a concurrent unsynchronized
 * amax. */

static void seam_free(SeamObj* S);

/* Returns 1 when the launch was issued, 0 when fail-stop suppressed it. */
static int seam_launch_post(SeamObj* S, int layer, const void* x, size_t xs, const void* ids,
                            bool ids64, const void* tw, int n, cudaStream_t st) {
  SeamDev& d = S->dev;
  /* Fail-stop is terminal: after a missed round nothing more is launched, so
   * the wire never sees a post whose predecessor was never completed. */
  if (__atomic_load_n(&S->st_host->failed, __ATOMIC_ACQUIRE)) return 0;
  /* amax+finish is one node: the last amax block finalizes scale/header in
   * place. publish (the post path's last node) re-arms amax and fin for the next
   * post, so no separate memset node is needed. This launch must be stream-
   * ordered after the previous post's publish (contract above). */
  const float floor_bf16 = __bfloat162float(__float2bfloat16_rn(1e-6f));
  seam_amax_kernel<<<seam_blocks((size_t)n * K_HIDDEN), SEAM_THREADS, 0, st>>>(
      (const __nv_bfloat16*)x, xs, n, d, floor_bf16);
  /* Crossover measured on GPU0: 5.64/4.64 us at 6 rows and 29.5/43.5 us at 128
   * rows for vector/scalar, so the vector pack (which folds the doorbell in) takes
   * every n >= 128 and the byte-at-a-time pack keeps its separate publish launch
   * below that. Same payload, one dispatch, no runtime switch. */
  const bool pack_vec = n >= SEAM_PACK_VEC_MIN_ROWS && seam_pack_vec_ok(x, xs, d);
  if (pack_vec) {
    seam_pack_vec_kernel<<<seam_blocks(((size_t)n * K_HIDDEN) / 8), SEAM_THREADS, 0, st>>>(
        d, (const __nv_bfloat16*)x, xs, ids, ids64 ? 1 : 0, (const float*)tw, n, layer);
  } else {
    seam_pack_kernel<<<seam_blocks((size_t)n * K_HIDDEN), SEAM_THREADS, 0, st>>>(
        d, (const __nv_bfloat16*)x, xs, ids, ids64 ? 1 : 0, (const float*)tw, n);
    seam_publish_kernel<<<1, 1, 0, st>>>(d, layer, n);
  }
  return 1;
}

/* Returns 1 when the launches were issued, 0 when fail-stop suppressed them. */
static int seam_launch_collect(SeamObj* S, void* out, size_t os, bool out_f32, int n,
                               cudaStream_t st) {
  /* Fail-stop is terminal: once a round has been missed nothing is collected,
   * so the combine can never launder a stale slab into model output. This
   * collect must be stream-ordered after its post and before the next post
   * (contract above). */
  seam_collect_wait_kernel<<<1, 1, 0, st>>>(S->dev);
  const size_t total = (size_t)n * K_HIDDEN;
  /* 16B chunks halve the mapped-read count per element only when every chunk
   * of the local row is aligned; otherwise the scalar kernel keeps the exact
   * same result one element at a time. */
  const bool cmb_vec = seam_combine_vec_ok(out, os, out_f32);
  if (cmb_vec) {
    const int bv = seam_blocks(total / 8);
    if (out_f32)
      seam_combine_vec_kernel<true><<<bv, SEAM_THREADS, 0, st>>>(S->dev, out, os, n);
    else
      seam_combine_vec_kernel<false><<<bv, SEAM_THREADS, 0, st>>>(S->dev, out, os, n);
  } else if (out_f32) {
    seam_combine_kernel<true><<<seam_blocks(total), SEAM_THREADS, 0, st>>>(S->dev, out, os, n);
  } else {
    seam_combine_kernel<false><<<seam_blocks(total), SEAM_THREADS, 0, st>>>(S->dev, out, os, n);
  }
  return 1;
}

static SeamObj* seam_create(PyObject* peers, int layers, int rows, int device) {
  if (device != 0) { PyErr_SetString(PyExc_ValueError, "Seam: device must be 0 (peer slabs live on GPU0)"); return nullptr; }
  if (!PyList_Check(peers) && !PyTuple_Check(peers)) { PyErr_SetString(PyExc_TypeError, "Seam: peers must be a list"); return nullptr; }
  Py_ssize_t np = PySequence_Size(peers);
  if (np < 1 || np > SEAM_MAXP) { PyErr_SetString(PyExc_ValueError, "Seam: 1..4 peers"); return nullptr; }
  SeamObj* S = new SeamObj();
  S->np = (int)np; S->layers = layers; S->rows = rows; S->device = device;
  if (layers < 1 || layers > NL || rows < 1 || rows > R_MAX) {
    delete S; PyErr_SetString(PyExc_ValueError, "Seam: layers/rows out of range"); return nullptr;
  }
  seam_check(cudaSetDevice(0), "setDevice");
  for (int i = 0; i < S->np; i++) {
    PyObject* o = PySequence_GetItem(peers, i);
    if (!PyObject_TypeCheck(o, &PeerType)) { Py_XDECREF(o); delete S; PyErr_SetString(PyExc_TypeError, "Seam: peers must be _qmesh_afd.Peer"); return nullptr; }
    S->peers[i].p = ((PeerObj*)o)->p;   /* reference kept below; Python owns Peer */
    S->peer_objs[i] = o;
  }
  seam_check(cudaMalloc(&S->amax, sizeof(uint32_t)), "malloc amax");
  uint32_t* fin = nullptr;
  seam_check(cudaMalloc(&fin, sizeof(uint32_t)), "malloc fin");
  seam_check(cudaMalloc(&S->scale, 2 * sizeof(float)), "malloc scale");
  seam_check(cudaMalloc(&S->ticket, sizeof(uint64_t)), "malloc ticket");
  seam_check(cudaMemset(S->amax, 0, sizeof(uint32_t)), "memset amax");
  seam_check(cudaMemset(fin, 0, sizeof(uint32_t)), "memset fin");
  seam_check(cudaMemset(S->scale, 0, 2 * sizeof(float)), "memset scale");
  seam_check(cudaMemset(S->ticket, 0, sizeof(uint64_t)), "memset ticket");
  seam_check(cudaHostAlloc((void**)&S->st_host, sizeof(SeamStatus), cudaHostAllocMapped), "alloc status");
  memset(S->st_host, 0, sizeof(SeamStatus));
  seam_check(cudaHostGetDevicePointer((void**)&S->st_dev, S->st_host, 0), "status devptr");
  int clk_khz = 0;
  seam_check(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0), "clock rate");
  S->dev.np = S->np;
  S->dev.amax = S->amax; S->dev.fin = fin; S->dev.scale = S->scale; S->dev.ticket = S->ticket;
  S->dev.st = S->st_dev;
  /* Backstop only: the proxies detect a dead worker at 2 s and set the sticky
   * failure themselves. Bounded so a broken peer cannot hang close()'s drain. */
  S->dev.timeout_cyc = (long long)clk_khz * 1000 * 3;
  for (int i = 0; i < S->np; i++) {
    Peer& p = *S->peers[i].p;
    SeamDoorbell* db = nullptr;
    uint64_t* rx = nullptr;
    seam_check(cudaHostAlloc((void**)&db, sizeof(SeamDoorbell), cudaHostAllocMapped), "alloc doorbell");
    seam_check(cudaHostAlloc((void**)&rx, sizeof(uint64_t), cudaHostAllocMapped), "alloc rx");
    memset(db, 0, sizeof(*db)); *rx = 0;
    S->peers[i].db = db; S->peers[i].rx = rx;
    void* dp = nullptr;
    seam_check(cudaHostGetDevicePointer(&dp, p.buf, 0), "slab devptr");
    S->dev.in[i] = (uint8_t*)dp;
    S->dev.out[i] = (const __nv_bfloat16*)((char*)dp + OUT_OFF);
    seam_check(cudaHostGetDevicePointer(&dp, p.rbuf, 0), "rout devptr");
    S->dev.rout[i] = (char*)dp;
    seam_check(cudaHostGetDevicePointer(&dp, db, 0), "doorbell devptr");
    S->dev.db[i] = (SeamDoorbell*)dp;
    seam_check(cudaHostGetDevicePointer(&dp, rx, 0), "rx devptr");
    S->dev.rx[i] = (uint64_t*)dp;
  }
  for (int i = 0; i < S->np; i++)
    S->peers[i].th = std::thread(seam_proxy_main, S, i);
  return S;
}

static void seam_free(SeamObj* S) {
  /* Drain the device while the proxies are still alive: queued post/collect
   * kernels (and any graph replay in flight) read the mapped doorbell, rx,
   * status and in/out slab words, and a collect may still need an rx word the
   * proxy publishes. combine also preloads mapped out slabs before its failed
   * gate. Only after every kernel has retired is it safe to stop the proxies
   * and release the mapped memory they serve; do not reorder this teardown. */
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "seam: drain: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaGetLastError();   /* teardown continues; the seam is being destroyed anyway */
  }
  S->stop.store(true);
  for (int i = 0; i < S->np; i++)
    if (S->peers[i].th.joinable()) S->peers[i].th.join();
  for (int i = 0; i < S->np; i++) {
    if (S->peers[i].db) cudaFreeHost(S->peers[i].db);
    if (S->peers[i].rx) cudaFreeHost(S->peers[i].rx);
    Py_XDECREF(S->peer_objs[i]);
  }
  if (S->st_host) cudaFreeHost(S->st_host);
  if (S->amax) cudaFree(S->amax);
  if (S->dev.fin) cudaFree(S->dev.fin);
  if (S->scale) cudaFree(S->scale);
  if (S->ticket) cudaFree(S->ticket);
  delete S;
}
