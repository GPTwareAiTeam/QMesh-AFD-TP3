/* Python binding for Seam. Included by qmesh_afd_peer.cpp after PeerType. */

/* quant_test(x_ptr, x_stride, n, out_ptr, scale_ptr[, stream]): runs the real
 * fused amax+finish and pack kernels with no peers, writing e4m3 rows to
 * out_ptr and [s, inv] to scale_ptr, so the numerics can be checked against
 * torch bit for bit without a wire. */
static PyObject* seam_quant_test(PyObject*, PyObject* args) {
  unsigned long long x, xs, out, scale, stream = 0;
  int n;
  if (!PyArg_ParseTuple(args, "KKiKK|K", &x, &xs, &n, &out, &scale, &stream)) return nullptr;
  if (n < 1 || n > R_MAX) { PyErr_SetString(PyExc_ValueError, "quant_test: n out of range"); return nullptr; }
  cudaStream_t st = (cudaStream_t)(uintptr_t)stream;
  uint32_t* amax = nullptr;
  uint32_t* fin = nullptr;
  uint64_t* tk = nullptr;
  SeamStatus* stt = nullptr;
  seam_check(cudaMalloc(&amax, sizeof(uint32_t)), "qt malloc amax");
  seam_check(cudaMalloc(&fin, sizeof(uint32_t)), "qt malloc fin");
  seam_check(cudaMalloc(&tk, sizeof(uint64_t)), "qt malloc ticket");
  seam_check(cudaHostAlloc((void**)&stt, sizeof(SeamStatus), cudaHostAllocMapped), "qt status");
  memset(stt, 0, sizeof(SeamStatus));
  SeamDev d{};
  d.np = 1;
  d.amax = amax;
  d.fin = fin;
  d.scale = (float*)(uintptr_t)scale;
  d.ticket = tk;
  d.in[0] = (uint8_t*)(uintptr_t)out;
  d.st = stt;
  seam_check(cudaMemsetAsync(amax, 0, sizeof(uint32_t), st), "qt memset amax");
  seam_check(cudaMemsetAsync(fin, 0, sizeof(uint32_t), st), "qt memset fin");
  seam_amax_kernel<<<seam_blocks((size_t)n * K_HIDDEN), SEAM_THREADS, 0, st>>>(
      (const __nv_bfloat16*)(uintptr_t)x, (size_t)xs, n, d,
      __bfloat162float(__float2bfloat16_rn(1e-6f)));
  seam_pack_kernel<<<seam_blocks((size_t)n * K_HIDDEN), SEAM_THREADS, 0, st>>>(
      d, (const __nv_bfloat16*)(uintptr_t)x, (size_t)xs, nullptr, 0, nullptr, n);
  seam_check(cudaStreamSynchronize(st), "qt sync");
  unsigned long long nf = stt->nonfinite;
  cudaFree(amax); cudaFree(fin); cudaFree(tk); cudaFreeHost(stt);
  if (nf) return PyLong_FromUnsignedLongLong(nf);
  Py_RETURN_NONE;
}

typedef struct { PyObject_HEAD SeamObj* s; } SeamPyObj;

static PyObject* SeamPy_new(PyTypeObject* type, PyObject*, PyObject*) {
  SeamPyObj* self = (SeamPyObj*)type->tp_alloc(type, 0);
  if (self) self->s = nullptr;
  return (PyObject*)self;
}

static void SeamPy_dealloc(SeamPyObj* self) {
  if (self->s) { seam_free(self->s); self->s = nullptr; }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int SeamPy_init(SeamPyObj* self, PyObject* args, PyObject* kwds) {
  static const char* kw[] = {"peers", "layers", "rows", "device", nullptr};
  PyObject* peers = nullptr; int layers = NL, rows = 1, device = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", (char**)kw, &peers, &layers, &rows, &device))
    return -1;
  SeamObj* s = seam_create(peers, layers, rows, device);
  if (!s) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Seam init failed"); return -1; }
  self->s = s;
  return 0;
}

/* post(layer, x_ptr, x_stride, ids_ptr, ids_int64, tw_ptr, n, stream) */
static PyObject* SeamPy_post(SeamPyObj* self, PyObject* args) {
  int layer, ids64 = 0, n;
  unsigned long long x, xs, ids, tw, stream;
  if (!PyArg_ParseTuple(args, "iKKKiKiK", &layer, &x, &xs, &ids, &ids64, &tw, &n, &stream))
    return nullptr;
  if (!self->s) { PyErr_SetString(PyExc_RuntimeError, "Seam closed"); return nullptr; }
  if (n == 0) Py_RETURN_NONE;   /* contract: n=0 is a no-op, no doorbell */
  if (n < 0 || n > self->s->rows || layer < 0 || layer >= self->s->layers) {
    PyErr_SetString(PyExc_ValueError, "Seam.post: layer or n out of range"); return nullptr;
  }
  if (!seam_launch_post(self->s, layer, (const void*)(uintptr_t)x, (size_t)xs,
                        (const void*)(uintptr_t)ids, ids64 != 0, (const void*)(uintptr_t)tw, n,
                        (cudaStream_t)(uintptr_t)stream)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Seam.post: transport failed (sticky); see status()['failed']");
    return nullptr;
  }
  Py_RETURN_NONE;
}

/* collect(layer, out_ptr, out_stride, out_is_fp32, n, stream) */
static PyObject* SeamPy_collect(SeamPyObj* self, PyObject* args) {
  int layer, out_f32 = 0, n;
  unsigned long long out, os, stream;
  if (!PyArg_ParseTuple(args, "iKKpiK", &layer, &out, &os, &out_f32, &n, &stream)) return nullptr;
  if (!self->s) { PyErr_SetString(PyExc_RuntimeError, "Seam closed"); return nullptr; }
  if (n == 0) Py_RETURN_NONE;
  if (n < 0 || n > self->s->rows || layer < 0 || layer >= self->s->layers) {
    PyErr_SetString(PyExc_ValueError, "Seam.collect: layer or n out of range"); return nullptr;
  }
  if (!seam_launch_collect(self->s, (void*)(uintptr_t)out, (size_t)os, out_f32 != 0, n,
                           (cudaStream_t)(uintptr_t)stream)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Seam.collect: transport failed (sticky); see status()['failed']");
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyObject* SeamPy_status(SeamPyObj* self, PyObject*) {
  if (!self->s) { PyErr_SetString(PyExc_RuntimeError, "Seam closed"); return nullptr; }
  SeamObj* s = self->s;
  PyObject* d = PyDict_New();
  if (!d) return nullptr;
  const struct { const char* k; unsigned long long v; } kv[] = {
    {"nonfinite", s->st_host->nonfinite},
    {"timeouts",  s->st_host->timeouts},
    {"posts",     s->st_host->posts},
    {"failed",    s->st_host->failed},
  };
  for (size_t i = 0; i < sizeof(kv) / sizeof(kv[0]); i++) {
    PyObject* v = PyLong_FromUnsignedLongLong(kv[i].v);
    if (!v) { Py_DECREF(d); return nullptr; }
    const int rc = PyDict_SetItemString(d, kv[i].k, v);   /* INCREFs v */
    Py_DECREF(v);
    if (rc) { Py_DECREF(d); return nullptr; }
  }
  PyObject* tk = PyList_New(s->np);
  if (!tk) { Py_DECREF(d); return nullptr; }
  for (int i = 0; i < s->np; i++) {
    PyObject* v = PyLong_FromUnsignedLongLong(*s->peers[i].rx);
    if (!v) { Py_DECREF(tk); Py_DECREF(d); return nullptr; }
    PyList_SET_ITEM(tk, i, v);   /* steals v */
  }
  const int rc = PyDict_SetItemString(d, "rx", tk);
  Py_DECREF(tk);
  if (rc) { Py_DECREF(d); return nullptr; }
  return d;
}

static PyObject* SeamPy_close(SeamPyObj* self, PyObject*) {
  if (self->s) { seam_free(self->s); self->s = nullptr; }
  Py_RETURN_NONE;
}

static PyMethodDef SeamPy_methods[] = {
  {"post", (PyCFunction)SeamPy_post, METH_VARARGS,
   "post(layer, x_ptr, x_stride, ids_ptr, ids_int64, tw_ptr, n, stream): "
   "device-only, capture-safe. The post/collect pair and every later post must "
   "be stream-ordered (same stream or explicit GPU ordering)."},
  {"collect", (PyCFunction)SeamPy_collect, METH_VARARGS,
   "collect(layer, out_ptr, out_stride, out_is_fp32, n, stream): device-only, "
   "capture-safe. Must be stream-ordered after its post and before the next post."},
  {"status", (PyCFunction)SeamPy_status, METH_NOARGS, "status() -> dict."},
  {"close", (PyCFunction)SeamPy_close, METH_NOARGS, "close()."},
  {nullptr, nullptr, 0, nullptr},
};

static PyTypeObject SeamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  .tp_name = "_qmesh_afd.Seam",
  .tp_basicsize = sizeof(SeamPyObj),
  .tp_dealloc = (destructor)SeamPy_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Device-driven AFD seam over connected Peer objects. Exactly one "
            "post may be in flight: every post must be followed by its own "
            "collect before the next post. All launches for a post+collect pair "
            "and each later post must be ordered on the GPU: same CUDA stream, or "
            "an explicit stream/event sync. The per-call stream does not order "
            "different unsynchronized streams; concurrent posts, overlapping "
            "graph replays, and abandoning a post without its collect are "
            "unsupported (amax/fin/scale/slots are single-instance and are not "
            "re-armed to recover such a round). This also applies inside captured "
            "graphs.",
  .tp_methods = SeamPy_methods,
  .tp_init = (initproc)SeamPy_init,
  .tp_new = SeamPy_new,
};
