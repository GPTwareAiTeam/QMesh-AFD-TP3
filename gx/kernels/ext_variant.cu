// SPDX-FileCopyrightText: 2026 QMesh contributors
// SPDX-License-Identifier: MIT
//
// Torch glue for one kernel variant. The kernel file is chosen at build time via
// -DKERNEL_FILE="\"<path>\"" so v1 and v2 build from identical glue.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include KERNEL_FILE

void run(torch::Tensor out, torch::Tensor x, torch::Tensor ids, torch::Tensor tw,
         torch::Tensor w13, torch::Tensor s13, torch::Tensor g13,
         torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
         torch::Tensor act, double swiglu_limit, bool has_limit) {
  auto st = at::cuda::getCurrentCUDAStream().stream();
  int R = x.size(0);
  decode_moe<5120, 896, 864>(
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(),
      w13.data_ptr<uint8_t>(), s13.data_ptr<uint8_t>(), g13.data_ptr<float>(),
      w2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), g2.data_ptr<float>(),
      act.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), R,
      (float)swiglu_limit, has_limit ? 1 : 0, st);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
