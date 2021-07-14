#ifndef TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_HELPER_H_
#define TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_HELPER_H_

#include "tim/vx/tensor.h"

#include <tvm/relay/expr.h>
#include <tvm/ir/expr.h>
#include "op_setup.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {
namespace op_map {

template <typename T>
bool AsConstant(const Expr& expr, std::vector<T>& out) {
  if (!expr->IsInstance<ConstantNode>()) {
    return false;
  }
  runtime::NDArray data = Downcast<Constant>(expr)->data;

  if (data->ndim == 0) {
    runtime::NDArray data = Downcast<Constant>(expr)->data;
    out.push_back(*static_cast<T*>(data->data));
  } else {
    int64_t size = data->shape[0];
    for (uint32_t i = 0; i < size; i++) {
      out.push_back(static_cast<T*>(data->data)[i]);
    }
  }
  return true;
}

void UpdateOutputQuantInfo(const Call& c, uint32_t scale_idx, uint32_t zp_idx,
                           tim::vx::Quantization& quant_info);

tim::vx::DataType GetTvxType(DataType dtype);

}  // namespace op_map
}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif