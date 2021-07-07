#ifndef TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_VSI_NPU_API_H_
#define TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_VSI_NPU_API_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tim/vx/ops/elementwise.h>
#include <tim/vx/tensor.h>
#include <tim/vx/types.h>

namespace tvx = tim::vx;

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {

using TensorInfoTable = std::map<Expr, std::vector<tim::vx::TensorSpec>>;
using VxTensorTable =
    std::map<Expr, std::vector<std::shared_ptr<tim::vx::Tensor>>>;
using VxOperationTable = std::map<Expr, std::shared_ptr<tim::vx::Operation>>;

struct CallbackExpr {
  CallbackExpr(Expr expr) : expr_(expr){};
  CallbackExpr(Expr expr, std::shared_ptr<CallbackExpr> ptr_pre_callback)
      : expr_(expr), ptr_pre_callback_(ptr_pre_callback){};

  Expr expr_;
  std::shared_ptr<CallbackExpr> ptr_pre_callback_ = nullptr;
  // tvx::Quantization quant_info_;
};

class OpSetup {
public:
  OpSetup(tvx::TensorSpec spec, std::shared_ptr<CallbackExpr> pCallbackexpr)
      : pCallbackexpr_(pCallbackexpr) {
    specs_.push_back(spec);
  };
  OpSetup(tvx::TensorSpec spec) { specs_.push_back(spec); };
  OpSetup(std::vector<tvx::TensorSpec> specs,
          std::shared_ptr<CallbackExpr> pCallbackexpr)
      : pCallbackexpr_(pCallbackexpr) {
    specs_ = std::move(specs);
  };
  OpSetup(std::vector<tvx::TensorSpec> specs) { specs_ = std::move(specs); };
  void SetSpec(tvx::TensorSpec spec) { specs_.push_back(spec); }

  void SetTensor(std::shared_ptr<tvx::Tensor> ptensor) {
    ptensors_.push_back(ptensor);
  }

  virtual void
  SetupOperand(const CallNode *cn, tim::vx::Quantization &quant_info,
               std::map<Expr, std::shared_ptr<OpSetup>> &vxOpmap_tbl) {
    std::cout << "something wrong in OpSetup::SetupOperand!" << std::endl;
  };
  virtual void
  SetupOperation(const CallNode *cn, std::shared_ptr<tvx::Graph> graph,
                 std::map<Expr, std::shared_ptr<OpSetup>> &vxOpmap_tbl);

  virtual std::shared_ptr<tvx::Operation>
  CreateOperation(std::shared_ptr<tvx::Graph> graph) {
    std::cout << "something wrong in OpSetup::CreateOperation!" << std::endl;
    return nullptr;
  };

  Call call_;
  Expr expr_key_;
  Expr input_key_;
  std::shared_ptr<CallbackExpr> pCallbackexpr_ = nullptr;
  std::vector<tvx::TensorSpec> specs_;
  std::vector<std::shared_ptr<tvx::Tensor>> ptensors_;
  // std::shared_ptr<tvx::Operation> operation_;
};

using VxOpTable = std::map<Expr, std::shared_ptr<OpSetup>>;

namespace {
// Get a T from a constant represented by a NDArray.
template <typename T> bool AsConstant(const Expr &expr, T *out) {
  if (!expr->IsInstance<ConstantNode>()) {
    return false;
  }
  runtime::NDArray data = Downcast<Constant>(expr)->data;
  *out = *static_cast<T *>(data->data);
  return true;
}

// void printTensor(std::string info,tim::vx::Tensor* tensor){
//   std::cout<<info<<" ";
//   for(size_t i=0;i<tensor->GetShape().size();i++){
//     std::cout<<tensor->GetShape()[i]<<" ";
//   }
//   std::cout<<std::endl;
// }

void transformShape(tim::vx::TensorSpec &spec) {
  std::reverse(spec.shape_.begin(), spec.shape_.end());
}
} // namespace

} // namespace vsi_npu
} // namespace contrib
} // namespace relay
} // namespace tvm

#endif