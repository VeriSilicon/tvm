#include "op_setup.h"
#include "filed.h"
#include "helper.h"
#include "attribute.h"
#include <tvm/relay/function.h>
#include <tvm/relay/attrs/transform.h>
#include <map>
#include <memory>

#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/addn.h"
#include "tim/vx/ops/arg.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reduce.h"
#include "tim/vx/ops/resize.h"
#include "tim/vx/ops/simple_operations.h"
#include "tim/vx/ops/softmax.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/reshape.h"
#include <tim/vx/ops/clip.h>
#include <tim/vx/ops/fullyconnected.h>
#include <tim/vx/ops/squeeze.h>
#include <tim/vx/ops/depth2space.h>
#include <tim/vx/ops/logical.h>
#include <tim/vx/ops/pad.h>
#include <tim/vx/ops/deconv.h>

namespace tvx = tim::vx;

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {
namespace op_map {

void shape_setup(const Call &c, uint32_t arg_idx,
                 tim::vx::ShapeType &result_shape) {
  std::transform(
      c->args[arg_idx]->checked_type().as<TensorTypeNode>()->shape.rbegin(),
      c->args[arg_idx]->checked_type().as<TensorTypeNode>()->shape.rend(),
      std::back_inserter(result_shape), [](const PrimExpr &dim) {
        return static_cast<uint32_t>(dim.as<IntImmNode>()->value);
      });
}

void UpdateInputTableInfo(std::map<Expr, std::shared_ptr<OpSetup>>& VxOp_tb, Expr expr,
                          tim::vx::Graph* graph, uint32_t idx) {
  auto tensor_spec = VxOp_tb[expr]->specs_[idx];

  void* data = expr->IsInstance<ConstantNode>() ? expr.as<ConstantNode>()->data->data : nullptr;
  auto input_tensor =
      data == nullptr ? graph->CreateTensor(tensor_spec) : graph->CreateTensor(tensor_spec, data);
  VxOp_tb[expr]->SetTensor(input_tensor);
}

void UpdateOutputTableInfo(std::map<Expr, std::shared_ptr<OpSetup>>& VxOp_tb, Expr expr,
                           tim::vx::Graph* graph) {
  if (VxOp_tb[expr]->specs_[0].attr_ == tim::vx::TensorAttribute::OUTPUT) {
    VxOp_tb[expr]->SetTensor(graph->CreateTensor(VxOp_tb[expr]->specs_[0]));
  }
}

void UpdateOutputQuantInfo(const Call& c, uint32_t scale_idx, uint32_t zp_idx,
                           tim::vx::Quantization& quant_info) {
  float scale;
  int32_t zp;

  AsConstant(c->args[scale_idx], &scale);
  AsConstant(c->args[zp_idx], &zp);

  quant_info.SetType(tim::vx::QuantType::ASYMMETRIC)
      .SetScales({scale})
      .SetZeroPoints({zp});
}

void OpSetup::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                             std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());
  auto op = CreateOperation(graph);
  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
};

void VsiNpuQnnConv2d::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                   std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  constexpr uint32_t kRequant_output_scale_idx = 3;
  constexpr uint32_t kRequant_output_zp_idx = 4;

  const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
  auto expr = GetRef<Expr>(callnode);
  // extract calls in pattern
  Call requantize = Downcast<Call>(expr);
  Call add = Downcast<Call>(requantize->args[0]);
  conv_ = Downcast<Call>(add->args[0]);

  if (vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::UINT8) {
    using Input_Field = Field_ASYMM_U8<0, 4, 2, tim::vx::TensorAttribute::TRANSIENT,
                                       tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;
    using Weight_Field = Field_ASYMM_U8<1, 5, 3, tim::vx::TensorAttribute::CONSTANT,
                                        tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;

    using Bias_Field = Field_ASYMM_U8<1, 1, 2, tim::vx::TensorAttribute::CONSTANT,
                                      tim::vx::DataType::INT32, tim::vx::QuantType::ASYMMETRIC>;
    input_key_ = call_->args[Input_Field::arg_pos];
    weight_key_ = conv_->args[Weight_Field::arg_pos];
    bias_key_ = add->args[Bias_Field::arg_pos];

    if (vxOpmap_tbl.find(input_key_) == vxOpmap_tbl.end()) {
      vxOpmap_tbl[input_key_] =
          std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(conv_, conv_));
    }

    tim::vx::TensorSpec weight_spec = Weight_Field::AsTimVxTensorSpec(conv_, conv_);
    tim::vx::TensorSpec bias_spec = Bias_Field::AsTimVxTensorSpec(add, requantize);

    vxOpmap_tbl[weight_key_] =
        std::make_shared<OpSetup>(Weight_Field::AsTimVxTensorSpec(conv_, conv_));
    bias_spec.shape_.resize(1);
    bias_spec.quantization_.Scales()[0] =
        weight_spec.quantization_.Scales()[0] *
        vxOpmap_tbl[input_key_]->specs_[0].quantization_.Scales()[0];
    vxOpmap_tbl[bias_key_] = std::make_shared<OpSetup>(bias_spec);
  }else if(vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::INT8){
    using Input_Field = Field_ASYMM_U8<0, 4, 2, tim::vx::TensorAttribute::TRANSIENT,
                                       tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;
    using Weight_Field = Field_ASYMM_U8<1, 5, 3, tim::vx::TensorAttribute::CONSTANT,
                                        tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;

    using Bias_Field = Field_ASYMM_U8<1, 1, 2, tim::vx::TensorAttribute::CONSTANT,
                                      tim::vx::DataType::INT32, tim::vx::QuantType::ASYMMETRIC>;
    input_key_ = call_->args[Input_Field::arg_pos];
    weight_key_ = conv_->args[Weight_Field::arg_pos];
    bias_key_ = add->args[Bias_Field::arg_pos];

    if (vxOpmap_tbl.find(input_key_) == vxOpmap_tbl.end()) {
      vxOpmap_tbl[input_key_] =
          std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(conv_, conv_));
    }

    tim::vx::TensorSpec weight_spec = Weight_Field::AsTimVxTensorSpec(conv_, conv_);
    tim::vx::TensorSpec bias_spec = Bias_Field::AsTimVxTensorSpec(add, requantize);

    vxOpmap_tbl[weight_key_] =
        std::make_shared<OpSetup>(Weight_Field::AsTimVxTensorSpec(conv_, conv_));
    bias_spec.shape_.resize(1);
    bias_spec.quantization_.Scales()[0] =
        weight_spec.quantization_.Scales()[0] *
        vxOpmap_tbl[input_key_]->specs_[0].quantization_.Scales()[0];
    vxOpmap_tbl[bias_key_] = std::make_shared<OpSetup>(bias_spec);
  }
  
  UpdateOutputQuantInfo(requantize, kRequant_output_scale_idx, kRequant_output_zp_idx, quant_info);
};

void VsiNpuQnnConv2d::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                     std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxConv2dAttrs tvx_attrs(conv_);
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, bias_key_, graph.get());

  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  bool is_depthwise_conv =
      (static_cast<uint32_t>(tvx_attrs.groups) == vxOpmap_tbl[input_key_]->specs_[0].shape_[0]);

  auto weight_spec = vxOpmap_tbl[weight_key_]->specs_[0];
  uint32_t kernel_ic = weight_spec.shape_[1];
  uint32_t kernel_oc = weight_spec.shape_[0];

  if(is_depthwise_conv && kernel_oc != 1){
    vxOpmap_tbl[weight_key_]->specs_[0].shape_[0]= 1;
    vxOpmap_tbl[weight_key_]->specs_[0].shape_[1]= kernel_ic*kernel_oc;
  }
  UpdateInputTableInfo(vxOpmap_tbl, weight_key_, graph.get());

  auto op = graph->CreateOperation<tim::vx::ops::Conv2d>(
      is_depthwise_conv ? kernel_ic : kernel_oc, tvx_attrs.pad_type,
      std::array<uint32_t, 2>{tvx_attrs.kernel_size[0], tvx_attrs.kernel_size[1]},
      std::array<uint32_t, 2>{tvx_attrs.strides[0], tvx_attrs.strides[1]},
      std::array<uint32_t, 2>{tvx_attrs.dilation[0], tvx_attrs.dilation[1]},
      std::array<uint32_t, 4>{tvx_attrs.padding[0], tvx_attrs.padding[1], tvx_attrs.padding[2],
                              tvx_attrs.padding[3]},
      is_depthwise_conv ? kernel_oc : 0, tim::vx::DataLayout::CWHN, tim::vx::DataLayout::OcIcWH);

  (*op).BindInputs({vxOpmap_tbl[input_key_]->ptensors_[0], vxOpmap_tbl[weight_key_]->ptensors_[0],
                    vxOpmap_tbl[bias_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void VsiNpuQnnAvgPool::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_Field = Field_NoDType<0, tim::vx::TensorAttribute::TRANSIENT>;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
  auto expr = GetRef<Expr>(callnode);

  // extract calls in pattern
  Call cast_out = Downcast<Call>(expr);
  avgpool_ = Downcast<Call>(cast_out->args[0]);
  Call cast_in = Downcast<Call>(avgpool_->args[0]);

  input_key_ = call_->args[Input_Field::arg_pos];

  if (vxOpmap_tbl.find(input_key_) != vxOpmap_tbl.end()) {
    return;
  }

  auto input_callback =
      std::make_shared<CallbackExpr>(input_key_, vxOpmap_tbl[expr_key_]->pCallbackexpr_);
  vxOpmap_tbl[input_key_] =
      std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(call_), input_callback);
  vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tvx::DataType::UINT8);

};

std::shared_ptr<tim::vx::Operation> VsiNpuQnnAvgPool::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxPool2DAttrs tvx_attrs(avgpool_, TvxPool2DAttrs::kAvgPool);
  return graph->CreateOperation<tvx::ops::Pool2d>(
        tvx::PoolType::AVG, tvx_attrs.pad_type,
        std::array<uint32_t, 2>{tvx_attrs.pool_size[0], tvx_attrs.pool_size[1]},
        std::array<uint32_t, 2>{tvx_attrs.strides[0], tvx_attrs.strides[1]},
        tvx_attrs.ceil_mode, tvx::DataLayout::CWHN);
}

void VsiNpuQnnMean::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_Field = Field_NoDType<0, tim::vx::TensorAttribute::TRANSIENT>;

  constexpr uint32_t kRequant_intput_scale_idx = 1;
  constexpr uint32_t kRequant_input_zp_idx = 2;
  constexpr uint32_t kRequant_output_scale_idx = 3;
  constexpr uint32_t kRequant_output_zp_idx = 4;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
  auto expr = GetRef<Expr>(callnode);

  // extract calls in pattern
  Call requantize = Downcast<Call>(expr);
  mean_ = Downcast<Call>(requantize->args[0]);
  Call cast = Downcast<Call>(mean_->args[0]);

  input_key_ = call_->args[Input_Field::arg_pos];

  vxOpmap_tbl[input_key_] =
      std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(call_));

  tim::vx::Quantization input_quant_info;
  UpdateOutputQuantInfo(requantize, kRequant_intput_scale_idx, kRequant_input_zp_idx, input_quant_info);
  if(vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::UINT8){
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tvx::DataType::UINT8).SetQuantization(input_quant_info);
  }else if(vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::INT8){
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tvx::DataType::INT8).SetQuantization(input_quant_info);
  }

  UpdateOutputQuantInfo(requantize, kRequant_output_scale_idx, kRequant_output_zp_idx, quant_info);
};

std::shared_ptr<tim::vx::Operation> VsiNpuQnnMean::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxReduceMeanAttrs tvx_attrs(mean_);
  return graph->CreateOperation<tim::vx::ops::ReduceMean>(tvx_attrs.axis, tvx_attrs.keepdims);
}

void VsiNpuQnnSoftmax::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxSoftmaxAttrs tvx_attrs(op_);
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto input_sec = vxOpmap_tbl[input_key_]->specs_[0];
  auto op = graph->CreateOperation<tim::vx::ops::Softmax>(
      1.0, input_sec.shape_.size() - 1 - tvx_attrs.axis);

  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void QnnSingleInputOpSetup::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                  std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
    constexpr uint32_t kdequantize_output_scale_idx = 1;
    constexpr uint32_t kdequantize_output_zp_idx = 2;
    call_ = GetRef<Call>(cn);
    expr_key_ = GetRef<Expr>(cn);

    const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
    auto expr = GetRef<Expr>(callnode);
    // extract calls in pattern
    Call quantize = Downcast<Call>(expr);
    op_ = Downcast<Call>(quantize->args[0]);
    Call dequantize = Downcast<Call>(op_->args[0]);
  if (vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::UINT8) {
    using Input_Field = Field_ASYMM_U8<0, 1, 2, tim::vx::TensorAttribute::TRANSIENT,
                                       tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;

    input_key_ = call_->args[Input_Field::arg_pos];
    vxOpmap_tbl[input_key_] =
        std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(dequantize, dequantize));
  } else if (vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::INT8) {
    using Input_Field = Field_ASYMM_U8<0, 1, 2, tim::vx::TensorAttribute::TRANSIENT,
                                       tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;
    input_key_ = call_->args[Input_Field::arg_pos];
    vxOpmap_tbl[input_key_] =
        std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(dequantize, dequantize));
  }

  UpdateOutputQuantInfo(quantize, kdequantize_output_scale_idx, kdequantize_output_zp_idx,
                        quant_info);
};

void VsiNpuConcat::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                 std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_Field = Field_TUPLE_U8<0, 1, 2, tim::vx::TensorAttribute::TRANSIENT,
                                     tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;
  constexpr uint32_t kdequantize_output_scale_idx = 3;
  constexpr uint32_t kdequantize_output_zp_idx = 4;
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  input_key_ = call_->args[Input_Field::arg_pos];

  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(call_, call_));

  UpdateOutputQuantInfo(call_, kdequantize_output_scale_idx, kdequantize_output_zp_idx, quant_info);
}

void VsiNpuConcat::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                   std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  auto attrs = call_->attrs.as<ConcatenateAttrs>();
  int32_t axis = attrs->axis;
  int32_t input_size = vxOpmap_tbl[input_key_]->specs_.size();
  int32_t input_tensor_dims = vxOpmap_tbl[input_key_]->specs_[0].shape_.size();

  axis = input_tensor_dims - axis - 1;
  auto op = graph->CreateOperation<tim::vx::ops::Concat>(axis, input_size);

  for (int32_t i = 0; i < input_size; i++) {
    auto input = graph->CreateTensor(vxOpmap_tbl[input_key_]->specs_[i]);
    vxOpmap_tbl[input_key_]->ptensors_.push_back(input);
    (*op).BindInputs({input});
  }

  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void VsiNpuQnnDeconv::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                   std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  using Input_Field = Field_ASYMM_U8<0, 4, 2, tim::vx::TensorAttribute::TRANSIENT,
                                     tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;
  using Weight_Field = Field_ASYMM_U8<1, 5, 3, tim::vx::TensorAttribute::CONSTANT,
                                      tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;

  constexpr uint32_t kRequant_output_scale_idx = 3;
  constexpr uint32_t kRequant_output_zp_idx = 4;

  const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
  auto expr = GetRef<Expr>(callnode);
  // extract calls in pattern
  Call requantize = Downcast<Call>(expr);
  conv_ = Downcast<Call>(requantize->args[0]);
  dilate_ = Downcast<Call>(conv_->args[0]);

  input_key_ = call_->args[Input_Field::arg_pos];
  weight_key_ = conv_->args[Weight_Field::arg_pos];

  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(call_, conv_));

  tim::vx::TensorSpec weight_spec = Weight_Field::AsTimVxTensorSpec(conv_, conv_);

  vxOpmap_tbl[weight_key_] =
      std::make_shared<OpSetup>(Weight_Field::AsTimVxTensorSpec(conv_, conv_));

  UpdateOutputQuantInfo(requantize, kRequant_output_scale_idx, kRequant_output_zp_idx, quant_info);
};

void VsiNpuQnnDeconv::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                     std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, weight_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  int oc_count = static_cast<int>(vxOpmap_tbl[expr_key_]->specs_[0].shape_[0]);

  TvxConv2dAttrs tvx_attrs(conv_);
  TvxDilateAttrs tvx_dilate_attrs(dilate_);

  uint32_t stride_h = tvx_dilate_attrs.strides[1];
  uint32_t stride_w = tvx_dilate_attrs.strides[2];

  uint32_t pad_top = stride_h - tvx_attrs.padding[0] - 1;
  uint32_t pad_left = stride_w - tvx_attrs.padding[1] - 1;
  uint32_t pad_bottom = stride_h - tvx_attrs.padding[2] - 1;
  uint32_t pad_right = stride_w - tvx_attrs.padding[3] - 1;

  
  auto op = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      oc_count, tvx_attrs.pad_type, std::array<uint32_t, 2>{tvx_attrs.kernel_size[1], tvx_attrs.kernel_size[2]},
      std::array<uint32_t, 2>{tvx_dilate_attrs.strides[1],tvx_dilate_attrs.strides[2]},
      std::array<uint32_t, 2>{0, 0},
      std::array<uint32_t, 4>{pad_top, pad_left, pad_bottom, pad_right}, tvx_attrs.groups,
      tim::vx::DataLayout::CWHN, tim::vx::DataLayout::OcIcWH);

  (*op).BindInputs({vxOpmap_tbl[input_key_]->ptensors_[0], vxOpmap_tbl[weight_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void SingleFloatInputSetup::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                         std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_Float32<0, tim::vx::TensorAttribute::TRANSIENT>;
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input_key_ = call_->args[Input_0::arg_pos];
  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_));
  (void)quant_info;
}

void Softmax::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                             std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxSoftmaxAttrs tvx_attrs(call_);
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto input_sec = vxOpmap_tbl[input_key_]->specs_[0];
  auto op = graph->CreateOperation<tim::vx::ops::Softmax>(
      1.0, input_sec.shape_.size() - 1 - tvx_attrs.axis);

  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

std::shared_ptr<tim::vx::Operation> AvgPool::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxPool2DAttrs tvx_attrs(call_, TvxPool2DAttrs::kAvgPool);
  return graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::AVG, tim::vx::PadType::VALID,
      std::array<uint32_t, 2>{tvx_attrs.pool_size[0], tvx_attrs.pool_size[1]},
      std::array<uint32_t, 2>{tvx_attrs.strides[0], tvx_attrs.strides[1]}, tvx_attrs.ceil_mode,
      tim::vx::DataLayout::CWHN);
}

std::shared_ptr<tim::vx::Operation> Transpose::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxTransposeAttrs tvx_attrs(call_);
  return graph->CreateOperation<tim::vx::ops::Transpose>(tvx_attrs.axes);
}

void ElementWiseQnnOp::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  constexpr uint32_t out_scale_idx = 6;
  constexpr uint32_t out_zp_idx = 7;
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  if (vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::UINT8) {
    using Input_0 = Field_ASYMM_U8<0, 2, 3, tim::vx::TensorAttribute::TRANSIENT,
                                   tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;
    using Input_1 =
        Field_NoAttribute_U8<1, 4, 5, tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;

    if (vxOpmap_tbl.find(input_key_) == vxOpmap_tbl.end()) {
      input_key_ = call_->args[Input_0::arg_pos];
      vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_, call_));
    }

    if (vxOpmap_tbl.find(input2_key_) == vxOpmap_tbl.end()) {
      input2_key_ = call_->args[Input_1::arg_pos];
      if (input2_key_->IsInstance<ConstantNode>()) {
        vxOpmap_tbl[input2_key_] = std::make_shared<OpSetup>(
            Input_1::AsTimVxTensorSpec(call_, tim::vx::TensorAttribute::CONSTANT));
      } else {
        vxOpmap_tbl[input2_key_] = std::make_shared<OpSetup>(
            Input_1::AsTimVxTensorSpec(call_, tim::vx::TensorAttribute::TRANSIENT));
      }
    }
  } else if (vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::INT8) {
    using Input_0 = Field_ASYMM_U8<0, 2, 3, tim::vx::TensorAttribute::TRANSIENT,
                                   tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;
    using Input_1 =
        Field_NoAttribute_U8<1, 4, 5, tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;

    if (vxOpmap_tbl.find(input_key_) == vxOpmap_tbl.end()) {
      input_key_ = call_->args[Input_0::arg_pos];
      vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_, call_));
    }

    if (vxOpmap_tbl.find(input2_key_) == vxOpmap_tbl.end()) {
      input2_key_ = call_->args[Input_1::arg_pos];
      if (input2_key_->IsInstance<ConstantNode>()) {
        vxOpmap_tbl[input2_key_] = std::make_shared<OpSetup>(
            Input_1::AsTimVxTensorSpec(call_, tim::vx::TensorAttribute::CONSTANT));
      } else {
        vxOpmap_tbl[input2_key_] = std::make_shared<OpSetup>(
            Input_1::AsTimVxTensorSpec(call_, tim::vx::TensorAttribute::TRANSIENT));
      }
    }
  }
  UpdateOutputQuantInfo(call_, out_scale_idx, out_zp_idx, quant_info);
}

void ElementWiseQnnOp::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, input2_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());
  auto op = CreateOperation(graph);
  (*op).BindInputs({vxOpmap_tbl[input_key_]->ptensors_[0], vxOpmap_tbl[input2_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
};

void ElementWiseNotypeOp::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_NoDType<0, tim::vx::TensorAttribute::TRANSIENT>;
  using Input_1 = Field_NoDType<1, tim::vx::TensorAttribute::TRANSIENT>;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  input_key_ = call_->args[Input_0::arg_pos];
  input2_key_ = call_->args[Input_1::arg_pos];

  auto input_callback =
      std::make_shared<CallbackExpr>(input_key_, vxOpmap_tbl[expr_key_]->pCallbackexpr_);

  auto input2_callback =
      std::make_shared<CallbackExpr>(input2_key_, vxOpmap_tbl[expr_key_]->pCallbackexpr_);

  auto dtype = input_key_->checked_type().as<TensorTypeNode>()->dtype;

  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_),input_callback);
  vxOpmap_tbl[input2_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_),input2_callback);

  if (dtype.is_float()) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::FLOAT32);
    vxOpmap_tbl[input2_key_]->specs_[0].SetDataType(tim::vx::DataType::FLOAT32);
  } else if (dtype.is_uint()) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::UINT8);
    vxOpmap_tbl[input2_key_]->specs_[0].SetDataType(tim::vx::DataType::UINT8);
  } else if (dtype.is_int()) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::INT32);
    vxOpmap_tbl[input2_key_]->specs_[0].SetDataType(tim::vx::DataType::INT32);
  }

  (void)quant_info;
}

void ElementWiseNotypeOp::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, input2_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());
  auto op = CreateOperation(graph);
  (*op).BindInputs({vxOpmap_tbl[input_key_]->ptensors_[0], vxOpmap_tbl[input2_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
};

std::shared_ptr<tim::vx::Operation> Maximum::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::Maximum>();
}

std::shared_ptr<tim::vx::Operation> Minimum::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::Minimum>();
}

std::shared_ptr<tim::vx::Operation> QnnAdd::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::Add>();
}

std::shared_ptr<tim::vx::Operation> QnnSubtract::CreateOperation(
    std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::Sub>();
}

std::shared_ptr<tim::vx::Operation> QnnMul::CreateOperation(
    std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::Multiply>();
}

void TwoBoolInputSetup::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                        std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_Bool<0, tim::vx::TensorAttribute::TRANSIENT>;
  using Input_1 = Field_Bool<1, tim::vx::TensorAttribute::TRANSIENT>;
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input0_key_ = call_->args[Input_0::arg_pos];
  input1_key_ = call_->args[Input_1::arg_pos];
  vxOpmap_tbl[input0_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_));
  vxOpmap_tbl[input1_key_] = std::make_shared<OpSetup>(Input_1::AsTimVxTensorSpec(call_));

  (void)quant_info;

};

void LogicalAnd::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                         std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input0_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, input1_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto op = graph->CreateOperation<tim::vx::ops::LogicalAnd>();
  (*op).BindInputs(
      {vxOpmap_tbl[input0_key_]->ptensors_[0], vxOpmap_tbl[input1_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void LogicalOr::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                         std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input0_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, input1_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto op = graph->CreateOperation<tim::vx::ops::LogicalOr>();
  (*op).BindInputs(
      {vxOpmap_tbl[input0_key_]->ptensors_[0], vxOpmap_tbl[input1_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void TwoFloatInputOpSetup::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                        std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_Float32<0, tim::vx::TensorAttribute::TRANSIENT>;
  using Input_1 = Field_NoAttribute_Float32<1>;
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input0_key_ = call_->args[Input_0::arg_pos];
  input1_key_ = call_->args[Input_1::arg_pos];
  vxOpmap_tbl[input0_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_));
  if (input1_key_->IsInstance<ConstantNode>()) {
    vxOpmap_tbl[input1_key_] = std::make_shared<OpSetup>(
        Input_1::AsTimVxTensorSpec(call_, tim::vx::TensorAttribute::CONSTANT));
  } else {
    vxOpmap_tbl[input1_key_] = std::make_shared<OpSetup>(
        Input_1::AsTimVxTensorSpec(call_, tim::vx::TensorAttribute::TRANSIENT));
    (void)quant_info;
  };
}

void Add::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                         std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input0_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, input1_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto op = graph->CreateOperation<tim::vx::ops::Add>();
  (*op).BindInputs(
      {vxOpmap_tbl[input0_key_]->ptensors_[0], vxOpmap_tbl[input1_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

std::shared_ptr<tim::vx::Operation> Mean::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxReduceMeanAttrs tvx_attrs(call_);
  return graph->CreateOperation<tim::vx::ops::ReduceMean>(tvx_attrs.axis, tvx_attrs.keepdims);
}

bool is_float_equal(float a, float b, float tol) {
  return fabs(a - b) < tol;
}

float unquantize(float val, float scale, float zero_point) {
  return (val - zero_point) * scale;
}

void VsiNpuQnnClip::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                             std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());
  const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
  auto expr = GetRef<Expr>(callnode);
  // extract calls in pattern
  Call call = Downcast<Call>(expr);
  if (call->args[0]->IsInstance<CallNode>()) {
    call = Downcast<Call>(call->args[0]);
  }
  auto op = CreateOperation(graph, call, vxOpmap_tbl);
  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
};

std::shared_ptr<tim::vx::Operation> VsiNpuQnnClip::CreateOperation(
    std::shared_ptr<tim::vx::Graph> graph, Call call,
    std::map<Expr, std::shared_ptr<OpSetup>> &vxOpmap_tbl) {
  TvxClipAttrs tvx_attrs(call);
  auto output = vxOpmap_tbl[expr_key_]->specs_[0].quantization_;
  if (output.Type() == tim::vx::QuantType::NONE) {
    float tol = 1e-3;
    if (is_float_equal(tvx_attrs.min, 0.0, tol) &&
        is_float_equal(tvx_attrs.max, 6.0, tol)) {
      return graph->CreateOperation<tim::vx::ops::Relu6>();
    }
    if (is_float_equal(tvx_attrs.min, 0.0, tol) &&
        is_float_equal(tvx_attrs.max, std::numeric_limits<float>::max(), tol)) {
      return graph->CreateOperation<tim::vx::ops::Relu>();
    }
    return graph->CreateOperation<tim::vx::ops::Clip>(tvx_attrs.min,
                                                      tvx_attrs.max);
  } else {
    float min =
        unquantize(tvx_attrs.min, output.Scales()[0], output.ZeroPoints()[0]);
    float max =
        unquantize(tvx_attrs.max, output.Scales()[0], output.ZeroPoints()[0]);
    float uint8_max =
        unquantize(255.0, output.Scales()[0], output.ZeroPoints()[0]);
    float tol = output.Scales()[0];
    if (is_float_equal(min, 0.0, tol) && is_float_equal(max, 6.0, tol)) {
      return graph->CreateOperation<tim::vx::ops::Relu6>();
    }
    if (is_float_equal(min, 0.0, tol) && is_float_equal(max, uint8_max, tol)) {
      return graph->CreateOperation<tim::vx::ops::Relu>();
    }
    return graph->CreateOperation<tim::vx::ops::Clip>(min, max);
  }
}

std::shared_ptr<tim::vx::Operation> LeakyRelu::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxLeakyReluAttrs tvx_attrs(call_);
  return graph->CreateOperation<tim::vx::ops::LeakyRelu>(tvx_attrs.alpha);
}

std::shared_ptr<tim::vx::Operation> VsiNpuQnnLeakyRelu::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxLeakyReluAttrs tvx_attrs(op_);
  return graph->CreateOperation<tim::vx::ops::LeakyRelu>(tvx_attrs.alpha);
}

void VsiNpuQnnDense::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                  std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  constexpr uint32_t kRequant_output_scale_idx = 3;
  constexpr uint32_t kRequant_output_zp_idx = 4;

  const CallNode* callnode = cn->op.as<FunctionNode>()->body.as<CallNode>();
  auto expr = GetRef<Expr>(callnode);
  // extract calls in pattern
  Call requantize = Downcast<Call>(expr);
  Call add = Downcast<Call>(requantize->args[0]);
  dense_ = Downcast<Call>(add->args[0]);

  if(vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::UINT8){
  using Input_Field = Field_ASYMM_U8<0, 4, 2, tim::vx::TensorAttribute::TRANSIENT,
                                     tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;
  using Weight_Field = Field_ASYMM_U8<1, 5, 3, tim::vx::TensorAttribute::CONSTANT,
                                      tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;

  using Bias_Field = Field_ASYMM_U8<1, 1, 2, tim::vx::TensorAttribute::CONSTANT,
                                    tim::vx::DataType::INT32, tim::vx::QuantType::ASYMMETRIC>;

  input_key_ = call_->args[Input_Field::arg_pos];
  weight_key_ = dense_->args[Weight_Field::arg_pos];
  bias_key_ = add->args[Bias_Field::arg_pos];

  vxOpmap_tbl[input_key_] =
      std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(dense_, dense_));
  tim::vx::TensorSpec weight_spec = Weight_Field::AsTimVxTensorSpec(dense_, dense_);
  tim::vx::TensorSpec bias_spec = Bias_Field::AsTimVxTensorSpec(add, requantize);
  vxOpmap_tbl[weight_key_] =
      std::make_shared<OpSetup>(Weight_Field::AsTimVxTensorSpec(dense_, dense_));
  bias_spec.shape_.resize(1);
  bias_spec.quantization_.Scales()[0] =
      weight_spec.quantization_.Scales()[0] *
      vxOpmap_tbl[input_key_]->specs_[0].quantization_.Scales()[0];

  vxOpmap_tbl[bias_key_] = std::make_shared<OpSetup>(bias_spec);
  } else if (vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::INT8) {
    using Input_Field = Field_ASYMM_U8<0, 4, 2, tim::vx::TensorAttribute::TRANSIENT,
                                       tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;
    using Weight_Field = Field_ASYMM_U8<1, 5, 3, tim::vx::TensorAttribute::CONSTANT,
                                        tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;

    using Bias_Field = Field_ASYMM_U8<1, 1, 2, tim::vx::TensorAttribute::CONSTANT,
                                      tim::vx::DataType::INT32, tim::vx::QuantType::ASYMMETRIC>;

    input_key_ = call_->args[Input_Field::arg_pos];
    weight_key_ = dense_->args[Weight_Field::arg_pos];
    bias_key_ = add->args[Bias_Field::arg_pos];

    vxOpmap_tbl[input_key_] =
        std::make_shared<OpSetup>(Input_Field::AsTimVxTensorSpec(dense_, dense_));
    tim::vx::TensorSpec weight_spec = Weight_Field::AsTimVxTensorSpec(dense_, dense_);
    tim::vx::TensorSpec bias_spec = Bias_Field::AsTimVxTensorSpec(add, requantize);
    vxOpmap_tbl[weight_key_] =
        std::make_shared<OpSetup>(Weight_Field::AsTimVxTensorSpec(dense_, dense_));
    bias_spec.shape_.resize(1);
    bias_spec.quantization_.Scales()[0] =
        weight_spec.quantization_.Scales()[0] *
        vxOpmap_tbl[input_key_]->specs_[0].quantization_.Scales()[0];

    vxOpmap_tbl[bias_key_] = std::make_shared<OpSetup>(bias_spec);
  }
  UpdateOutputQuantInfo(requantize, kRequant_output_scale_idx, kRequant_output_zp_idx, quant_info);
};

void VsiNpuQnnDense::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, bias_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, weight_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto op = graph->CreateOperation<tim::vx::ops::FullyConnected>(
      0, vxOpmap_tbl[bias_key_]->specs_[0].shape_[0]);

  (*op).BindInputs({vxOpmap_tbl[input_key_]->ptensors_[0], vxOpmap_tbl[weight_key_]->ptensors_[0],
                    vxOpmap_tbl[bias_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void Conv::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                          std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxConv2dAttrs tvx_attrs(call_);
  UpdateInputTableInfo(vxOpmap_tbl, input0_key_, graph.get());

  auto input1_spec = vxOpmap_tbl[input1_key_]->specs_[0];
  uint32_t kernel_ic = input1_spec.shape_[1];
  uint32_t kernel_oc = input1_spec.shape_[0];

  UpdateInputTableInfo(vxOpmap_tbl, input1_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  bool is_depthwise_conv =
      (static_cast<uint32_t>(tvx_attrs.groups) ==
       vxOpmap_tbl[input0_key_]->specs_[0].shape_[0]);  // group == input_channel
  auto op = graph->CreateOperation<tim::vx::ops::Conv2d>(
      is_depthwise_conv ? kernel_ic : kernel_oc, tim::vx::PadType::SAME,
      std::array<uint32_t, 2>{tvx_attrs.kernel_size[0], tvx_attrs.kernel_size[1]},
      std::array<uint32_t, 2>{tvx_attrs.strides[0], tvx_attrs.strides[1]},
      std::array<uint32_t, 2>{tvx_attrs.dilation[0], tvx_attrs.dilation[1]},
      std::array<uint32_t, 4>{tvx_attrs.padding[0], tvx_attrs.padding[1], tvx_attrs.padding[2],
                              tvx_attrs.padding[3]},
      is_depthwise_conv ? 1 : 0, tim::vx::DataLayout::CWHN, tim::vx::DataLayout::OcIcWH);

  (*op).BindInputs(
      {vxOpmap_tbl[input0_key_]->ptensors_[0], vxOpmap_tbl[input1_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void Quantize::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                            std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_Float32<0, tim::vx::TensorAttribute::TRANSIENT>;
  constexpr uint32_t out_scale_idx = 1;
  constexpr uint32_t out_zp_idx = 2;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input_key_ = call_->args[Input_0::arg_pos];

  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_));
  UpdateOutputQuantInfo(call_, out_scale_idx, out_zp_idx, quant_info);
};

std::shared_ptr<tim::vx::Operation> Quantize::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::DataConvert>();
}

void Dequantize::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                              std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_ASYMM_U8<0, 1, 2, tim::vx::TensorAttribute::TRANSIENT,
                                 tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input_key_ = call_->args[Input_0::arg_pos];

  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_, call_));
  (void)quant_info;
};

std::shared_ptr<tim::vx::Operation> Dequantize::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::DataConvert>();
}

void QnnRequantize::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                              std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  constexpr uint32_t out_scale_idx = 3;
  constexpr uint32_t out_zp_idx = 4;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);

  if(vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::INT8){
    using Input_0 = Field_ASYMM_U8<0, 1, 2, tim::vx::TensorAttribute::TRANSIENT,
                                 tim::vx::DataType::UINT8, tim::vx::QuantType::ASYMMETRIC>;
    input_key_ = call_->args[Input_0::arg_pos];
    vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_, call_));
  }else if(vxOpmap_tbl[expr_key_]->specs_[0].datatype_ == tim::vx::DataType::UINT8){
    using Input_0 = Field_ASYMM_U8<0, 1, 2, tim::vx::TensorAttribute::TRANSIENT,
                                 tim::vx::DataType::INT8, tim::vx::QuantType::ASYMMETRIC>;
    input_key_ = call_->args[Input_0::arg_pos];
    vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_, call_));
  }

  UpdateOutputQuantInfo(call_, out_scale_idx, out_zp_idx, quant_info);
};

std::shared_ptr<tim::vx::Operation> QnnRequantize::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  return graph->CreateOperation<tim::vx::ops::DataConvert>();
}

void NoTypeOpSetup::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                 std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input = Field_NoDType<0, tim::vx::TensorAttribute::TRANSIENT>;
  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input_key_ = call_->args[Input::arg_pos];

  if (vxOpmap_tbl.find(input_key_) != vxOpmap_tbl.end()) {
    return;
  }

  tim::vx::Quantization output_quant = vxOpmap_tbl[expr_key_]->specs_[0].quantization_ ;
  if(output_quant.Type() == tim::vx::QuantType::NONE){
    auto input_callback =
      std::make_shared<CallbackExpr>(input_key_, vxOpmap_tbl[expr_key_]->pCallbackexpr_);
    vxOpmap_tbl[input_key_] =
      std::make_shared<OpSetup>(Input::AsTimVxTensorSpec(call_), input_callback);
  }else{
    vxOpmap_tbl[input_key_] =
      std::make_shared<OpSetup>(Input::AsTimVxTensorSpec(call_));
      vxOpmap_tbl[input_key_]->specs_[0].SetQuantization(output_quant);
  }

  auto dtype = input_key_->checked_type().as<TensorTypeNode>()->dtype;

  if (dtype.is_float()) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::FLOAT32);
  } else if (dtype.is_uint()) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::UINT8);
  } else if (dtype.is_int() && dtype.bits() == 8) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::INT8);
  } else if (dtype.is_int()) {
    vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::INT32);
  }
}

std::shared_ptr<tim::vx::Operation> Reshape::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxReshapeAttrs tvx_attrs(call_);
  std::reverse(tvx_attrs.newshape.begin(), tvx_attrs.newshape.end());
  return graph->CreateOperation<tim::vx::ops::Reshape>(tvx_attrs.newshape);
}

std::shared_ptr<tim::vx::Operation> Squeeze::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxSqueezeAttrs tvx_attrs(call_);
  return graph->CreateOperation<tim::vx::ops::Squeeze>(tvx_attrs.axis);
}

std::shared_ptr<tim::vx::Operation> DepthtoSpace::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxDepthtoSpaceAttrs tvx_attrs(call_);
  return graph->CreateOperation<tim::vx::ops::DepthToSpace>(tvx_attrs.block_size,tim::vx::DataLayout::CWHN);
}

void ArgMax::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                            std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxReduceMeanAttrs tvx_attrs(call_);
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto input_sec = vxOpmap_tbl[input_key_]->specs_[0];
  auto op =
      graph->CreateOperation<tim::vx::ops::ArgMax>(input_sec.shape_.size() - 1 - tvx_attrs.axis[0]);

  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void ArgMin::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                            std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxReduceMeanAttrs tvx_attrs(call_);
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto input_sec = vxOpmap_tbl[input_key_]->specs_[0];
  auto op =
      graph->CreateOperation<tim::vx::ops::ArgMin>(input_sec.shape_.size() - 1 - tvx_attrs.axis[0]);

  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

std::shared_ptr<tim::vx::Operation> Resize::CreateOperation(std::shared_ptr<tim::vx::Graph> graph) {
  TvxResizeAttrs tvx_attrs(call_);
  bool align_corners = false;
  bool half_pixel_centers = false;
  if (tvx_attrs.coordinate_transformation_mode == "align_corners") {
    align_corners = true;
  } else if (tvx_attrs.coordinate_transformation_mode == "half_pixel") {
    half_pixel_centers = true;
  }

  return graph->CreateOperation<tim::vx::ops::Resize>(
      tvx_attrs.method == "bilinear" ? tim::vx::ResizeType::BILINEAR
                                     : tim::vx::ResizeType::NEAREST_NEIGHBOR,
      0.0f, align_corners, half_pixel_centers, tvx_attrs.size[0],
      tvx_attrs.size[1], tim::vx::DataLayout::CWHN);
}

std::shared_ptr<tim::vx::Operation> MaxPool2d::CreateOperation(
    std::shared_ptr<tim::vx::Graph> graph) {
  TvxPool2DAttrs tvx_attrs(call_, TvxPool2DAttrs::kMaxPool);
  return graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::MAX, tvx_attrs.pad_type,
      std::array<uint32_t, 2>{tvx_attrs.pool_size[0], tvx_attrs.pool_size[1]},
      std::array<uint32_t, 2>{tvx_attrs.strides[0], tvx_attrs.strides[1]}, tvx_attrs.ceil_mode,
      tim::vx::DataLayout::CWHN);
}

void Pad::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                         std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  TvxPadAttrs tvx_attrs(call_);
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  auto pad_width = tvx_attrs.pad_width;

  uint32_t size_0 = pad_width.size();
  uint32_t size_1 = pad_width[0].size();
  std::vector<uint32_t> front_size;
  std::vector<uint32_t> back_size;

  std::vector<uint32_t> pad_size;

  for (uint32_t i = 0; i < size_0; i++) {
    for (uint32_t j = 0; j < size_1; j++) {
      pad_size.push_back(pad_width[i][j]);
    }
  }

  for (int i = pad_size.size() - 1; i >= 0; i -= 2) {
    back_size.push_back(pad_size[i]);
    front_size.push_back(pad_size[i - 1]);
  }

  auto op = graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size, 0);

  (*op).BindInput(vxOpmap_tbl[input_key_]->ptensors_[0]);
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

void DeConv::SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                                 std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  using Input_0 = Field_NoDType<0, tim::vx::TensorAttribute::TRANSIENT>;
  using Input_1 = Field_NoDType<1, tim::vx::TensorAttribute::CONSTANT>;

  call_ = GetRef<Call>(cn);
  expr_key_ = GetRef<Expr>(cn);
  input_key_ = call_->args[Input_0::arg_pos];
  weight_key_ = call_->args[Input_1::arg_pos];

  if (vxOpmap_tbl.find(input_key_) != vxOpmap_tbl.end()) {
    return;
  }

  quant_info.SetType(tim::vx::QuantType::ASYMMETRIC).SetScales({1.0}).SetZeroPoints({0});

  vxOpmap_tbl[input_key_] = std::make_shared<OpSetup>(Input_0::AsTimVxTensorSpec(call_));
  vxOpmap_tbl[input_key_]->specs_[0].SetQuantization(quant_info);

  vxOpmap_tbl[weight_key_] = std::make_shared<OpSetup>(Input_1::AsTimVxTensorSpec(call_));
  vxOpmap_tbl[weight_key_]->specs_[0].SetQuantization(quant_info);

  vxOpmap_tbl[input_key_]->specs_[0].SetDataType(tim::vx::DataType::UINT8);
  vxOpmap_tbl[weight_key_]->specs_[0].SetDataType(tim::vx::DataType::UINT8);
}

void DeConv::SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                            std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
  UpdateInputTableInfo(vxOpmap_tbl, input_key_, graph.get());
  UpdateInputTableInfo(vxOpmap_tbl, weight_key_, graph.get());
  UpdateOutputTableInfo(vxOpmap_tbl, expr_key_, graph.get());

  int oc_count = static_cast<int>(vxOpmap_tbl[expr_key_]->specs_[0].shape_[0]);

  TvxDeconvAttrs tvx_attrs(call_);

  auto op = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      oc_count, tvx_attrs.pad_type,
      std::array<uint32_t, 2>{tvx_attrs.kernel_size[0], tvx_attrs.kernel_size[1]},
      std::array<uint32_t, 2>{tvx_attrs.strides[0], tvx_attrs.strides[1]},
      std::array<uint32_t, 2>{0, 0},
      std::array<uint32_t, 4>{tvx_attrs.padding[0], tvx_attrs.padding[1], tvx_attrs.padding[2],
                              tvx_attrs.padding[3]},
      tvx_attrs.groups, tim::vx::DataLayout::CWHN, tim::vx::DataLayout::WHIcOc);
  (*op).BindInputs({vxOpmap_tbl[input_key_]->ptensors_[0],vxOpmap_tbl[weight_key_]->ptensors_[0]});
  (*op).BindOutput(vxOpmap_tbl[expr_key_]->ptensors_[0]);
}

}  // namespace op_map
}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
