#ifndef TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_FIELD_H_
#define TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_FIELD_H_

#include "helper.h"
#include "tim/vx/tensor.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {
namespace op_map {
void shape_setup(const Call& c, uint32_t arg_idx, tim::vx::ShapeType& result_shape);

template <uint32_t Idx, uint32_t Scale_Idx, uint32_t Zp_Idx,
          tim::vx::QuantType QType = tim::vx::QuantType::ASYMMETRIC>
struct Field_Quant_Operand {
  static const uint32_t arg_pos = Idx;

  static tim::vx::TensorSpec AsTimVxTensorSpec(const Call& c, const Call& c1) {
    tim::vx::ShapeType shape;
    tim::vx::DataType dataType;
    tim::vx::TensorAttribute role;
    float scale = 0;
    int32_t zp = 0;

    Expr expr = c->args[Idx];
    auto dtype = expr->checked_type().as<TensorTypeNode>()->dtype;
    role = expr->IsInstance<ConstantNode>() ? tim::vx::TensorAttribute::CONSTANT
                                            : tim::vx::TensorAttribute::TRANSIENT;
    if (dtype.is_uint()) {
      dataType = tim::vx::DataType::UINT8;
    } else if (dtype.is_int() && dtype.bits() == 8) {
      dataType = tim::vx::DataType::INT8;
    } else if (dtype.is_int() && dtype.bits() == 32) {
      dataType = tim::vx::DataType::INT32;
    }

    shape_setup(c, Idx, shape);
    AsConstant(c1->args[Scale_Idx], &scale);
    AsConstant(c1->args[Zp_Idx], &zp);
    auto quant_spec = tim::vx::Quantization(QType, scale, zp);
    tim::vx::TensorSpec spec(dataType, shape, role, quant_spec);
    return spec;
  }
};

template <uint32_t Idx>
struct Field_NoQuant_Operand {
  static const uint32_t arg_pos = Idx;

  static tim::vx::TensorSpec AsTimVxTensorSpec(const Call& c) {
    tim::vx::ShapeType shape;
    tim::vx::DataType dataType;
    tim::vx::TensorAttribute role;

    Expr expr = c->args[Idx];

    auto dtype = expr->checked_type().as<TensorTypeNode>()->dtype;
    role = expr->IsInstance<ConstantNode>() ? tim::vx::TensorAttribute::CONSTANT
                                            : tim::vx::TensorAttribute::TRANSIENT;

    dataType = GetTvxType(dtype);

    shape_setup(c, Idx, shape);
    tim::vx::TensorSpec spec(dataType, shape, role);
    return spec;
  }
};

template <uint32_t Idx, uint32_t Scale_Idx, uint32_t Zp_Idx, tim::vx::TensorAttribute Role,
          tim::vx::DataType DType, tim::vx::QuantType QType>
struct Field_TUPLE_U8 {
  static const uint32_t arg_pos = Idx;

  static std::vector<tim::vx::TensorSpec> AsTimVxTensorSpec(const Call& c, const Call& c1) {
    auto input_node_tensors_type = c->args[Idx]->checked_type().as<TupleTypeNode>()->fields;
    auto input_node_scales = c->args[Scale_Idx].as<TupleNode>()->fields;
    auto input_node_zps = c->args[Zp_Idx].as<TupleNode>()->fields;

    std::vector<tim::vx::TensorSpec> specs;
    uint32_t input_node_num = input_node_tensors_type.size();
    for (uint32_t i = 0; i < input_node_num; i++) {
      tim::vx::ShapeType shape;
      std::transform(
          input_node_tensors_type[i].as<TensorTypeNode>()->shape.rbegin(),
          input_node_tensors_type[i].as<TensorTypeNode>()->shape.rend(), std::back_inserter(shape),
          [](const PrimExpr& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });
      float scale = 0;
      int32_t zp = 0;

      AsConstant<float>(input_node_scales[i], &scale);
      AsConstant<int>(input_node_zps[i], &zp);

      auto quant_spec = tim::vx::Quantization(QType, scale, zp);
      tim::vx::TensorSpec spec(DType, shape, Role, quant_spec);
      specs.push_back(spec);
    }
    return specs;
  }
};

}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace relay
}  // namespace tvm

#endif