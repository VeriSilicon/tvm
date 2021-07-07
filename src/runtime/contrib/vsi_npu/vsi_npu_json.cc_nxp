/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef USE_VSI_NPU_RUNTIME
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/softmax.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/resize.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/batchnorm.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/clip.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/dropout.h"
#include "tim/vx/ops/split.h"
#include "tim/vx/ops/stridedslice.h"
#include "tim/vx/ops/reduce.h"

#include "vsi_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

#ifdef USE_VSI_NPU_RUNTIME
class VsiNpuJSONRuntime : public JSONRuntimeBase {

 public:
  VsiNpuJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "vsi_npu_json"; }

  void Init(const Array<NDArray>& consts) override {
    // Setup constants entries for weights.
    SetupConstants(consts);

    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    BuildEngine();
  }

  void Run() override {
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      uint32_t eid = EntryID(nid, 0);
      if (nodes_[nid].GetOpType() == "input") {
	auto vsi_tensor = entry_out_tensor_[eid];

        void* data = data_entry_[eid]->data;
	int data_size = 1;
        for (int j = 0; j < data_entry_[eid]->ndim; j++) {
          data_size *= data_entry_[eid]->shape[j];
        }
        assert(vsi_tensor->CopyDataToTensor(data, data_size));
      }
    }

    assert(graph_->Run());

    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      void* data = data_entry_[eid]->data;

      auto vsi_tensor = entry_out_tensor_[eid];
      vsi_tensor->CopyDataFromTensor(data);
    }
  }
 private:

  void BuildEngine() {
    context_ = tim::vx::Context::Create();
    graph_ = context_->CreateGraph();

    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
	LOG(INFO) << "Build op: " << op_name;
        if ("nn.batch_flatten" == op_name or "reshape" == op_name) {
	  Reshape(nid);
        } else if ("nn.dense" == op_name or "qnn.dense" == op_name) {
	  Dense(nid);
        } else if ("nn.relu" == op_name or "sigmoid" == op_name) {
          Activation(nid);
        } else if ("nn.softmax" == op_name || "qnn.softmax" == op_name) {
          Softmax(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.conv2d" == op_name || "qnn.conv2d" == op_name) {
          Conv2D(nid);
        } else if (("nn.global_avg_pool2d" == op_name) || ("nn.global_max_pool2d" == op_name)) {
          GlobalPool2d(nid);
        } else if ("nn.max_pool2d" == op_name || "nn.avg_pool2d" == op_name || "qnn.avg_pool2d" == op_name) {
          Pool2d(nid);
        } else if ("add" == op_name || "qnn.add" == op_name or "multiply" == op_name
                   or "divide" == op_name) {
          Elementwise(nid);
        } else if ("clip" == op_name) {
          Clip(nid);
        } else if ("layout_transform" == op_name or "transpose" == op_name) {
          Permute(nid);
        } else if ("nn.dropout" == op_name) {
          Dropout(nid);
        } else if ("concatenate" == op_name || "qnn.concatenate" == op_name) {
          Concat(nid);
        } else if ("image.resize" == op_name) {
          Resize(nid);
        } else if ("split" == op_name) {
          Split(nid);
        } else if ("strided_slice" == op_name) {
          StridedSlice(nid);
        } else if ("mean" == op_name) {
          Reduce(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    assert(graph_->Compile());
    LOG(INFO) << "Build graph successfully" << std::endl;
  }

  void Reshape(const size_t& nid) {
    auto node = nodes_[nid];
    JSONGraphNodeEntry out_entry(nid, 0);
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();

    CHECK(inputs.size() == 1U) << "Flatten layer requires 1 inputs.";

    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    std::vector<uint32_t> output_shape = vsi_output->GetShape();

    auto reshape = graph_->CreateOperation<tim::vx::ops::Reshape>(output_shape);
    (*reshape).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(reshape);
  }

  void Resize(const size_t& nid) {
    auto node = nodes_[nid];
    JSONGraphNodeEntry out_entry(nid, 0);
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    std::vector<std::string> size = node.GetAttr<std::vector<std::string>>("size");
    std::string method = node.GetAttr<std::vector<std::string>>("method")[0];
    std::string mode = node.GetAttr<std::vector<std::string>>("coordinate_transformation_mode")[0];
    std::string layout = node.GetAttr<std::vector<std::string>>("layout")[0];

    std::vector<int32_t> vsi_size;
    tim::vx::ResizeType vsi_type;
    bool align_corners = 0;
    bool half_pixel = 0;

    vsi_size.push_back(std::stoi(size[0]));
    vsi_size.push_back(std::stoi(size[1]));
    if(method == "bilinear") {
      vsi_type = tim::vx::ResizeType::BILINEAR;
    } else if (method == "nearest_neighbor") {
      vsi_type = tim::vx::ResizeType::NEAREST_NEIGHBOR;
    } else {
      LOG(FATAL) << "Unsupported method for Resize layer " << method;
    }
    if (mode == "half_pixel") {
      half_pixel = 1;
    } else if (mode == "align_corners") {
      align_corners = 1;
    }

    CHECK(inputs.size() == 1U) << "Resize layer requires 1 inputs.";

    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    if(layout == "NHWC") {
        vsi_input = PermuteVsiTensor(vsi_input, {1, 2, 0, 3}, true);
        vsi_output = PermuteVsiTensor(vsi_output, {2, 0, 1, 3}, false);
    } else {
        CHECK(layout == "NCHW") << "Resize layer requires 1 inputs.";
    }

    auto resize = graph_->CreateOperation<tim::vx::ops::Resize>(vsi_type, 0, align_corners,
                                                                half_pixel, vsi_size[0], vsi_size[1]);
    (*resize).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(resize);
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    // Collect inputs and outputs, handling both nn.dense and qnn.dense cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    JSONGraphNodeEntry out_entry(nid, 0);
    
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;

    bool has_bias;
    if (node.GetOpName() == "qnn.dense") {
      //qnn.densn
      CHECK(num_inputs >= 10U && num_inputs <= 11U)
          << "Quantized convolution requires 11 inputs with a bias, 9 inputs without.";
      has_bias = num_inputs == 11;
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3]));
      if (has_bias) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[6], &inputs[9], &inputs[10]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Fully connected (dense) layer requires 3 inputs with a bias, 2 inputs without.";
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto weight_tensor = vsi_inputs[1];
    auto fc = graph_->CreateOperation<tim::vx::ops::FullyConnected>(1, weight_tensor->GetShape()[1]);
    (*fc).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(fc);
  }

  void Activation(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<tim::vx::Tensor> vsi_input;
    std::shared_ptr<tim::vx::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);

    std::shared_ptr<tim::vx::Operation> _op;
    if ("nn.relu" == op_name) {
      _op = graph_->CreateOperation<tim::vx::ops::Relu>();
    } else if ("sigmoid" == op_name){
      _op = graph_->CreateOperation<tim::vx::ops::Sigmoid>();
    }
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(_op);
  }

  void Elementwise(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto inputs = node.GetInputs();

    CHECK(inputs.size() >= 2U) << "BatchNormal layer requires at least 2 inputs.";

    JSONGraphNodeEntry out_entry(nid, 0);
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;

    if (op_name == "qnn.add") {
      auto input_cnt = inputs.size();
      input_cnt = (input_cnt - 2) / 3; //Each input has 3 tensor(data, scale, offset)
      for (size_t j = 0; j < input_cnt; j ++) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[j],
                   &inputs[j + input_cnt], &inputs[j + input_cnt * 2]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry,
                   &inputs[input_cnt * 3], &inputs[input_cnt * 3 + 1]));
    } else {
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    std::shared_ptr<tim::vx::Operation> _op;
    if ("add" == op_name || "qnn.add" == op_name) {
        _op = graph_->CreateOperation<tim::vx::ops::Add>();
    } else if ("multiply" == op_name) {
        _op = graph_->CreateOperation<tim::vx::ops::Multiply>();
    } else if ("divide" == op_name) {
        _op = graph_->CreateOperation<tim::vx::ops::Div>();
    }
    (*_op).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(_op);
  }

  void Clip(const size_t& nid) {
    auto node = nodes_[nid];
    auto inputs = node.GetInputs();
    std::string min = node.GetAttr<std::vector<std::string>>("a_min")[0];
    std::string max = node.GetAttr<std::vector<std::string>>("a_max")[0];

    CHECK(inputs.size() == 1U) << "Clip layer requires 1 input.";

    JSONGraphNodeEntry out_entry(nid, 0);
    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto clip = graph_->CreateOperation<tim::vx::ops::Clip>(std::stof(min), std::stof(max));
    (*clip).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(clip);
  }

  void Split(const size_t& nid) {
    auto node = nodes_[nid];
    auto inputs = node.GetInputs();
    auto outputs_num = node.GetNumOutput();
    auto tvm_axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    auto slices_num = std::stoi(node.GetAttr<std::vector<std::string>>("indices_or_sections")[0]);
    auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];
    auto vx_axis = ConvertAxis(tvm_axis, shape_tvm.size());

    CHECK((uint32_t)slices_num == outputs_num) << "Split layer slices number doesn't match outputs number.";

    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;
    std::vector<uint32_t> slices;
    for (int i = 0; i < slices_num; i++){
      JSONGraphNodeEntry out_entry(nid, i);
      auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());
      vsi_outputs.push_back(vsi_output);
      slices.push_back(vsi_output->GetShape()[vx_axis]);
    }
    auto split = graph_->CreateOperation<tim::vx::ops::Split>(vx_axis, slices);
    (*split).BindInput(vsi_input).BindOutputs(vsi_outputs);
    ops_.push_back(split);
  }

  void StridedSlice(const size_t& nid) {
    auto node = nodes_[nid];
    auto inputs = node.GetInputs();
    auto begin = node.GetAttr<std::vector<std::string>>("begin");
    auto end = node.GetAttr<std::vector<std::string>>("end");
    auto strides = node.GetAttr<std::vector<std::string>>("strides");

    std::vector<int32_t> vx_begin;
    for (unsigned int i = 0; i < begin.size(); i ++) {
      vx_begin.push_back(std::stoi(begin[begin.size() - i - 1]));
    }
    std::vector<int32_t> vx_end;
    for (unsigned int i = 0; i < end.size(); i ++) {
      vx_end.push_back(std::stoi(end[end.size() - i - 1]));
    }
    std::vector<int32_t> vx_stride;
    for (unsigned int i = 0; i < strides.size(); i ++) {
      vx_stride.push_back(std::stoi(strides[strides.size() - i - 1]));
    }

    JSONGraphNodeEntry out_entry(nid, 0);
    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output =  MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto slice = graph_->CreateOperation<tim::vx::ops::StridedSlice>(vx_begin, vx_end, vx_stride, 0, 0, 0);
    (*slice).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(slice);
  }

  void Reduce(const size_t& nid) {
    auto node = nodes_[nid];
    auto inputs = node.GetInputs();
    auto axis = node.GetAttr<std::vector<std::string>>("axis");
    auto keepdims = node.GetAttr<std::vector<std::string>>("keepdims")[0];
    auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];

    std::vector<int32_t> vx_axis;
    for (unsigned int i = 0; i < axis.size(); i ++) {
      auto tmp = ConvertAxis(std::stoi(axis[axis.size() - i - 1]), shape_tvm.size());
      vx_axis.push_back(tmp);
    }

    bool vx_keepdims;
    if (keepdims == "True") {
      vx_keepdims = true;
    } else {
      vx_keepdims = false;
    }

    JSONGraphNodeEntry out_entry(nid, 0);
    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output =  MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto op = graph_->CreateOperation<tim::vx::ops::ReduceMean>(vx_axis, vx_keepdims);
    (*op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(op);
  }

  void Permute(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto inputs = node.GetInputs();
    std::vector<uint32_t> perm;

    if ("layout_transform" == op_name) {
      std::string src_layout = node.GetAttr<std::vector<std::string>>("src_layout")[0];
      std::string dst_layout = node.GetAttr<std::vector<std::string>>("dst_layout")[0];
      if (src_layout == "NHWC" && dst_layout == "NCHW"){
        perm = {1, 2, 0, 3};
      } else if (src_layout == "NCHW" && dst_layout == "NHWC") {
        perm = {2, 0, 1, 3};
      } else {
        LOG(FATAL) << "Unsupported layout transform from " << src_layout << " to " << dst_layout;
      }
    } else if ("transpose" == op_name){
      auto tvm_perm = node.GetAttr<std::vector<std::string>>("axes");
      for (unsigned int i = 0; i < tvm_perm.size(); i ++) {
        perm.push_back(std::stoi(tvm_perm[tvm_perm.size() - i - 1]));
      }
    }

    JSONGraphNodeEntry out_entry(nid, 0);
    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output =  MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto permute = graph_->CreateOperation<tim::vx::ops::Transpose>(perm);
    (*permute).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(permute);
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto inputs = node.GetInputs();

    /* inputs[0]: input,
     * inputs[1]: gamma,
     * inputs[2]: beta,
     * inputs[3]: moving_mean,
     * inputs[4]: moving_var
     */

    CHECK(inputs.size() == 5U) << "BatchNormal layer requires 5 inputs.";

    JSONGraphNodeEntry out_entry(nid, 0);

    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;

    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[3]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[4]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[2]));

    vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));

    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    auto bn = graph_->CreateOperation<tim::vx::ops::BatchNorm>(epsilon);
    (*bn).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(bn);
  }

  void Pool2d(const size_t& nid) {

    tim::vx::PoolType pool_type;
    tim::vx::RoundType round_type;

    auto node = nodes_[nid];
    auto inputs = node.GetInputs();

    /* inputs[0]: input data,
     * attr: pool_size,
     * attr: strides,
     * attr: padding,
     */

    JSONGraphNodeEntry out_entry(nid, 0);

    std::vector<std::string> tvm_pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
    std::vector<std::string> tvm_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> tvm_pad = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> tvm_ceil_mode = node.GetAttr<std::vector<std::string>>("ceil_mode");

    if (std::stoi(tvm_ceil_mode[0]) > 0) {
        round_type = tim::vx::RoundType::CEILING;
    } else {
        round_type = tim::vx::RoundType::FLOOR;
    }

    if (node.GetOpName() == "nn.max_pool2d") {
      pool_type = tim::vx::PoolType::MAX;
    } else if (node.GetOpName() == "nn.avg_pool2d" || node.GetOpName() == "qnn.avg_pool2d") {
      pool_type = tim::vx::PoolType::AVG;
    } else {
      LOG(FATAL) << "Pooling type not supported: " << node.GetOpName();
    }

    std::array<uint32_t, 2> vsi_ksize;
    vsi_ksize[0] = std::stoi(tvm_pool_size[0]);
    vsi_ksize[1] = std::stoi(tvm_pool_size[1]);

    std::array<uint32_t, 2> vsi_stride;
    vsi_stride[0] = std::stoi(tvm_strides[0]);
    vsi_stride[1] = std::stoi(tvm_strides[1]);

    std::array<uint32_t, 4> vsi_pad;
    vsi_pad[0] = std::stoi(tvm_pad[1]);
    vsi_pad[1] = std::stoi(tvm_pad[3]);
    vsi_pad[2] = std::stoi(tvm_pad[0]);
    vsi_pad[3] = std::stoi(tvm_pad[2]);

    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto pool = graph_->CreateOperation<tim::vx::ops::Pool2d>(pool_type, tim::vx::PadType::AUTO,
                                                         vsi_ksize, vsi_stride, vsi_pad, round_type);
    (*pool).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(pool);
  }

  void GlobalPool2d(const size_t& nid) {

    tim::vx::PoolType pool_type;

    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    JSONGraphNodeEntry out_entry(nid, 0);

    if (node.GetOpName() == "nn.global_max_pool2d") {
      pool_type = tim::vx::PoolType::MAX;
    } else if (node.GetOpName() == "nn.global_avg_pool2d") {
      pool_type = tim::vx::PoolType::AVG;
    } else {
      LOG(FATAL) << "Pooling type not supported: " << node.GetOpName();
    }

    std::shared_ptr<tim::vx::Tensor> vsi_input;
    std::shared_ptr<tim::vx::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);
    auto vsi_shap = vsi_input->GetShape();
    //layout is swapped, NCHW-->WHCN, [0]:W, [1]:H
    std::array<uint32_t, 2> ksize = {vsi_shap[0], vsi_shap[1]};
    //stride
    std::array<uint32_t, 2> stride = {1, 1};
    std::array<uint32_t, 4> pad = {0, 0, 0, 0};

    auto _op = graph_->CreateOperation<tim::vx::ops::Pool2d>(pool_type, tim::vx::PadType::AUTO, ksize, stride, pad);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(_op);
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto inputs = node.GetInputs();
    //softmax aixs
    auto axis_data_tvm = node.GetAttr<std::vector<std::string>>("axis")[0];
    auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];
    uint32_t axis_data_vsi = 1;

    axis_data_vsi = ConvertAxis(std::stoi(axis_data_tvm), shape_tvm.size());

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<tim::vx::Tensor> vsi_input;
    std::shared_ptr<tim::vx::Tensor> vsi_output;
    std::cout << "inputs.size" << inputs.size() << std::endl;
    CHECK(inputs.size() == 1U || inputs.size() == 5U)
          << "Softmax requires 5 inputs with quantization, 1 inputs without.";
    if (inputs.size() == 5) {
      vsi_input = MakeVSITensorFromJSONEntry(inputs[0], &inputs[1], &inputs[2]);
      vsi_output = MakeVSITensorFromJSONEntry(out_entry, &inputs[3], &inputs[4]);
    } else {
      vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
      vsi_output = MakeVSITensorFromJSONEntry(out_entry);
    }

    //set beta to 1.0
    auto _op = graph_->CreateOperation<tim::vx::ops::Softmax>(1.0f, axis_data_vsi);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(_op);
  }

  void Dropout(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto inputs = node.GetInputs();

    auto ratio_tvm = node.GetAttr<std::vector<std::string>>("rate")[0];
    //auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];

    float ratio_vsi = 1.0;
    //ratio_vsi = ConvertAxis(std::stoi(ratio_tvm), shape_tvm.size());
    //ratio_vsi = std::stof(ratio_tvm);

    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;

    for (const auto& i : inputs) {
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
    }

    JSONGraphNodeEntry out_entry(nid, 0);
    vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));

    auto dropout = graph_->CreateOperation<tim::vx::ops::Dropout>(ratio_vsi);
    (*dropout).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(dropout);
  }

  void Concat(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto inputs = node.GetInputs();

    auto axis_tvm = node.GetAttr<std::vector<std::string>>("axis")[0];
    auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];

    uint32_t axis_vsi = 1;
    axis_vsi = ConvertAxis(std::stoi(axis_tvm), shape_tvm.size());

    auto input_cnt = inputs.size();

    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;

    JSONGraphNodeEntry out_entry(nid, 0);
    if (node.GetOpName() == "qnn.concatenate") {
      input_cnt = (input_cnt - 2) / 3; //Each input has 3 tensor(data, scale, offset)
      for (size_t j = 0; j < input_cnt; j ++) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[j],
			       	                        &inputs[j + input_cnt],
						       	&inputs[j + input_cnt * 2]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry,
			                               &inputs[input_cnt * 3],
						       &inputs[input_cnt * 3 + 1]));
    } else {
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto concat = graph_->CreateOperation<tim::vx::ops::Concat>(axis_vsi, input_cnt);
    (*concat).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(concat);
  }

  void Conv2D(const size_t& nid) {
    auto node = nodes_[nid];
    std::vector<std::string> pad = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");

    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    int channels = std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0]);

    // Collect inputs and outputs, handling both nn.conv2d and qnn.conv2d cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<tim::vx::Tensor>> vsi_outputs;
    size_t num_inputs = inputs.size();
    JSONGraphNodeEntry out_entry(nid, 0);
    bool has_bias;
    int32_t vsi_multiplier = 0;
    std::vector<int64_t> data_shape = nodes_[inputs[0].id_].GetOpShape()[0];
    std::vector<int64_t> weight_shape = nodes_[inputs[1].id_].GetOpShape()[0];
    std::vector<int64_t> bias_shape = {channels};

    if (groups == data_shape[1] && groups == weight_shape[0] && groups != 1) {
      vsi_multiplier = static_cast<int32_t>(weight_shape[1]);
      if (channels != vsi_multiplier) {
          CHECK(channels == weight_shape[0] * weight_shape[1]) << "Invalid channels for depthwise conv2d.";
	  weight_shape[0] = 1;
	  weight_shape[1] = channels;
      }
    }

    if (node.GetOpName() == "qnn.conv2d") {
      CHECK(num_inputs >= 10U && num_inputs <= 11U)
          << "Quantized convolution requires 11 inputs with a bias, 9 inputs without.";
      has_bias = num_inputs == 11;
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3], &weight_shape));
      if (has_bias) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[6], &inputs[9], &inputs[10], &bias_shape));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Convolution requires 3 inputs with a bias, 2 inputs without.";
      has_bias = num_inputs == 3;
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0]));
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1], nullptr, nullptr, &weight_shape));
      if (has_bias) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[2], nullptr, nullptr, &bias_shape));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    // TVM: top, left, bottom, right -> VSI: left, right, top, bottom
    auto weight_tensor = vsi_inputs[1];
    std::array<uint32_t, 4> vsi_pad;
    vsi_pad[0] = std::stoi(pad[1]);
    vsi_pad[1] = std::stoi(pad[3]);
    vsi_pad[2] = std::stoi(pad[0]);
    vsi_pad[3] = std::stoi(pad[2]);

    std::array<uint32_t, 2> vsi_strides;
    vsi_strides[0] = std::stoi(strides[0]);
    vsi_strides[1] = std::stoi(strides[1]);

    std::array<uint32_t, 2> vsi_dilation;
    vsi_dilation[0] = std::stoi(dilation[0]);
    vsi_dilation[1] = std::stoi(dilation[1]);

    std::array<uint32_t, 2> vsi_ksize;
    vsi_ksize[0] = weight_tensor->GetShape()[0];
    vsi_ksize[1] = weight_tensor->GetShape()[1];

    if (!has_bias) {
      vsi_inputs.push_back(MakeDummyBiasTensor(vsi_inputs[0]->GetDataType(),
			      {weight_tensor->GetShape()[3]}));
    }

    auto conv = graph_->CreateOperation<tim::vx::ops::Conv2d>(channels,
		    tim::vx::PadType::AUTO, vsi_ksize, vsi_strides, vsi_dilation,
		    vsi_pad, groups, vsi_multiplier);
    (*conv).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(conv);
  }


  void BatchFlatten(const size_t& nid) {
  }

  bool IsInputNode(const size_t& nid) {
    return std::find(input_nodes_.begin(), input_nodes_.end(), nid) != input_nodes_.end();
  }

  bool IsOutputNode(const size_t& nid) {
    int size = outputs_.size();
    for(int i = 0; i< size; i++) {
      if(outputs_[i].id_ == nid)
        return true;
    }
    return false;
  }

  /*!
   * \brief Create an VSI tensor given the JSON representation. If scale
   * and offset are given, then create a quantized VSI tensor.
   *
   * \param tensor The tensor to represent.
   * \param scale (optional) The scale of the tensor as an input.
   * \param offset (optional) The offset of the tensor as an input.
   * \return VSI Tensor.
   */
  std::shared_ptr<tim::vx::Tensor> MakeVSITensorFromJSONEntry(const JSONGraphNodeEntry& tensor,
                                                 JSONGraphNodeEntry* scale = nullptr,
                                                 JSONGraphNodeEntry* offset = nullptr,
                                                 std::vector<int64_t> *in_shape = nullptr) {
    tim::vx::Quantization vsi_quant;
    if (scale != nullptr && offset != nullptr) {
      auto scale_tensor = data_entry_[EntryID(*scale)];
      auto offset_tensor = data_entry_[EntryID(*offset)];
      std::vector<float> scale_data = GetVectorFromDLTensor<float>(scale_tensor);
      std::vector<int> offset_data = GetVectorFromDLTensor<int>(offset_tensor);
      CHECK(scale_data.size() == 1 && offset_data.size() == 1)
            << "Currently only per-layer quantization is supported in the VSI runtime.";
      vsi_quant = tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, scale_data[0], offset_data[0]);
    }

    return MakeVSITensorFromJSONEntry(tensor, vsi_quant, in_shape);
  }
  std::shared_ptr<tim::vx::Tensor> MakeVSITensorFromJSONEntry(const JSONGraphNodeEntry& tensor,
                                                 const tim::vx::Quantization vsi_quant,
                                                 std::vector<int64_t> *in_shape = nullptr) {
    auto eid = EntryID(tensor);

    if (entry_out_tensor_.count(eid) != 0) {
      //using the existed VSItensor
      return entry_out_tensor_[eid];
    }
    //create new VSItensor
    JSONGraphNode node = nodes_[tensor.id_];
    void* node_data = nullptr;
    tim::vx::TensorAttribute vsi_attr;

    if (node.GetOpType() == "const") {
      node_data = data_entry_[EntryID(tensor)]->data;
      vsi_attr = tim::vx::TensorAttribute::CONSTANT;
    } else if (IsInputNode(tensor.id_)) {
      vsi_attr = tim::vx::TensorAttribute::INPUT;
    } else if (IsOutputNode(tensor.id_)) {
      vsi_attr = tim::vx::TensorAttribute::OUTPUT;
    } else {
      vsi_attr = tim::vx::TensorAttribute::TRANSIENT;
    }

    auto vsi_tensor = MakeVSITensor(node, node_data, vsi_attr, vsi_quant, in_shape);
    entry_out_tensor_.insert({eid, vsi_tensor});
    return entry_out_tensor_[eid];
  }

  std::shared_ptr<tim::vx::Tensor> MakeDummyBiasTensor(tim::vx::DataType dtype,
		 		 tim::vx::ShapeType bias_shape) {
    std::vector<float> bias_data(bias_shape[0], 0);

    tim::vx::TensorSpec bias_spec(dtype, bias_shape,
		    tim::vx::TensorAttribute::CONSTANT);
    auto bias = graph_->CreateTensor(bias_spec, bias_data.data());
    dummy_tensor_.push_back(bias);

    return bias;
  }

  std::shared_ptr<tim::vx::Tensor> MakeVSITensor(const JSONGraphNode& tensor_rep, void* data,
				  tim::vx::TensorAttribute vsi_attr,
				  const tim::vx::Quantization vsi_quant,
                                  std::vector<int64_t> *in_shape = nullptr) {
    //VSI parameter
    tim::vx::ShapeType vsi_shape;
    tim::vx::DataType vsi_dtype;
    //TVM parameter
    std::vector<int64_t> tvm_shape;
    DLDataType tvm_dtype = tensor_rep.GetOpDataType()[0];

    if (in_shape != nullptr) {
      tvm_shape = *in_shape;
    } else {
      tvm_shape = tensor_rep.GetOpShape()[0];
    }
    if (tvm_shape.size() == 0) {
      vsi_shape.push_back(1);
    } else {
      for (unsigned int i = 0; i < tvm_shape.size(); i ++) {
        vsi_shape.push_back(tvm_shape[tvm_shape.size() - i - 1]);
      }
    }

    if (tvm_dtype.code == DLDataTypeCode::kDLFloat && tvm_dtype.bits == 32) {
      vsi_dtype = tim::vx::DataType::FLOAT32;
    } else if (tvm_dtype.code == DLDataTypeCode::kDLUInt && tvm_dtype.bits == 8) {
      vsi_dtype = tim::vx::DataType::UINT8;
    } else if (tvm_dtype.code == DLDataTypeCode::kDLInt && tvm_dtype.bits == 32) {
      vsi_dtype = tim::vx::DataType::INT32;
    } else {
      LOG(FATAL) << "Unsupported data type.";
    }

    auto input_spec = tim::vx::TensorSpec(vsi_dtype, vsi_shape, vsi_attr, vsi_quant);
    std::shared_ptr<tim::vx::Tensor> tensor;
    if (data != nullptr)
      tensor = graph_->CreateTensor(input_spec, data);
    else
      tensor = graph_->CreateTensor(input_spec);
    return tensor;
  }

  std::shared_ptr<tim::vx::Tensor> PermuteVsiTensor(std::shared_ptr<tim::vx::Tensor> tensor,
                                                std::vector<uint32_t> perm,
                                                bool input) {
    auto temp = graph_->CreateTensor(tensor->GetSpec().AsTransientSpec());

    std::shared_ptr<tim::vx::Operation> op = graph_->CreateOperation<tim::vx::ops::Transpose>(perm);
    if (input) {
      (*op).BindInput(tensor);
      (*op).BindOutput(temp);
    } else {
      (*op).BindInput(temp);
      (*op).BindOutput(tensor);
    }
    ops_.push_back(op);
    dummy_tensor_.push_back(temp);

    return temp;
  }

  std::shared_ptr<tim::vx::Context> context_;
  std::shared_ptr<tim::vx::Graph> graph_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>> entry_out_tensor_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> dummy_tensor_;
  std::vector<std::shared_ptr<tim::vx::Operation>> ops_;
};

#else

class VsiNpuJSONRuntime : public JSONRuntimeBase {

 public:
  VsiNpuJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "vsi_npu_json"; }

  void Init(const Array<NDArray>& consts) override {

  }

  void Run() override {
  }
};
#endif



runtime::Module VsiNpuJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<VsiNpuJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.VsiNpuJSONRuntimeCreate").set_body_typed(VsiNpuJSONRuntimeCreate);

#ifdef USE_VSI_NPU_RUNTIME
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_vsi_npu_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<VsiNpuJSONRuntime>);
#endif
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
