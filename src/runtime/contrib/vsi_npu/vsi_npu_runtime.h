#ifndef TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_NPU_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_NPU_RUNTIME_H_

#include <tvm/runtime/packed_func.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/tensor.h>

namespace tvm {
namespace runtime {
namespace vsi_npu {
struct TensorSpecIR {
  tim::vx::QuantType quant_type;
  int32_t channel_dim;
  std::vector<float> scales;
  std::vector<int32_t> zps;

  tim::vx::DataType data_type;
  std::vector<uint32_t> shape;
  tim::vx::TensorAttribute attr;
};

class VsiNpuModule : public ModuleNode {
 public:
  VsiNpuModule(const std::shared_ptr<char>& nbg_buffer,
               uint32_t nbg_size,
               const std::vector<tim::vx::TensorSpec>& inputs_spec,
               const std::vector<tim::vx::TensorSpec>& outputs_spec)
      : compiled_nbg_(nbg_buffer), nbg_size_(nbg_size), inputs_(inputs_spec), outputs_(outputs_spec){};

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  const char* type_key() const override { return "vsi_npu"; }

  /*!
   * \brief Save a compiled network to a binary stream, which can then be
   * serialized to disk.
   * \param stream The stream to save the binary.
   * \note See EthosnModule::LoadFromBinary for the serialization format.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

  /*!
   * \brief Load a compiled network from stream.
   * \param strm The binary stream to load.
   * \return The created Ethos-N module.
   * \note The serialization format is:
   *
   *       size_t : number of functions
   *       [
   *         std::string : name of function (symbol)
   *         std::string : serialized command stream
   *         size_t      : number of inputs
   *         std::vector : order of inputs
   *         size_t      : number of outputs
   *         std::vector : order of outputs
   *       ] * number of functions
   */
  static Module LoadFromBinary(void* strm);

 private:
  void SerializeTensorSpec(tim::vx::TensorSpec& t_spec, std::ostream& out);
  static tim::vx::TensorSpec DeSerializeTensorSpec(std::istream& in);
  // todo: TODO we need handle multiply nbg in real life
  std::shared_ptr<char> compiled_nbg_;
  uint32_t nbg_size_;
  std::vector<tim::vx::TensorSpec> inputs_;
  std::vector<tim::vx::TensorSpec> outputs_;

  std::shared_ptr<tim::vx::Context> vx_context_;
  std::shared_ptr<tim::vx::Graph> vx_graph_;
};
}  // namespace vsi_npu
}  // namespace runtime
}  // namespace tvm

#endif