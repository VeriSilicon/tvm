import tvm.ir
from tvm.relay import transform
from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table
from ... import qnn as _qnn
from . import vsi_npu_ffi_api as support_api
from tvm.relay.build_module import bind_params_by_name

# def _register_external_op_helper(op_name, supported=True):
#     """The helper function to indicate that a given operator can be supported
#     by DNNL.

#     Paramters
#     ---------
#     op_name : Str
#         The name of operator that will be registered.

#     Returns
#     -------
#     f : callable
#         A function that returns if the operator is supported by DNNL.
#     """
#     print("Sven-TVM: python vsi-npu")
#     @tvm.ir.register_op_attr(op_name, "target.vsi_npu")
#     def _func_wrapper(attrs, args):
#         return supported

#     return _func_wrapper

# _register_external_op_helper("add")

@register_pattern_table("vsi_npu")
def vsi_npu_pattern_table():
    # conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    # conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    # dnnl_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]

    print("vsi_npu.py --> pattern_table()")
    def qnn_conv_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("qnn.conv2d")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant())))
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_avg_pool2d_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def qnn_softmax_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.softmax")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_sigmoid_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("sigmoid")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_clip_pattern():
        pattern = is_op("clip")(wildcard())
        pattern = pattern.optional(lambda x: (is_op("qnn.requantize")(
            x, is_constant(), is_constant(), is_constant(), is_constant()
        )))
        return pattern

    def qnn_leaky_relu_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.leaky_relu")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_tanh_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("tanh")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    # def qnn_fullconnected_pattern():
    #     pattern = is_op("reshape")(wildcard())
    #     pattern = is_op("qnn.dense")(
    #         pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    #     )
    #     pattern = pattern.optional(lambda x: (is_op("nn.bias_add")(
    #         x, is_constant()) | is_op("add")(x, is_constant())))
    #     pattern = is_op("qnn.requantize")(
    #         pattern, is_constant(), is_constant(), is_constant(), is_constant()
    #     )

    def qnn_dense_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant())))
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_mean_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("mean")(pattern)
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant())
        return pattern

    def qnn_deconv_pattern():
        pattern = is_op("qnn.conv2d_transpose")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )    
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    vsi_npu_patterns = [
            ("vsi_npu.qnn_deconv", qnn_deconv_pattern()),
            ("vsi_npu.qnn_dense",qnn_dense_pattern()),
            ("vsi_npu.qnn_conv2d", qnn_conv_pattern()),
            ("vsi_npu.qnn_avgpool2d", qnn_avg_pool2d_pattern()),
            ("vsi_npu.qnn_softmax", qnn_softmax_pattern()),
            ("vsi_npu.qnn_sigmoid", qnn_sigmoid_pattern()),
            ("vsi_npu.qnn_clip", qnn_clip_pattern()),
            ("vsi_npu.qnn_mean", qnn_mean_pattern()),
            ("vsi_npu.qnn_leaky_relu", qnn_leaky_relu_pattern()),
            ("vsi_npu.qnn_tanh", qnn_tanh_pattern()),
            ]
    return vsi_npu_patterns

def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.vsi_npu")
    def _func_wrapper(args):
        print("vsi_npu.py --> {}".format(op_name))
        return supported

    return _func_wrapper

_register_external_op_helper("qnn.add")
_register_external_op_helper("qnn.subtract")
_register_external_op_helper("qnn.mul")
_register_external_op_helper("maximum")
_register_external_op_helper("minimum")
_register_external_op_helper("logical_and")
_register_external_op_helper("logical_or")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("mean")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("reshape")
_register_external_op_helper("squeeze")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("clip")
_register_external_op_helper("qnn.quantize")
_register_external_op_helper("qnn.dequantize")
_register_external_op_helper("qnn.requantize")
_register_external_op_helper("qnn.concatenate")
_register_external_op_helper("image.resize")
_register_external_op_helper("argmax")
_register_external_op_helper("argmin")
_register_external_op_helper("transpose")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.depth_to_space")
_register_external_op_helper("nn.pad")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("nn.conv2d_transpose")

# @tvm.ir.register_op_attr("layout_transform", "target.vsi_npu")
# def layout_transform(attrs, args):
#     """Check if the external VSI codegen should be used."""
#     if attrs.src_layout == "NHWC" and attrs.dst_layout == "NCHW" and args[0].checked_type.dtype != "uint8":
#         return True
#     return True

def partition_for_vsi_npu(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to VSI NPU.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    # desired_layouts = {'nn.conv2d' : ['WHCN', 'WHIO'],
    #                     'nn.avg_pool2d' : ['WHCN']}
    #desired_layouts = {'nn.conv2d' : ['NCHW', 'OIHW']}
    desired_layouts = {}
    seq = tvm.transform.Sequential(
        [
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(desired_layouts),
            transform.FoldConstant(),
            transform.MergeComposite(vsi_npu_pattern_table()),
            transform.AnnotateTarget("vsi_npu"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)
