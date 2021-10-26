# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import logging
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor as graph_runtime
from tvm.relay.op.contrib import vsi_npu
from tvm import rpc
from tvm.contrib import utils as util
from tvm.testing import assert_allclose

logging.basicConfig(level=logging.DEBUG)

RPC_HOST = os.environ["RPC_HOST"]
RPC_PORT = int(os.environ["RPC_PORT"])
CROSS_CC = os.environ["CROSS_CC"]
ROOTFS = os.environ["ROOTFS"]
logging.info("Connect to {}:{} ...".format(RPC_HOST, RPC_PORT))
remote = rpc.connect(RPC_HOST, RPC_PORT, session_timeout=6000)


def make_module(func, params):
    func = relay.Function(relay.analysis.free_vars(func), func)
    if params:
        relay.build_module.bind_params_by_name(func, params)
    return tvm.IRModule.from_expr(func)

def get_vsi_model(mod, params):
    tmp_path = util.tempdir()
    lib_name = "model.so"
    lib_path = tmp_path.relpath(lib_name)

    kwargs = {}
    kwargs["cc"] = CROSS_CC
    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    kwargs["options"] = ["-L"+ROOTFS+"/lib " , "-L" +ROOTFS+"/usr/lib " , "-L" + ROOTFS+"/usr/lib/aarch64-poky-linux/9.2.0 ", "--sysroot=" +ROOTFS]
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        lib  = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    logging.info(lib_path)
    logging.info(lib_name)
    remote.upload(lib_path)
    lib = remote.load_module(lib_name)
    ctx = remote.cpu()

    rt_mod = graph_runtime.GraphModule(lib["default"](ctx))
    return rt_mod, ctx

def get_vsi_result(data, mod, params, out_shape, dtype):
    rt_mod, ctx = get_vsi_model(mod, params)
    #rt_mod.set_input("data", data)
    rt_mod.set_input(**data)
    rt_mod.run()
    rt_out = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
    rt_mod.get_output(0, rt_out)
    logging.info(data)
    return rt_out

def benchmark_vsi(mod, params, repeat=50):
    rt_mod, ctx = get_vsi_model(mod, params)

    logging.info("Evaluate graph runtime inference cost on VSI NPU")
    ftimer = rt_mod.module.time_evaluator("run", ctx, number=1, repeat=repeat)
    # Measure in millisecond.
    prof_res = np.array(ftimer().results) * 1000
    logging.info("VSI NPU runtime inference time (std dev): {} ms ({} ms)".format(round(np.mean(prof_res), 2), round(np.std(prof_res), 2)))

    return np.mean(prof_res)

def get_ref_result(data, mod, params, out_shape, dtype):
    # with tvm.transform.PassContext():
    #     with tvm.target.Target("llvm"):
    #         f = relay.build_module.bind_params_by_name(mod["main"], params)
    #         mod = tvm.IRModule()
    #         mod["main"] = f
    #         mod = relay.transform.AnnotateTarget("vsi_npu")(mod)
    #         mod = relay.transform.MergeCompilerRegions()(mod)
    #         mod = relay.transform.PartitionGraph()(mod)
    #         lib = relay.build(mod, params=params)
    #         lib.export_library(lib_path, fcompile=False, **kwargs)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib  = relay.build(mod, target, params=params)
    logging.info(mod.astext())

    cpu_mod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    cpu_mod.set_input(**data)
    cpu_mod.run()
    cpu_out = cpu_mod.get_output(0, tvm.nd.empty(out_shape, dtype))
    return cpu_out

def verify_vsi_result(inputs, model, params, data_shape, out_shape, dtype="float32"):
    # wait=input("1. press any key and continue...")
    mod = make_module(model, params)
    ref_out = get_ref_result(inputs, mod, params, out_shape, dtype)
    vsi_out = get_vsi_result(inputs, mod, params, out_shape, dtype)
    if dtype == "uint8":
        atol = 1
        rtol = 1.0 / 255
    else:
        atol = 1e-3
        rtol = 1e-3
    assert_allclose(vsi_out.numpy(), ref_out.numpy(), rtol=rtol, atol=atol)
