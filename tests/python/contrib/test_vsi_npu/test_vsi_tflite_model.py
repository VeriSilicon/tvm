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
import sys
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import tvm
from tvm import relay, transform
from tvm import rpc
from tvm.contrib import graph_runtime
from tvm import te
from tvm.contrib import graph_runtime as runtime
from tvm.contrib import util
from tvm.relay.op.contrib import vsi_npu
from tvm.contrib.download import download_testdata

from tflite_deeplab import *
import tflite

RPC_HOST = os.environ["RPC_HOST"]
RPC_PORT = int(os.environ["RPC_PORT"])
CROSS_CC = os.environ["CROSS_CC"]
ROOTFS = os.environ["ROOTFS"]
lib_path = os.environ["MOD_SO_NAME"]

def get_img_data(image_path,shape, is_quant):
    resized_image = Image.open(image_path).resize(shape)

    DTYPE = "uint8" if is_quant else "float32"

    image_data = np.asarray(resized_image).astype(DTYPE)

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    if not is_quant:
        # Preprocess image as described here:
        # https://hub.fastgit.org/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243

        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1

    return image_data

def get_ref_result(shape, model_path,image_data,input_tensor_name,DTYPE):
    inputs = input_tensor_name
    tflite_model_buf = open(model_path, "rb").read()
    model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    mod, params = relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: DTYPE}
    )
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3,
                                   disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target, params=params)

    ctx = tvm.cpu()
    cpu_mod = graph_runtime.GraphModule(lib["default"](ctx))
    cpu_mod.set_input(inputs, tvm.nd.array(image_data))

    if True:
        print("Evaluate graph runtime inference cost on CPU")
        ftimer = cpu_mod.module.time_evaluator("run", ctx, number=1, repeat=1)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("CPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))

    cpu_mod.run()
    ref_out = cpu_mod.get_output(0).asnumpy()
    return ref_out

def compile_tflite_model(shape,model_path,input_tensor_name,DTYPE):
    inputs = input_tensor_name
    tflite_model_buf = open(model_path, "rb").read()
    model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: DTYPE}
    )

    kwargs = {}
    kwargs["cc"] = CROSS_CC
    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    kwargs["options"] = ["-L"+ROOTFS+"/lib ", "-L" + ROOTFS+"/usr/lib ",
                         "-L" + ROOTFS+"/usr/lib/aarch64-poky-linux/9.2.0 ", "--sysroot=" + ROOTFS]
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        #print(mod.astext(show_meta_data=False))
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)


def run_on_server(shape,input_tensor_name):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    lib = remote.load_module(lib_path)
    ctx = remote.cpu()
    rt_mod = graph_runtime.GraphModule(lib["default"](ctx))

    inputs = {
        #input_tensor_name: tvm.nd.array(np.arange(shape[0]*shape[1]*shape[2]*shape[3]).reshape(shape).astype("float32")),
        input_tensor_name: tvm.nd.array(np.ones(shape, "float32")),
    }
    rt_mod.set_input(**inputs)
    rt_mod.run()
    print(inputs)
    #rt_out = rt_mod.get_output(0).asnumpy()
    print(rt_mod.get_output(0))

shape = (1, 224, 224, 3)
input_tensor_name = "input"

model_name = "./model/mobilenet_v1_1.0_224.tflite"
DTYPE = "float"
wait=input("1. press any key and continue...")
compile_tflite_model(shape, model_name,input_tensor_name, DTYPE)

run_on_server(shape,input_tensor_name)
