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
import math
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pytest
import logging

import tvm
from tvm import relay, transform
from tvm import rpc
from tvm import te
from tvm.contrib import utils as util
from tvm.relay.op.contrib import vsi_npu
from tvm.contrib.download import download_testdata
from tvm.testing import assert_allclose

# TODO(Sven) : this is workaround for new version of TVM
from tvm.contrib import graph_executor as graph_runtime
from tvm.contrib import graph_executor as runtime

#from tflite_deeplab import *
import tflite

# import os
# input("pid: " + str(os.getpid()) +", press enter after attached")

logging.basicConfig(level=logging.INFO)

np.set_printoptions(threshold=np.inf)
RPC_HOST = os.environ["RPC_HOST"]
RPC_PORT = int(os.environ["RPC_PORT"])
CROSS_CC = os.environ["CROSS_CC"]
ROOTFS = os.environ["ROOTFS"]
lib_name = os.environ["MOD_NAME"]
lib_path = os.environ["MOD_PATH"]
model_path = os.environ["TFLITE_MODEL_PATH"]

remote = rpc.connect(RPC_HOST, RPC_PORT)

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

    # if True:
    #     logging.info("Evaluate graph runtime inference cost on CPU")
    #     ftimer = cpu_mod.module.time_evaluator("run", ctx, number=1, repeat=1)
    #     # Measure in millisecond.
    #     prof_res = np.array(ftimer().results) * 1000
    #     logging.info("CPU runtime inference time (std dev): {} ms ({} ms)".format
    #           (round(np.mean(prof_res), 2), round(np.std(prof_res), 2)))

    cpu_mod.run()
    rf_output = []
    output_num = cpu_mod.get_num_outputs()
    for i in range(output_num):
        rf_output.append(cpu_mod.get_output(i).asnumpy())
    return rf_output

def compile_tflite_model(shape,model_path,input_data,input_tensor_name,DTYPE):
    inputs = input_tensor_name
    tflite_model_buf = open(model_path, "rb").read()
    model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: DTYPE}
    )
    logging.info(mod.astext())
    tmp_path = util.tempdir()
    lib_name = "model.so"
    lib_path = tmp_path.relpath(lib_name)

    kwargs = {}
    kwargs["cc"] = CROSS_CC
    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    kwargs["options"] = ["-L"+ROOTFS+"/lib ", "-L" + ROOTFS+"/usr/lib ",
                         "-L" + ROOTFS+"/usr/lib/aarch64-poky-linux/9.2.0 ", "--sysroot=" + ROOTFS]
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    # remote = rpc.connect(RPC_HOST, RPC_PORT)
    # remote.upload(lib_path + lib_name)
    # lib = remote.load_module(lib_name)

    remote.upload(lib_path)
    lib = remote.load_module(lib_name)
    ctx = remote.cpu()

    rt_mod = graph_runtime.GraphModule(lib["default"](ctx))

    # ctx = remote.cpu()
    # rt_mod = graph_runtime.GraphModule(lib["default"](ctx))
    rt_mod.set_input(**input_data)
    rt_mod.run()
    rf_output = []
    output_num = rt_mod.get_num_outputs()
    for i in range(output_num):
        rf_output.append(rt_mod.get_output(i).asnumpy())
    return rf_output

def print_top5(input):
    k = 5
    n = input.flatten()
    n_arg0 = np.argpartition(n, -k)[-k:]
    n = n[n_arg0]
    n_arg1 = np.argsort(n)[::-1]
    n_arg0 = n_arg0[n_arg1]
    for i in range(k):
        logging.info("{} : {}".format(n_arg0[i], n[n_arg1[i]]))

def get_model(model_list, model_name):
    for model in model_list:
        if (model['name'] == model_name):
            return model

model_list = [
    {'name': 'mobilenet_v1_1.0_224_quant.tflite',
     'shape': (1, 224, 224, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'mobilenet_v2_quant.tflite',
     'shape': (1, 224, 224, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'mobilenet_v3_quant.tflite',
     'shape': (1, 512, 512, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'inception_v1_224_quant.tflite',
     'shape': (1, 224, 224, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'inception_v2_224_quant.tflite',
     'shape': (1, 224, 224, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'inception_v3_299_quant.tflite',
     'shape': (1, 299, 299, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'efficientnet-edgetpu-S_quant.tflite',
     'shape': (1, 224, 224, 3),
     'input_tensor_name': 'images',
     'dtype': "uint8"},
    {'name': 'deeplab_v3_plus_quant.tflite',
     'shape': (1, 513, 513, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'deeplabv3_mnv2_pascal_quant.tflite',
     'shape': (1, 513, 513, 3),
     'input_tensor_name': 'MobilenetV2/MobilenetV2/input',
     'dtype': "uint8"},
    {'name': 'unet.M865SW-632.tflite',
     'shape': (1, 120, 160, 1),
     'input_tensor_name': 'input_1',
     'dtype': "float32"},
    {'name': 'deeplab_v3_plus_quant.tflite',
     'shape': (1, 513, 513, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'yolov3-tiny_uint8_acuity.tflite',
     'shape': (1, 416, 416, 3),
     'input_tensor_name': 'input_0:out0',
     'dtype': "uint8"},
    {'name': 'yolov3_uint8_acuity.tflite',
     'shape': (1, 608, 608, 3),
     'input_tensor_name': 'input_0:out0',
     'dtype': "uint8"},
    {'name': 'unet_quant.tflite',
     'shape': (1, 384, 384, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'srgan_quant.tflite',
     'shape': (1, 128, 128, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'pynet_quant.tflite',
     'shape': (1, 96, 96, 3),
     'input_tensor_name': 'input',
     'dtype': "uint8"},
    {'name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_quant.M865SW-669.tflite',
     'shape': (1, 320, 320, 3),
     'input_tensor_name': 'serving_default_input:0',
     'dtype': "uint8"},
    {'name': 'operation_1_in_29_out_4.tflite',
     'shape': (1, 416, 416, 3),
     'input_tensor_name': 'input_0:out0',
     'dtype': "uint8"},
]

def process(model_name):
    model = get_model(model_list, model_name)
    logging.info(model)
    shape = model['shape']
    input_tensor_name = model['input_tensor_name']
    DTYPE = model['dtype']
    # wait=input("press any key and continue...")

    path = "./"
    img = Image.open(path + "space_shuttle_224x224.jpg")
    img = img.resize((shape[1], shape[2]))
    n1 = np.array(img)
    #n1 = n1[:, :, 0] # pick one channel
    #n1 = np.broadcast_to(n1, (4, 224, 224, 3)) # batch the image
    n1 = n1.reshape(shape)
    input_data = n1.astype(np.uint8)

    # input_data = np.ones(shape, DTYPE)

    vsi_input_data = {
        input_tensor_name: tvm.nd.array(input_data),
    }
    model_file = model_path + "/" + model_name
    print(model_file)
    ref_output = get_ref_result(shape, model_file, input_data, input_tensor_name, DTYPE)
    vsi_output = compile_tflite_model(shape, model_file, vsi_input_data, input_tensor_name, DTYPE)

    output_num = len(ref_output)

    if DTYPE == "uint8":
        tolerance = 5
    else:
        tolerance = 1e-3

    for i in range(output_num):
        logging.info("top5 of ref output %d:",i)
        print_top5(ref_output[i])
        logging.info("top5 of vsi output %d:",i)
        print_top5(vsi_output[i])

        diff = abs(vsi_output[i].astype("float32") - ref_output[i].astype("float32"))
        np.savetxt(path + model_name+ "_output_" + str(i) +"_ref_output.txt", ref_output[i].flatten(), fmt='%.3f')
        np.savetxt(path + model_name+ "_output_" + str(i) +"_vsi_output.txt", vsi_output[i].flatten(), fmt='%.3f')
        np.savetxt(path + model_name+ "_output_" + str(i) +"_diff.txt", diff.flatten(), fmt='%.3f')
        assert_allclose(vsi_output[i], ref_output[i], rtol=0, atol=tolerance)


def test_mobilenet_v1_224_quant():
    model = 'mobilenet_v1_1.0_224_quant.tflite'
    process(model)

def test_mobilenet_v2_quant():
    model = 'mobilenet_v2_quant.tflite'
    process(model)

def test_mobilenet_v3_quant():
    model = 'mobilenet_v3_quant.tflite'
    process(model)

def test_inception_v1_224_quant():
    model = 'inception_v1_224_quant.tflite'
    process(model)

def test_inception_v2_224_quant():
    model = 'inception_v2_224_quant.tflite'
    process(model)

def test_inception_v3_299_quant():
    model = 'inception_v3_299_quant.tflite'
    process(model)

def test_efficientnet_edgetpu_S_quant():
    model = 'efficientnet-edgetpu-S_quant.tflite'
    process(model)

def test_deeplab_v3_plus_quant():
    model = 'deeplab_v3_plus_quant.tflite'
    process(model)

def test_unet():
    model = 'unet.M865SW-632.tflite'
    process(model)

def test_deeplab_v3_plus_quant():
    model = 'deeplab_v3_plus_quant.tflite'
    process(model)

def test_srgan_quant():
    model = 'srgan_quant.tflite'
    process(model)

def test_pynet_quant():
    model = 'pynet_quant.tflite'
    process(model)

if __name__ == "__main__":
    pytest.main([__file__])