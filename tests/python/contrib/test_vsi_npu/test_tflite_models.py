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

RPC_HOST = "10.10.166.166"
RPC_PORT = 9092
MEASURE_PERF = False
VERBOSE = False
SUPPORTED_MODELS = {}  # name to TFModel mapping


class TFModel:
    def __init__(self, name, where, is_quant, formats='tgz', suffix=''):

        # expect the name looks like "mobilenet_v1_0.25_128"
        fields = name.split("_")

        try:
            size = int(fields[-1])
        except Exception:
            size = 224

        if is_quant:
            name = "{}_quant".format(name)

        self.name = name
        self.formats = formats
        self.url = "{}/{}{}.{}".format(where, name, suffix, formats)

        self.is_quant = is_quant
        self.input_size = size
        self.inputs = 'input'

class TFUnitmodel:
    def __init__(self, name, intput_size, where, is_quant):

        if intput_size:
            size = intput_size
        else:
            size = 224

        if is_quant:
            name = "{}_quant".format(name)
        self.name = name
        self.location = where

        self.is_quant = is_quant
        self.intput_size = intput_size
        self.inputs = 'input'

def add_supported_model(name, where, is_quant=False, formats='tgz',
                        suffix=''):
    m = TFModel(name, where, is_quant, formats, suffix)
    SUPPORTED_MODELS[m.name] = m

    return m

def add_unit_supported_model(name,intput_size,where,is_quant=False):
    m = TFUnitmodel(name,intput_size,where,is_quant)
    SUPPORTED_MODELS[m.name] = m

    return m
def init_unit_test_supported_models(operation_floder,data_type,input_size):
    QUANT = True
    location = operation_floder + "/export_tflite/" + data_type
    model_name = os.listdir(location)
    m = add_unit_supported_model(location + '/' + model_name[0])
    m.input_size = input_size
    

def init_supported_models():
    QUANT = True
    where = "http://download.tensorflow.org/models/object_detection"
    m = add_supported_model("ssd_mobilenet_v3", where, formats='tar.gz', suffix="_small_coco_2020_01_14")
    m.input_size = 320
    m.is_quant = True
    m.inputs = 'normalized_input_image_tensor'

    where = "http://10.192.208.75/images/deepview/models/float/mobilenet_ssd_v1"
    m = add_supported_model("mobilenet_ssd_v1_trimmed_converted", where, formats='tflite')
    m.input_size = 300
    m.inputs = 'Preprocessor/sub'

    where = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02"
    add_supported_model("mobilenet_v1_0.25_128", where, QUANT)
    add_supported_model("mobilenet_v1_0.25_224", where, QUANT)
    add_supported_model("mobilenet_v1_0.5_128", where, QUANT)
    add_supported_model("mobilenet_v1_0.5_224", where, QUANT)
    add_supported_model("mobilenet_v1_0.75_128", where, QUANT)
    add_supported_model("mobilenet_v1_0.75_224", where, QUANT)
    add_supported_model("mobilenet_v1_1.0_128", where, QUANT)
    add_supported_model("mobilenet_v1_1.0_224", where, QUANT)

    add_supported_model("mobilenet_v1_0.25_128", where)
    add_supported_model("mobilenet_v1_0.25_224", where)
    add_supported_model("mobilenet_v1_0.5_128", where)
    add_supported_model("mobilenet_v1_0.5_224", where)
    add_supported_model("mobilenet_v1_0.75_128", where)
    add_supported_model("mobilenet_v1_0.75_224", where)
    add_supported_model("mobilenet_v1_1.0_128", where)
    add_supported_model("mobilenet_v1_1.0_224", where)

    where = "https://storage.googleapis.com/"
    where += "download.tensorflow.org/models/tflite_11_05_08"
    add_supported_model("mobilenet_v2_1.0_224", where, QUANT)
    add_supported_model("mobilenet_v2_1.0_224", where)
    # the input size for inception_v3 model from tflite model zoo is 299
    # https://discuss.tvm.apache.org/t/possible-bug-relay-internal-invariant-was-violdated/2105/8
    m = add_supported_model("inception_v3", where, QUANT)
    m.input_size = 299

    where = "https://storage.googleapis.com/download.tensorflow.org/models"
    add_supported_model("inception_v1_224", where, QUANT, suffix="_20181026")
    add_supported_model("inception_v2_224", where, QUANT, suffix="_20181026")
    add_supported_model("inception_v4_299", where, QUANT, suffix="_20181026")

    where = "https://storage.googleapis.com/download.tensorflow.org/"
    where += "models/tflite/model_zoo/upload_20180427"
    m = add_supported_model("inception_v3", where, suffix="_2018_04_27")
    m.input_size = 299
    m = add_supported_model("inception_v4", where, suffix="_2018_04_27")
    m.input_size = 299

    where = "https://storage.googleapis.com/download.tensorflow.org/"
    where += "models/tflite/gpu"
    m = add_supported_model("deeplabv3_257_mv_gpu", where, formats='tflite')
    m.input_size = 257
    m.inputs = "sub_7"

    where = "https://hub.fastgit.org/google-coral/edgetpu/raw/master/test_data"
    m = add_supported_model("deeplabv3_mnv2_pascal", where, QUANT,
                            formats='tflite')
    m.input_size = 513
    m.inputs = "MobilenetV2/MobilenetV2/input"

    where = "http://download.tensorflow.org/models/object_detection"
    m = add_supported_model("ssdlite_mobiledet_cpu_320x320_coco", where,
                            formats='tar.gz', suffix="_2020_05_19")
    m.input_size = 320
    m.inputs = 'normalized_input_image_tensor'

    return SUPPORTED_MODELS


def extract(path, model_name):
    import tarfile
    import zipfile

    dir_path = os.path.dirname(path)
    tmp_dir = os.path.join(dir_path, model_name)
    os.makedirs(tmp_dir, exist_ok=True)
    if path.endswith("tgz") or path.endswith("gz") or path.endswith("tar.gz"):
        tar = tarfile.open(path)
        tar.extractall(path=tmp_dir)
        tar.close()
    elif path.endswith("zip"):
        zf = zipfile.ZipFile(path, 'r')
        for f in zf.namelist():
            zf.extract(f, tmp_dir)
    else:
        raise RuntimeError("Could not decompress the file: " + path)

    for dir_path, subpaths, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith("tflite"):
                return os.path.join(dir_path, f)

    raise RuntimeError("Could not find tflite model file.")

def get_tflite_model(model_name):

    m = SUPPORTED_MODELS[model_name]

    # Download model tar file and extract it to get tflite
    model_path = download_testdata(m.url,
                                   model_name + "." + m.formats,
                                   module=["tf", "official"])

    model_dir = os.path.dirname(model_path)
    if m.formats in ['tgz', 'zip', 'tar.gz']:
        model_name = extract(model_path, model_name)
    else:
        model_name = model_name + ".tflite"
    # Now we can open tflite model
    tflite_model_file = os.path.join(model_dir, model_name)
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    return model


def get_img_data(shape, is_quant):
    image_url =\
        "https://hub.fastgit.org/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"

    # image_path = download_testdata(image_url, "cat.png", module="data")
    image_path = "/home/sven/.tvm_test_data/data/cat.png"
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


def compile_tflite_model(shape, model_name):
    m = SUPPORTED_MODELS[model_name]

    DTYPE = "uint8" if m.is_quant else "float32"

    model = get_tflite_model(model_name)
    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={m.inputs: shape}, dtype_dict={m.inputs: DTYPE}
    )
    lib_path = "./model.so"

    kwargs = {}
    ROOTFS = "/home/software/Linux/freescale/L5447_64/sysroots/aarch64-poky-linux"
    kwargs["options"] = ["--sysroot=" + ROOTFS]
    kwargs["cc"] = "aarch64-poky-linux-gcc"
    target = "llvm  -mtriple=aarch64-linux-gnu"
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        if VERBOSE:
            print(mod.astext(show_meta_data=False))
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    return lib_path


def inference_remotely(tfmodel, lib_path, image_data):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    # Create a runtime executor module
    module = graph_runtime.GraphModule(lib["default"](ctx))
    # Feed input data
    module.set_input(tfmodel.inputs, tvm.nd.array(image_data))

    if MEASURE_PERF:
        print("Evaluate graph runtime inference cost on VSI NPU")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))
    module.run()
    tvm_output = module.get_output(0).asnumpy()

    return tvm_output


def get_ref_result(shape, model_name, image_data):

    m = SUPPORTED_MODELS[model_name]
    inputs = m.inputs
    DTYPE = "uint8" if m.is_quant else "float32"

    model = get_tflite_model(model_name)
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

    if MEASURE_PERF:
        print("Evaluate graph runtime inference cost on CPU")
        ftimer = cpu_mod.module.time_evaluator("run", ctx, number=1, repeat=1)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("CPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))

    cpu_mod.run()
    ref_out = cpu_mod.get_output(0).asnumpy()
    return ref_out


def verify_tvm_result(ref_output, shape, model_name, image_data):

    m = SUPPORTED_MODELS[model_name]
    lib_path = compile_tflite_model(shape, model_name)
    tvm_output = inference_remotely(m, lib_path, image_data)

    if m.name.startswith('deeplabv3') and m.is_quant:
        ref_output = ref_output.reshape(shape[1:3])
        tvm_output = tvm_output.reshape(shape[1:3])

        pix_acc = pixel_accuracy(ref_output, tvm_output)
        print("pixel accuracy:", pix_acc)
        m_acc = mean_accuracy(ref_output, tvm_output)
        print("mean accuracy:", m_acc)
        IoU = mean_IU(ref_output, tvm_output)
        print("mean IU:", IoU)
        freq_weighted_IU = frequency_weighted_IU(ref_output, tvm_output)
        print("frequency weighted IU:", freq_weighted_IU)

    elif 'deeplabv3' in m.name or "ssd" in m.name:
        # compare deeplabv3 float32 output
        np.testing.assert_allclose(ref_output, tvm_output,
                                   rtol=1e-4, atol=1e-4, verbose=True)
    else:  # label index comparison
        ref_idx = np.argmax(np.squeeze(ref_output))
        out_idx = np.argmax(np.squeeze(tvm_output))

        print(f'Expect predict id: {ref_idx}, got {out_idx}')
        assert ref_idx == out_idx


def print_help():
    print("Usage: \n",
          '  -r[--rpc]    IP:Port\n',
          '  -m[--models] "model_list ...", all models by default\n',
          '  [--perf] benchmark performance\n',
          '  [--verbose] print more logs\n')
    print("Supported models:")
    print(list(SUPPORTED_MODELS.keys()))
    sys.exit(1)


def parse_command_args():
    import getopt
    import re

    global RPC_HOST, RPC_PORT, MEASURE_PERF, VERBOSE
    models_input = []
    try:
        opts, args = getopt.getopt(
                         sys.argv[1:], "hr:m:v",
                         ["help", "rpc=", "models=", "perf", "verbose"])
    except getopt.GetoptError:   # not supported option
        print_help()

    if args:  # not support argument without '-'
        print_help()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
        elif opt in ("-r", "--rpc"):
            rpc_ = arg.split(":")
            if (len(rpc_) != 2):
                raise Exception("invalid rcp address: %s" % arg)
            RPC_HOST = rpc_[0].strip()
            RPC_PORT = int(rpc_[1].strip())
        elif opt in ("-m", "--models"):
            models_input = ' '.join(re.split(' |,|;', arg.strip())).split()
        elif opt in ("--perf"):
            MEASURE_PERF = True
        elif opt in ("-v", "--verbose"):
            VERBOSE = True

    return models_input


init_supported_models()
args = parse_command_args()
models_to_run = {}

for m in args:
    if m not in SUPPORTED_MODELS.keys():
        print("{} is not supported!".format(m))
        print("Supported models: {}".format(list(SUPPORTED_MODELS.keys())))
        sys.exit(1)
    else:
        models_to_run[m] = SUPPORTED_MODELS[m]


if len(models_to_run) == 0:
    models_to_run = SUPPORTED_MODELS

print(f"\nTesting {len(models_to_run)} model(s): {list(models_to_run.keys())}")

pass_cases = 0
failed_list = []
for model_name, m in models_to_run.items():
    print("\nTesting {0: <50}".format(model_name.upper()))

    is_quant = m.is_quant
    input_size = m.input_size

    shape = (1, input_size, input_size, 3)

    image_data = get_img_data(shape[1:3], is_quant)
    ref_output = get_ref_result(shape, model_name, image_data)

    try:
        verify_tvm_result(ref_output, shape, model_name, image_data)
    except Exception as err:
        print("Exception", err)
        print(model_name, ": FAIL")
        failed_list.append(model_name)
    else:
        print(model_name, ": PASS")
        pass_cases += 1

print("\n\nTest", len(models_to_run), "cases: ", pass_cases, "Passed")
if len(failed_list) > 0:
    print("Failed list is:", failed_list)
