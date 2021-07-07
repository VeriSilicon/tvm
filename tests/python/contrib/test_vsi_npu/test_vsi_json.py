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
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay import transform
import numpy as np
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib import vsi_npu
from tvm import rpc
from tvm.contrib import util

HOST = "10.193.20.6"
PORT = 9090
remote = rpc.connect(HOST, PORT)

batch_size = 1
num_class = 10
image_shape = (1, 28, 28)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype="float32"

mod, params = relay.testing.mlp.get_workload(
    batch_size=batch_size
)

tmp_path = util.tempdir()
lib_name = "mlp.so"
lib_path = tmp_path.relpath(lib_name)

kwargs = {}
kwargs["cc"] = "aarch64-linux-gnu-g++"
target = "llvm  -mtriple=aarch64-linux-gnu"
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    mod = vsi_npu.partition_for_vsi_npu(mod, params)
    json, lib, param  = relay.build(mod, target, params=params)
    lib.export_library(lib_path, fcompile=False, **kwargs)

print(mod.astext(show_meta_data=False))

remote.upload(lib_path)
lib = remote.load_module(lib_name)
ctx = remote.cpu()

rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)
data = np.random.uniform(size=data_shape).astype(dtype)
rt_mod.set_input("data", data)
rt_mod.run()
#out = rt_mod.get_output(0, tvm.nd.empty(out_shape, dtype))
#tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)
