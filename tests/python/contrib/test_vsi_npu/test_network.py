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

from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result
import sys

SUPPORTED_NETWORKS = {
   "mlp": relay.testing.mlp.get_workload,
   "resnet": relay.testing.resnet.get_workload,
   "mobilenet": relay.testing.mobilenet.get_workload,
   "vgg": relay.testing.vgg.get_workload,
   "inception_v3": relay.testing.inception_v3.get_workload,
   "densenet": relay.testing.densenet.get_workload,
   "squeezenet": relay.testing.squeezenet.get_workload
}

# get networks to run from cmdline. run all supported networks if no.
args = sys.argv[1:]
models_to_run = {}
for m in args:
    if m not in SUPPORTED_NETWORKS.keys():
        print("Supported networks: {}".format(list(SUPPORTED_NETWORKS.keys())))
        sys.exit(1)
    else:
        models_to_run[m] = SUPPORTED_NETWORKS[m]

if len(models_to_run) == 0:
    models_to_run = SUPPORTED_NETWORKS

for nn, get_workload in models_to_run.items():
    batch_size = 1
    num_class = 1000
    image_shape = (1, 224, 224)

    if nn == "mlp":
        num_class = 10
        image_shape = (1, 28, 28)

    if nn == "inception_v3":
        image_shape = (3, 299, 299)

    if nn == "squeezenet":
        image_shape = (3, 244, 244)

    if nn == "densenet":
        batch_size = 4
        image_shape = (3, 244, 244)

    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_class)
    dtype = "float32"

    if nn == "densenet":
        mod, params = get_workload(batch_size=batch_size, classes=num_class,
                                   image_shape=image_shape)
    else:
        mod, params = get_workload(batch_size=batch_size,
                                   num_classes=num_class,
                                   image_shape=image_shape)

    print("\nTesting {0: <50}".format(nn.upper()))
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)
