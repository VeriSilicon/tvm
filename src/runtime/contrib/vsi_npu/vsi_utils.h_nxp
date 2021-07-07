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

/*!
 * \file src/runtime/contrib/vsi_npu/vsi_utils.h
 * \brief VSI NPU util functions.
 */

#ifndef TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_UTILS_H_
#define TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_UTILS_H_

//Convert axis to VSI NPU axis
static inline int32_t
ConvertAxis(int32_t axis_in, uint64_t dim_num) {
    if (axis_in < 0) {
        return -axis_in - 1;
    } else {
        return dim_num - axis_in - 1;
    }
}

template <typename T>
static inline std::vector<T>
GetVectorFromDLTensor(const DLTensor* tensor) {
  CHECK(tensor) << "Cannot convert a nullptr";
  int len = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    len *= tensor->shape[i];
  }
  T* data = static_cast<T*>(tensor->data);
  return std::vector<T>(data, data + len);
}

#endif  // TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_UTILS_H_
