//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#ifndef DEEP_ENGINE_EXECUTOR_INCLUDE_OPERATORS_INNER_PRODUCT_HPP_
#define DEEP_ENGINE_EXECUTOR_INCLUDE_OPERATORS_INNER_PRODUCT_HPP_
#include <vector>
#include <string>
#include <unordered_map>
#include "oneapi/dnnl/dnnl.hpp"
#include "../operator.hpp"
#include "../common.hpp"

namespace executor {

using dnnl::memory;
using dnnl::engine;
using dnnl::prop_kind;
using dnnl::algorithm;

// \brief InnerProduct or Batchmatmul operators
class InnerProductOperator : public Operator {
 public:
  explicit InnerProductOperator(const OperatorConfig& conf);
  virtual ~InnerProductOperator();

 public:
  // void ParseOperatorConfig();
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);

  // Converting string variables from operators attrs to boolean, or int/float
 protected:
  // The input tensors x and y are [..., r_x, c_x] and [..., r_y, c_y].
  // The output tensor is [..., r_o, c_o], where:
  // r_o = c_x if adj_x else r_x
  // c_o = r_y if adj_y else c_y
  // Matrix can optionally be adjointed (to adjoint a matrix means to transpose and conjugate it).
  // So "adj_" decide the highest two dimensions, and is the built-in operation of InnerProduct OP.
  // While "perm" decide all dimensions, and is the external Trans OP. Both are transpose.
  bool is_asymm_;
  bool weight_cached_;
  bool has_bias_;
  bool format_any_;
  bool append_sum_;
  bool binary_add_;
  bool gelu_erf_;
  bool gelu_tanh_;
  bool gelu_split_;
  bool tanh_;
  bool append_eltwise_;
  float output_scale_ = 1.f;
  string output_dtype_ = "fp32";
  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  vector<int64_t> dst_perm_;

  dnnl::primitive_attr attr_;
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream eng_stream_ = dnnl::stream(eng_);
  dnnl::inner_product_forward::primitive_desc inner_product_pd_;
  dnnl::inner_product_forward inner_product_p_;
  unordered_map<int, memory> memory_args_;

  dnnl::engine gelu_eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream gelu_eng_stream_ = dnnl::stream(gelu_eng_);
  dnnl::eltwise_forward::primitive_desc gelu_pd_;
  dnnl::eltwise_forward gelu_p_;
  unordered_map<int, memory> gelu_memory_args_;

  memory::desc src1_md_;
  memory::desc any_src1_md_;
  memory::desc bias_md_;
  memory::desc any_bias_md_;
  memory src0_m_;
  memory src1_m_;
  memory bias_m_;
  memory dst_m_;
  memory gelu_m_;
  memory binary_m_;

  Tensor* src0_ = nullptr;
  Tensor* src1_ = nullptr;
  Tensor* bias_ = nullptr;
  Tensor* post_ = nullptr;
  Tensor* dst_ = nullptr;

  Tensor* src0_min_ = nullptr;
  Tensor* src0_max_ = nullptr;

  Tensor* src1_min_ = nullptr;
  Tensor* src1_max_ = nullptr;

  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
};
}  // namespace executor
#endif  // DEEP_ENGINE_EXECUTOR_INCLUDE_OPERATORS_INNER_PRODUCT_HPP_
