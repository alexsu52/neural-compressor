#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from collections import OrderedDict

import copy
import os
import logging
import tempfile
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from lpot.utils.utility import get_all_fp32_data
from lpot.utils.utility import get_tensor_histogram
from lpot.utils.utility import combine_histogram
from lpot.utils.utility import CaptureOutputToFile
from lpot.utils.utility import str2array
from lpot.utils.utility import Dequantize, DequantizeWeight
from lpot.conf.dotdict import deep_get
from lpot.experimental.common import Model
from .transform_graph.insert_logging import InsertLogging
from .transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from .transform_graph.bias_correction import BiasCorrection
from .util import iterator_sess_run
from .quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from .quantize_graph.quantize_graph_common import QuantizeGraphHelper
from .quantize_graph.quantize_graph_conv import FuseNodeStartWithConv2d

from .graph_rewriter.graph_util import GraphAnalyzer
from .graph_rewriter.generic.remove_training_nodes import RemoveTrainingNodesOptimizer
from .graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from .graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from .graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer

from .graph_rewriter.int8.freeze_value import FreezeValueTransformer
from .graph_rewriter.int8.freeze_value_without_calib import FreezeValueWithoutCalibTransformer
from .graph_rewriter.int8.freeze_fake_quant import FreezeFakeQuantOpOptimizer
from .graph_rewriter.int8.fuse_conv_requantize import FuseConvRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeDequantizeTransformer
from .graph_rewriter.int8.scale_propagation import ScaleProPagationTransformer
from .graph_rewriter.bf16.bf16_convert import BF16Convert
from .graph_rewriter.int8.post_quantized_op_cse import PostCseOptimizer
from .graph_rewriter.int8.meta_op_optimizer import MetaInfoChangingMemOpOptimizer
from .graph_rewriter.int8.rnn_convert import QuantizedRNNConverter
from .graph_rewriter.itex.itex_convert import GenerateITEXModel
from lpot.adaptor.tf_utils.graph_rewriter.generic.insert_print_node import InsertPrintMinMaxNode
from lpot.adaptor.tf_utils.graph_rewriter.graph_util import GraphRewriterHelper as Helper
from lpot.adaptor.tf_utils.graph_rewriter.int8.remove_fake_quant import RemoveFakeQuantOpOptimizer
from lpot.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper as helper


TF_SUPPORTED_MAX_VERSION = '2.6.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'

logger = logging.getLogger()
debug = bool(logger.level == logging.DEBUG)

class GraphConverter:
    def __init__(self,
                 model,
                 qt_config={},
                 recipes={},
                 int8_sequences={},
                 fp32_ops=[],
                 bf16_ops=[],
                 data_loader=None,
                 fake_quant=False,
                 itex_mode=False,
                 qat_model_parameters=None):
        """Convert graph.

        :param model: input tensorflow model.
        :param qt_config: quantization configs, including interation and op-wise quant config
        :param fp32_ops: fall back to fp32 dtype op list
        :param bf16_ops: fall back to bf16 dtype op list
        :param data_loader: for calibration phase used dataloader
        :param fake_quant: for quantization-aware training model conversion to default model
        """
        self.model = model
        #(TODO) does it right to make the internal model format as graph_def
        self.output_tensor_names = self.model.output_tensor_names
        self.input_tensor_names = self.model.input_tensor_names
        # quantize specific config
        self.calib_iteration = qt_config['calib_iteration'] if not fake_quant else 0
        self.op_wise_config = qt_config['op_wise_config']
        self.advance_config = deep_get(qt_config, 'advance')
        self.device = qt_config['device'] if 'device' in qt_config else 'cpu'
        self.int8_sequences = int8_sequences
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.recipes = recipes
        self.fake_quant = fake_quant
        self.itex_mode = itex_mode
        self.quantized_node_info = []
        self._calibration_data = []
        self._fp32_print_data = []
        self.data_loader = data_loader
        self._check_tf_version()
        self._check_args()
        self._gen_tmp_filenames()
        self._kl_op_dict = {}
        self._kl_keys = []
        self._print_node_mapping = {}
        self._enable_kl_op_names = [
            k for k in self.op_wise_config if self.op_wise_config[k][1] == 'kl'
        ]
        self.scale_info = {}
        self.scale_info.update(qt_config)
        self.scale_info.update({'recipes': self.recipes})
        self.scale_info.update({'int8_sequences': self.int8_sequences})
        self.scale_info.update({'bf16_ops': self.bf16_ops})
        self.scale_info.update({'fp32_ops': self.fp32_ops})

        self._fp32_model = Model(self.model._model, **self.model.kwargs)
        self._fp32_model.graph_def = self.model.graph_def
        self._fp32_model.output_tensor_names = self.output_tensor_names
        self._fp32_model.input_tensor_names = self.input_tensor_names

        self._sampling_model = Model(self.model._model, **self.model.kwargs)
        self._sampling_model.output_tensor_names = self.output_tensor_names
        self._sampling_model.input_tensor_names = self.input_tensor_names

        self._itex_model = Model(self.model._model, **self.model.kwargs)
        self._itex_model.graph_def = self.model.graph_def
        self._itex_model.output_tensor_names = self.output_tensor_names
        self._itex_model.input_tensor_names = self.input_tensor_names
        self._tmp_graph_def = copy.deepcopy(self.model.graph_def)

        self._qat_model_parameters = qat_model_parameters

    # pylint: disable=no-member
    def _inference(self, model):
        """Run the calibration on the input graph

        Args:
            model(TensorflowBaseModel): input TensorflowBaseModel
        """
        input_tensor = model.input_tensor
        output_tensor = model.output_tensor

        logger.info("Start sampling on calibration dataset.")
        for idx, (inputs, labels) in enumerate(self.data_loader):
            if len(input_tensor) == 1:
                feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), \
                    'inputs len must equal with input_tensor'
                feed_dict = dict(zip(input_tensor, inputs))

            _ = model.sess.run(output_tensor, feed_dict) if model.iter_op is None \
                else iterator_sess_run(model.sess, model.iter_op, \
                    feed_dict, output_tensor, self.calib_iteration)

            if idx + 1 == self.calib_iteration:
                break

    def _check_tf_version(self):
        is_supported_version = False
        try:
            from tensorflow import python
            if (hasattr(python, "pywrap_tensorflow")
                    and hasattr(python.pywrap_tensorflow, "IsMklEnabled")):
                from tensorflow.python.pywrap_tensorflow import IsMklEnabled
            elif hasattr(python.util, "_pywrap_util_port"):
                from tensorflow.python.util._pywrap_util_port import IsMklEnabled
            else:
                from tensorflow.python._pywrap_util_port import IsMklEnabled
            if IsMklEnabled() and (TF_SUPPORTED_MIN_VERSION <= tf.version.VERSION):
                is_supported_version = True

            if tf.version.VERSION == '2.6.0' and os.getenv('TF_ENABLE_ONEDNN_OPTS') == '1':
                is_supported_version = True
        except Exception as e:
            raise ValueError(e)
        finally:
            if tf.version.VERSION > TF_SUPPORTED_MAX_VERSION:
                logger.warning(
                    str('Please note the {} version of Intel® Optimizations for '
                        'TensorFlow is not fully verified! '
                        'Suggest to use the versions '
                        'between {} and {} if meet problem.').format(tf.version.VERSION,
                                                                     TF_SUPPORTED_MIN_VERSION,
                                                                     TF_SUPPORTED_MAX_VERSION))
            if tf.version.VERSION == '2.5.0' and os.getenv('TF_ENABLE_MKL_NATIVE_FORMAT') != '0':
                logger.fatal("Please set environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 "
                             "when Tensorflow 2.5.0 installed.")

            if tf.version.VERSION == '2.6.0' and os.getenv('TF_ENABLE_ONEDNN_OPTS') != '1':
                logger.fatal("Please set environment variable TF_ENABLE_ONEDNN_OPTS=1 "
                             "when Tensorflow 2.6.0 installed.")

            if not is_supported_version:
                raise ValueError(
                    str('Please install Intel® Optimizations for TensorFlow '
                        'or MKL enabled TensorFlow from source code '
                        'within version >={} and <={}.').format(TF_SUPPORTED_MIN_VERSION,
                                                                TF_SUPPORTED_MAX_VERSION))

    def _check_args(self):
        if self.model.workspace_path and not os.path.isdir(self.model.workspace_path) \
                and not os.path.exists(os.path.dirname(self.model.workspace_path)):
            raise ValueError('"output_graph" directory does not exist.')
        self._output_path = self.model.workspace_path

    def _gen_tmp_filenames(self):
        self._int8_dynamic_range_model_path = os.path.join(self._output_path, \
                                                      'int8_dynamic_range_graph')
        self._int8_logged_model_path = os.path.join(self._output_path, 'int8_logged_graph')
        self._fp32_logged_model_path = os.path.join(self._output_path, 'fp32_logged_graph')
        self._int8_frozen_range_model_path = os.path.join(self._output_path,
                                                          'int8_frozen_range_graph')
        self._bf16_mixed_precision_model_path = os.path.join(self._output_path,
                                                        'int8_bf16_mixed_precision_graph')

        self.output_graph = os.path.join(self._output_path, 'int8_final_fused_graph')
        # to keep temp model
        self._tmp_model = Model(self.model._model, **self.model.kwargs)
        self._tmp_model.output_tensor_names = self.output_tensor_names
        self._tmp_model.input_tensor_names = self.input_tensor_names

    def convert(self):
        """Do convert, including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.
            5) bf16 convert if the self.bf16_ops is not empty

        :return:
        """
        model = self._tmp_model
        if len(self.op_wise_config) > 0:
            model = self.quantize()

        if self.itex_mode:
            return self._itex_model

        if len(self.bf16_ops) > 0:
            model = self.bf16_convert()
        post_cse_graph_def = PostCseOptimizer(model.graph_def).do_transformation()
        post_cse_graph_def.library.CopyFrom(self.model.graph_def.library)
        model.graph_def = post_cse_graph_def

        if debug:
            model.save(self.output_graph)
            logger.info("Save converted graph file to {}.".format(self.output_graph))
        model.q_config = self.scale_info
        return model

    def _get_fp32_print_node_names(self, specified_op_list):
        offset_map = {
            "QuantizedConv2DWithBiasSumAndRelu": 3,
            "QuantizedConv2DWithBiasAndRelu": 2,
            "QuantizedConv2DWithBias": 1,
        }
        target_conv_op = []
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self._fp32_model.graph_def,
            self._fp32_model.input_node_names,
            self._fp32_model.output_node_names)

        node_name_mapping = {
            node.name: node for node in self._tmp_graph_def.node if node.op != "Const"
        }

        for node in self._tmp_graph_def.node:
            if node.op in offset_map:
                target_conv_op.append(node.name.split('_eightbit_')[0])
        fp32_node_name_mapping = {
            node.name: node
            for node in sorted_graph.node if node.op != "Const"
        }
        sorted_node_names = [i.name for i in sorted_graph.node if i.op != "Const"]

        output_node_names = []
        for i in target_conv_op:
            if specified_op_list and i not in specified_op_list:
                continue
            if node_name_mapping[i + "_eightbit_quantized_conv"].op == \
                    'QuantizedConv2DWithBiasSumAndRelu':
                start_index = sorted_node_names.index(i)
                for index, value in enumerate(sorted_node_names[start_index:]):
                    if fp32_node_name_mapping[value].op.startswith(
                            "Add") and fp32_node_name_mapping[sorted_node_names[start_index +
                                                                                index +
                                                                                1]].op == "Relu":
                        output_node_names.append(sorted_node_names[start_index + index + 1])
                        self._print_node_mapping[sorted_node_names[start_index + index + 1]] = i

            elif i in sorted_node_names:
                start_index = sorted_node_names.index(i)
                end_index = start_index + offset_map[node_name_mapping[
                    i + "_eightbit_quantized_conv"].op]
                output_node_names.append(sorted_node_names[end_index])
                self._print_node_mapping[sorted_node_names[end_index]] = i

        for i in output_node_names:
            self._kl_keys.append(';' + i + '__print__;__KL')

        fp32_graph_def = graph_pb2.GraphDef()
        fp32_graph_def.CopyFrom(self._fp32_model.graph_def)
        self._fp32_model.graph_def = InsertLogging(self._fp32_model.graph_def,
                      node_name_list=output_node_names,
                      message="__KL:",
                      summarize=-1,
                      dump_fp32=True).do_transformation()

        self._fp32_model.save(self._fp32_logged_model_path)
        self._fp32_model.graph_def = fp32_graph_def
        return self._fp32_model

    def inspect_tensor(self, original_op_list, iteration_list, work_dir, inspect_type):
        """dump the specified op's output tensor content

        Args:
            original_op_list (string list): the ops name
            iteration_list (int list): the specified iteration to dump tensor

        Returns:
            dict: key is op name while value is the content saved in np.array format.
        """
        assert iteration_list is not None, "The parameter iterations list could not be empty."
        graph_node_name_mapping = {}
        q_node_name = []
        fp32_node_name = []
        fp32_node_name_mapping = {}
        q_node_scale = {}
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self._fp32_model.graph_def,
            self._fp32_model.input_node_names,
            self._fp32_model.output_node_names)

        graph_q_node_name = []
        op_name_type_dict = {}
        quantized_node_name_postfix = '_eightbit_requantize'
        weights_tensor = {}
        g = GraphAnalyzer()
        g.graph = sorted_graph
        graph_info = g.parse_graph()

        for node in sorted_graph.node:
            node_name = node.name
            if node.op.find("Quantized") != -1:
                node_name = node.name.split(quantized_node_name_postfix)[0]
                graph_q_node_name.append(node_name)
            graph_node_name_mapping[node_name] = node

        for node in sorted_graph.node:
            node_name = node.name
            if node.op.find("Quantized") != -1:
                node_name = node.name.split(quantized_node_name_postfix)[0]

            if inspect_type in ('weight', 'all') and node.op.find("Conv") != -1:
                if node.op.find("Quantized") == -1:
                    weights_tensor[node_name] = {node.input[1]: tensor_util.MakeNdarray(
                        graph_node_name_mapping[\
                                node.input[1]].attr['value'].tensor).transpose(3,2,0,1)}
                    bias_node = None if \
                        not graph_info[node.name].outputs \
                            else graph_info[graph_info[node.name].outputs[0]].node
                    if bias_node and bias_node.op == 'BiasAdd':
                        weights_tensor[node_name][bias_node.name] = tensor_util.MakeNdarray(
                            graph_node_name_mapping[bias_node.input[1]].attr['value'].tensor)

                else:
                    if graph_info[node.input[5]].node.attr['value'].tensor.float_val:
                        min_filter_tensor = graph_info[\
                                node.input[5]].node.attr['value'].tensor.float_val
                        max_filter_tensor = graph_info[\
                                node.input[6]].node.attr['value'].tensor.float_val
                    else:
                        min_filter_tensor = tensor_util.MakeNdarray(\
                                graph_info[node.input[5]].node.attr['value'].tensor)
                        max_filter_tensor = tensor_util.MakeNdarray(\
                                graph_info[node.input[6]].node.attr['value'].tensor)

                    weight_tensor = tensor_util.MakeNdarray(\
                            graph_node_name_mapping[node.input[1]].attr['value'].tensor)
                    weight_tensor = weight_tensor = weight_tensor.astype('float')

                    DequantizeWeight(weight_tensor, min_filter_tensor,max_filter_tensor)
                    weights_tensor[node_name] = {node.input[1]:weight_tensor.transpose(3,2,0,1)}

                    weights_tensor[node_name][node.input[2]] = tensor_util.MakeNdarray(
                            graph_node_name_mapping[node.input[2]].attr['value'].tensor)

        for op_name in original_op_list:
            if isinstance(op_name, tuple):
                op_name = op_name[0]
                op_type = op_name[1]
            else:
                #TODO op_type set to conv2d for fast_bias_correction and weigh correction.
                op_type = "conv2d" #TODO

            if op_type not in ["conv2d"]:
                continue

            op_name_type_dict[op_name] = op_type
            if op_name in graph_q_node_name:
                q_node_name.append(op_name + quantized_node_name_postfix)
                q_node = graph_node_name_mapping[op_name]
                q_out_min = graph_node_name_mapping[
                    q_node.input[-2]].attr["value"].tensor.float_val[0]
                q_out_max = graph_node_name_mapping[
                    q_node.input[-1]].attr["value"].tensor.float_val[0]
                q_node_scale[op_name + quantized_node_name_postfix] = (q_node.op, q_out_min,
                                                                       q_out_max)
            else:
                fp32_node_name.append(op_name)
                node_op =  graph_node_name_mapping[op_name].op
                if node_op in ("Conv2D", "DepthwiseConv2dNative"):
                    _, matched_nodes = FuseNodeStartWithConv2d(
                        input_graph=sorted_graph,
                        patterns=self.int8_sequences[node_op],
                        remove_redundant_quant_flag=True,
                        op_wise_cfg=(False, "minmax", False, 7.0),
                        start_node_name=op_name,
                        device=self.device).get_longest_fuse()

                    if matched_nodes:
                        fp32_node_name_mapping[matched_nodes[-1]] = op_name
                else:
                    fp32_node_name_mapping[op_name] = op_name

        InsertLogging(sorted_graph,
                      node_name_list=fp32_node_name_mapping.keys(),
                      message="__KL:",
                      summarize=-1,
                      dump_fp32=True).do_transformation()

        if q_node_name:
            sorted_graph = InsertLogging(sorted_graph,
                                         node_name_list=q_node_name,
                                         message="__KL:",
                                         summarize=-1).do_transformation()

        tmp_dump_file = os.path.join(work_dir, 'kl.log')

        model = Model(sorted_graph)
        model.output_tensor_names = self.output_tensor_names
        model.input_tensor_names = self.input_tensor_names
        with CaptureOutputToFile(tmp_dump_file):
            self._inference(model)

        with open(tmp_dump_file) as f:
            disk_content = f.readlines()

        filter_content = (i for i in disk_content if i.startswith(';'))

        dump_tensor_content = {}

        for i in filter_content:
            contents = i.split('__print__;__KL:')
            node_name = contents[0][1:]
            node_content = str2array(contents[1])

            if node_name not in dump_tensor_content:
                dump_tensor_content[node_name] = []
            dump_tensor_content[node_name].append(node_content)

        activation_content = []
        for iter_idx in iteration_list:
            result_disk = {}
            for k, v in dump_tensor_content.items():
                if k in fp32_node_name_mapping:
                    key = fp32_node_name_mapping[k]
                    result_disk[(key, op_name_type_dict[key])] = \
                            {key: v[iter_idx - 1].transpose(0,3,1,2)}
                else:
                    result_key = k.split(quantized_node_name_postfix)[0]
                    result_disk[(result_key, op_name_type_dict[result_key])] = \
                            {result_key: Dequantize(v[0], q_node_scale[k]).transpose(0,3,1,2)}
            activation_content.append(result_disk)

        final_result = {'weight': weights_tensor, 'activation': activation_content}

        return final_result

    def _analysis_rnn_model(self):
        g = GraphAnalyzer()
        g.graph = self._tmp_graph_def
        graph_info = g.parse_graph()
        rnn_pattern = [['TensorArrayV3'], ['Enter'], ['TensorArrayReadV3'], \
            ['MatMul'], ['BiasAdd']]
        target_nodes = g.query_fusion_pattern_nodes(rnn_pattern)
        res = {}
        for i in target_nodes:
            if i[-3] not in self.bf16_ops and i[-3] not in self.fp32_ops:
                res[(i[-3], i[-2])] = graph_info[i[1]].node.attr['frame_name'].s.decode()

        return res

    def _search_y_pattern_for_itex(self):
        """Search the Y pattern for itex and return the op name.
        """
        g = GraphAnalyzer()
        g.graph = self._fp32_model.graph_def
        g.parse_graph()
        y_pattern = [['Conv2D', 'MatMul'], ['BiasAdd'], ['Add'], ('Relu',)]
        target_nodes = g.query_fusion_pattern_nodes(y_pattern)

        res = {}
        for i in target_nodes:
            if i[2] not in res:
                res[i[2]] = 1
            else:
                res[i[2]] += 1
        return [(i,) for i in res if res[i] == 2]

    def remove_fake_quantize(self):
        self._tmp_graph_def = RemoveFakeQuantOpOptimizer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        self._tmp_model.graph_def = self._tmp_graph_def

        # Debug
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
                '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
            'fp32_remove_fake_model.pb',
            as_text=False)

    def _trace_graph(self, graph_def):
        graph_analyzer = GraphAnalyzer()
        graph_analyzer.graph = graph_def
        graph_info = graph_analyzer.parse_graph()

        trace = OrderedDict()

        stack = []
        for input_name in self.input_tensor_names:
            stack.append(input_name)

        visited = {}

        while stack:
            node_name = stack.pop()
            node_info = graph_info[node_name]

            if node_name in visited:
                visited[node_name] += 1
            else:
                visited[node_name] = 1 if node_info.node.input else 0
                for node_input in node_info.node.input:
                    if node_input in graph_info:
                        if graph_info[node_input].node.op == 'Const':
                            visited[node_name] += 1
                    else:
                        visited[node_name] += 1

            if visited[node_name] == len(node_info.node.input):
                trace[node_name] = node_info
                for output in node_info.outputs:
                    stack.append(output)

        return trace, graph_info

    def find_next_fq_parameters(self, graph_info, graph_trace):
        print('Find FQ parameters:')
        while True:
            node_name, node_info = graph_trace.popitem(last=False)
            print(f'FP32_GRAPH:{node_name} | {node_info.node.op}')
            if node_info.node.op in ['FakeQuantWithMinMaxVars']:
                narrow_range = node_info.node.attr['narrow_range'].b
                min_node = graph_info[node_info.node.input[1]].node
                max_node = graph_info[node_info.node.input[2]].node
                min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
                max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
                q_type = tf.dtypes.qint8 if np.min(min_value) < 0 else tf.dtypes.quint8

                if q_type == tf.dtypes.qint8 and narrow_range == False:
                    print('Warning: type qint8, narrow_range = False')

                return min_value, max_value, q_type

    def get_input_type(self, graph_info, node_name):
        if graph_info[node_name].node.op in ['Requantize']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['out_type'].type)
            min_node = graph_info[graph_info[node_name].node.input[3]].node
            max_node = graph_info[graph_info[node_name].node.input[4]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        if graph_info[node_name].node.op in ['QuantizedMatMulWithBiasAndReluAndRequantize']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['Toutput'].type)
            min_node = graph_info[graph_info[node_name].node.input[7]].node
            max_node = graph_info[graph_info[node_name].node.input[8]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        if graph_info[node_name].node.op in ['QuantizedConv2DWithBiasAndReluAndRequantize',
                                             'QuantizedConv2DWithBiasAndRequantize']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['out_type'].type)
            min_node = graph_info[graph_info[node_name].node.input[7]].node
            max_node = graph_info[graph_info[node_name].node.input[8]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        if graph_info[node_name].node.op in ['Quantize', 'QuantizeV2']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['T'].type)
            min_node = graph_info[graph_info[node_name].node.input[1]].node
            max_node = graph_info[graph_info[node_name].node.input[2]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        return self.get_input_type(graph_info, graph_info[node_name].node.input[0])

    def _quantize(self, input, min_input, max_input, type):
        with tf.Graph().as_default() as quantized_graph:
            input_ = tf.compat.v1.placeholder(tf.float32, shape=input.shape, name='input')
            min_input_ = tf.constant(min_input, dtype=tf.float32, shape=min_input.shape)
            max_input_ = tf.constant(max_input, dtype=tf.float32, shape=max_input.shape)
            narrow_range = type == tf.dtypes.qint8
            q_input_, min_output_, max_output_ = tf.quantization.quantize(
                input_,
                min_input_,
                max_input_,
                type,
                mode='SCALED',
                round_mode='HALF_TO_EVEN',
                narrow_range=narrow_range,
                ensure_minimum_range=0.0)

            with tf.compat.v1.Session(graph=quantized_graph) as sess:
                out = sess.run(
                    [q_input_, min_output_, max_output_], feed_dict={input_: input})

                return tuple(out)

    def _generate_int32_bias_for_matmul(self, bias_tensor, input_range, filter_range):
        bias_scale = 255.0 * 127.0 / (input_range * filter_range)

        int32_bias =np.around(bias_tensor * bias_scale).astype('int32')

        return int32_bias


    def _fill_qat_parameters(self):
        quantized_graph_trace, quantized_graph_info = self._trace_graph(self._tmp_graph_def)
        fp32_graph_trace, fp32_graph_info = self._trace_graph(self._qat_model_parameters['const_fold_graph_def'])

        print('Fill QAT parameters:')
        for node_name, node_info in quantized_graph_trace.items():
            print(f'Q_GRAPH:{node_name} | {node_info.node.op}')
            if node_info.node.op in ['Quantize', 'QuantizeV2']:
                min, max, q_type = self.find_next_fq_parameters(
                    fp32_graph_info,
                    fp32_graph_trace)

                min_node = quantized_graph_info[node_info.node.input[1]].node
                max_node = quantized_graph_info[node_info.node.input[2]].node

                helper.set_attr_dtype(node_info.node, "T", q_type)
                helper.set_attr_string(node_info.node, "mode", b"SCALED")
                helper.set_attr_string(node_info.node, "round_mode", b"HALF_TO_EVEN")
                helper.set_attr_bool(node_info.node, "narrow_range", q_type == tf.dtypes.qint8)
                helper.set_attr_float(node_info.node, "ensure_minimum_range", 0.0)

                helper.set_attr_dtype(min_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_node, "value", min, tf.dtypes.float32, min.shape)
                helper.set_attr_dtype(max_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_node, "value", max, tf.dtypes.float32, max.shape)


            if node_info.node.op in ['Requantize']:
                min, max, q_type = self.find_next_fq_parameters(
                    fp32_graph_info,
                    fp32_graph_trace)

                min_node = quantized_graph_info[node_info.node.input[3]].node
                max_node = quantized_graph_info[node_info.node.input[4]].node

                helper.set_attr_dtype(node_info.node, "out_type", q_type)

                helper.set_attr_dtype(min_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_node, "value", min, tf.dtypes.float32, min.shape)
                helper.set_attr_dtype(max_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_node, "value", max, tf.dtypes.float32, max.shape)

            if node_info.node.op in ['QuantizedConv2DWithBiasAndRelu',
                                     'QuantizedConv2DWithBiasAndReluAndRequantize',
                                     'QuantizedConv2DWithBias',
                                     'QuantizedConv2DWithBiasAndRequantize']:
                q_input_type, min_input, max_input = self.get_input_type(quantized_graph_info, node_info.node.input[0])
                helper.set_attr_dtype(node_info.node, "Tinput", q_input_type)
                helper.set_attr_dtype(node_info.node, "Tfilter", tf.dtypes.qint8)
                helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.float32)

                conv_name = node_info.node.name.replace('_eightbit_quantized_conv', '')
                conv_name = conv_name.replace('_eightbit_requantize', '')
                filter = self._qat_model_parameters['conv_weights'][conv_name]
                min_filter = self._qat_model_parameters['fq_weights'][conv_name]['min']
                max_filter = self._qat_model_parameters['fq_weights'][conv_name]['max']
                print(f'{conv_name} : min {min_filter}, max {max_filter}')
                q_filter, q_min, q_max = self._quantize(filter, min_filter, max_filter, tf.dtypes.qint8)
                bias = self._qat_model_parameters['bias_adds'][conv_name]
                if conv_name in self._qat_model_parameters['scales']:
                    scale = self._qat_model_parameters['scales'][conv_name]
                    min_scaled_filter = min_filter * scale
                    max_scaled_filter = max_filter * scale
                else:
                    min_scaled_filter = min_filter
                    max_scaled_filter = max_filter

                filter_node = quantized_graph_info[node_info.node.input[1]].node
                bias_node = quantized_graph_info[node_info.node.input[2]].node
                min_filter_node = quantized_graph_info[node_info.node.input[5]].node
                max_filter_node = quantized_graph_info[node_info.node.input[6]].node

                helper.set_attr_dtype(filter_node, "dtype", tf.dtypes.qint8)
                helper.set_attr_tensor(filter_node, "value", q_filter, tf.dtypes.qint8, q_filter.shape)

                helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(bias_node, "value", bias, tf.dtypes.float32, bias.shape)

                helper.set_attr_dtype(min_filter_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_filter_node, "value", min_scaled_filter, tf.dtypes.float32, min_scaled_filter.shape)

                helper.set_attr_dtype(max_filter_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_filter_node, "value", max_scaled_filter, tf.dtypes.float32, max_scaled_filter.shape)

                if node_info.node.op in ['QuantizedConv2DWithBiasAndReluAndRequantize',
                                         'QuantizedConv2DWithBiasAndRequantize']:
                    re_min, re_max, req_type = self.find_next_fq_parameters(
                        fp32_graph_info,
                        fp32_graph_trace)

                    helper.set_attr_dtype(node_info.node, "out_type", req_type)

                    min_freezed_output = quantized_graph_info[node_info.node.input[7]].node
                    max_freezed_output = quantized_graph_info[node_info.node.input[8]].node

                    helper.set_attr_dtype(min_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(min_freezed_output, "value", re_min, tf.dtypes.float32,
                                           re_min.shape)

                    helper.set_attr_dtype(max_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(max_freezed_output, "value", re_max, tf.dtypes.float32,
                                           re_max.shape)

            if node_info.node.op in ['QuantizedMatMulWithBias',
                                     'QuantizedMatMulWithBiasAndReluAndRequantize']:
                q_input_type, min_input, max_input = self.get_input_type(quantized_graph_info, node_info.node.input[0])
                helper.set_attr_dtype(node_info.node, "T1", q_input_type)
                helper.set_attr_dtype(node_info.node, "T2", tf.dtypes.qint8)
                helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.float32)
                helper.set_attr_string(node_info.node, "input_quant_mode", b"SCALED")

                matmul_name = node_info.node.name.replace('_eightbit_quantized_mat_mul', '')
                matmul_name = matmul_name.replace('_eightbit_requantize', '')
                b = self._qat_model_parameters['mat_weights'][matmul_name]
                min_filter = self._qat_model_parameters['fq_weights'][matmul_name]['min']
                max_filter = self._qat_model_parameters['fq_weights'][matmul_name]['max']
                print(f'{matmul_name} : min {min_filter}, max {max_filter}')
                q_b, q_min, q_max = self._quantize(b, min_filter, max_filter, tf.dtypes.qint8)

                bias = self._qat_model_parameters['bias_adds'][matmul_name]

                b_node = quantized_graph_info[node_info.node.input[1]].node
                bias_node = quantized_graph_info[node_info.node.input[2]].node
                min_b_node = quantized_graph_info[node_info.node.input[5]].node
                max_b_node = quantized_graph_info[node_info.node.input[6]].node

                helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(bias_node, "value", bias, tf.dtypes.float32, bias.shape)

                helper.set_attr_dtype(b_node, "dtype", tf.dtypes.qint8)
                helper.set_attr_tensor(b_node, "value", q_b, tf.dtypes.qint8, q_b.shape)

                helper.set_attr_dtype(min_b_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_b_node, "value", min_filter, tf.dtypes.float32,
                                       min_filter.shape)

                helper.set_attr_dtype(max_b_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_b_node, "value", max_filter, tf.dtypes.float32,
                                       max_filter.shape)

                if node_info.node.op in ['QuantizedMatMulWithBiasAndReluAndRequantize']:
                    re_min, re_max, req_type = self.find_next_fq_parameters(
                        fp32_graph_info,
                        fp32_graph_trace)

                    helper.set_attr_dtype(node_info.node, "Toutput", req_type)

                    min_freezed_output = quantized_graph_info[node_info.node.input[7]].node
                    max_freezed_output = quantized_graph_info[node_info.node.input[8]].node

                    helper.set_attr_dtype(min_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(min_freezed_output, "value", re_min, tf.dtypes.float32,
                                           re_min.shape)

                    helper.set_attr_dtype(max_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(max_freezed_output, "value", re_max, tf.dtypes.float32,
                                           re_max.shape)

                    if q_input_type != tf.quint8:
                        raise RuntimeError(f'Input type is tf.qint8 for {node_info.node.name}. tf.quint8 is expected')
                    filter_range = np.maximum(np.abs(min_filter), np.abs(max_filter))
                    input_range = max_input
                    bias_int32 = self._generate_int32_bias_for_matmul(bias, input_range, filter_range)

                    helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.qint32)
                    helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.qint32)
                    helper.set_attr_tensor(bias_node, "value", bias_int32, tf.dtypes.qint32, bias_int32.shape)

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

        # Debug
        tf.io.write_graph(
            self._tmp_graph_def,
            '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
            'fill_qat_parameters.pb',
            as_text=False)

    def _fuse_requantize(self):
        self._tmp_graph_def = FuseConvRequantizeTransformer(
            self._tmp_graph_def,
            self.device).do_transformation()

        self._tmp_graph_def = FuseMatMulRequantizeTransformer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

        # Debug
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
            'fuse_requantize.pb',
            as_text=False)

    def _qat_fuse_requantize_with_fused_quantized_node(self):
        pass

        # if not self.fake_quant:
        #     self._tmp_graph_def = FuseMatMulRequantizeTransformer(
        #         self._tmp_graph_def).do_transformation()
        #
        #     self._tmp_graph_def = FuseMatMulRequantizeDequantizeTransformer(
        #         self._tmp_graph_def).do_transformation()
        # self._tmp_graph_def = StripUnusedNodesOptimizer(
        #     self._tmp_graph_def,
        #     self._tmp_model.input_node_names,
        #     self._tmp_model.output_node_names).do_transformation()
        #
        # self._tmp_graph_def = RemoveTrainingNodesOptimizer(
        #     self._tmp_graph_def,
        #     protected_nodes=self._tmp_model.output_node_names).do_transformation()
        #
        # # self._tmp_graph_def = FoldBatchNormNodesOptimizer(
        # #     self._tmp_graph_def).do_transformation()
        #
        # if 'scale_propagation_concat' in self.recipes and self.recipes['scale_propagation_concat']:
        #     self._tmp_graph_def = RerangeQuantizedConcat(self._tmp_graph_def,
        #                                              self.device).do_transformation()
        #
        # self._tmp_graph_def = MetaInfoChangingMemOpOptimizer(
        #     self._tmp_graph_def).do_transformation()
        #
        # if self.advance_config is not None and \
        #    deep_get(self.advance_config, 'bias_correction') is not None:
        #     self._tmp_graph_def = BiasCorrection(
        #         self._tmp_graph_def, self.model.graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

        #Debug
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
            'int8_fused_quantized_model.pb',
            as_text=False)

    def quantize(self):
        """Quantize graph only (without optimizing fp32 graph), including:
            1) quantize graph,
            2) calibration,
            3) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        try:
            if self.fake_quant:
                self.remove_fake_quantize()

            self._quantize_graph()
            self._rnn_details = self._analysis_rnn_model()
            self.quantized_node_info.extend(self._rnn_details.keys())
            self.quantized_node_info = [tuple(i) for i in self.quantized_node_info]

            if self.fake_quant:
                self._fuse_requantize()
                self._fill_qat_parameters()
                #self._qat_fuse_requantize_with_fused_quantized_node()
            else:
                if self._enable_kl_op_names:
                    self._get_fp32_print_node_names(self._enable_kl_op_names)
                    self._generate_calibration_data(self._fp32_logged_model_path,
                                                    self._fp32_print_data,
                                                    True)

                output_tensor_names = copy.deepcopy(self.model.output_tensor_names)
                sampling_graph_def = copy.deepcopy(self._fp32_model.graph_def)

                if self.itex_mode:
                    self.quantized_node_info.extend(self._search_y_pattern_for_itex())

                for i in self.quantized_node_info:
                    frame_name = self._rnn_details[i] if i in self._rnn_details else None
                    sampling_graph_def, output_names = InsertPrintMinMaxNode(
                        sampling_graph_def, i[0], i[-1], frame_name).do_transformation()
                    output_tensor_names.extend(output_names)

                # Debug
                import tensorflow as tf
                tf.io.write_graph(
                    sampling_graph_def,
                    '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
                    'sampling_model.pb',
                    as_text=False)

                if self.quantized_node_info:
                    sampling_graph_def.library.CopyFrom(self.model.graph_def.library)
                    self._sampling_model.graph_def = sampling_graph_def
                    self._sampling_model.output_tensor_names = output_tensor_names
                    tmp_dump_file = tempfile.mkstemp(suffix='.log')[1]
                    with CaptureOutputToFile(tmp_dump_file):
                        self._inference(self._sampling_model)
                    self._calibration_data = Helper.gen_valid_sampling_log(tmp_dump_file)

                if self.itex_mode:
                    self._itex_model.graph_def = GenerateITEXModel(
                        self._itex_model, self._calibration_data).do_transformation()
                    self._itex_model.graph_def.library.CopyFrom(
                        self.model.graph_def.library)

                    return self._itex_model

                if len(self._calibration_data) > 0:
                    self._freeze_requantization_ranges(self._kl_op_dict)
                    self._fuse_requantize_with_fused_quantized_node()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._tmp_model = None
            logger.error("Fail to quantize graph due to {}.".format(str(e)))
        finally:
            if not debug:
                self._post_clean()
            return self._tmp_model

    def bf16_convert(self):
        """Convert fp32 nodes in bf16_node to bf16 dtype based on
           FP32 + INT8 mixed precision graph.
        """
        try:
            self._tmp_model.graph_def = BF16Convert(
                self._tmp_model.graph_def,
                self.fp32_ops,
                self.bf16_ops).do_transformation()

        except Exception as e:
            self._tmp_model = None
            logger.error("Fail to convert graph due to {}.".format(str(e)))
        finally:
            if debug:
                self._tmp_model.save(self._bf16_mixed_precision_model_path)

            return self._tmp_model

    def _quantize_graph(self):
        """quantize graph."""

        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))

        #Debug
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
            'fp32_pre_quantized_model.pb',
            as_text=False)

        self._tmp_graph_def = FusePadWithConv2DOptimizer(
            self._tmp_graph_def,
            non_pad_ops,
            self._tmp_model.input_node_names,
            self.op_wise_config).do_transformation()

        self._tmp_graph_def = QuantizeGraphHelper().get_sorted_graph(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names)

        self._tmp_graph_def, self.quantized_node_info = QuantizeGraphForIntel(
            self._tmp_graph_def,
            self._tmp_model.output_node_names,
            self.op_wise_config,
            self.int8_sequences,
            self.device,
            self.fake_quant).do_transform()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        if debug:
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_dynamic_range_model_path)

    def _generate_calibration_data(self, tmp_path, output_data, enable_kl_algo=False):

        tmp_dump_file = os.path.join(os.path.dirname(self.output_graph), 'requant_min_max.log')

        logger.debug("Generate calibration data and save to {}.".format(tmp_dump_file))

        model = Model(tmp_path, **self._tmp_model.kwargs)
        model.output_tensor_names = self.output_tensor_names
        model.input_tensor_names = self.input_tensor_names

        with CaptureOutputToFile(tmp_dump_file):
            self._inference(model)

        with open(tmp_dump_file, errors='ignore') as f:
            output_data.extend(f.readlines())

        for line in output_data:
            if enable_kl_algo and line.rsplit(':')[0] in self._kl_keys:
                fp32_data = get_all_fp32_data(line.rsplit(':')[-1])
                key = self._print_node_mapping[line[1:].split('__print')
                                               [0]] + '_eightbit_requant_range'
                if key not in self._kl_op_dict:
                    self._kl_op_dict[key] = get_tensor_histogram(fp32_data)
                else:
                    self._kl_op_dict[key] = combine_histogram(self._kl_op_dict[key], fp32_data)

    def _freeze_requantization_ranges(self, additional_data=None):
        self._tmp_graph_def, quantizev2_max = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__max:').do_transformation()
        self._tmp_graph_def, quantizev2_min = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__min:').do_transformation()
        self._tmp_graph_def, requant_min_max= FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__requant_min_max',
            tensor_data= additional_data,
            device=self.device,
            ).do_transformation()

        self.scale_info.update(quantizev2_max)
        self.scale_info.update(quantizev2_min)
        self.scale_info.update(requant_min_max)

        self._tmp_graph_def = QuantizedRNNConverter(
            self._tmp_graph_def, self._calibration_data, self._rnn_details).do_transformation()

        if 'scale_propagation_max_pooling' in self.recipes and \
                self.recipes['scale_propagation_max_pooling']:
            self._tmp_graph_def = ScaleProPagationTransformer(
                self._tmp_graph_def).do_transformation()

        if debug:
            self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_frozen_range_model_path)

    def _fuse_requantize_with_fused_quantized_node(self):
        if self.fake_quant:
            self._tmp_graph_def = FreezeFakeQuantOpOptimizer(
                self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseConvRequantizeTransformer(
            self._tmp_graph_def,
            self.device).do_transformation()

        if not self.fake_quant:
            self._tmp_graph_def = FuseMatMulRequantizeTransformer(
                self._tmp_graph_def).do_transformation()

            self._tmp_graph_def = FuseMatMulRequantizeDequantizeTransformer(
                self._tmp_graph_def).do_transformation()
        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names).do_transformation()

        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self._tmp_graph_def,
            protected_nodes=self._tmp_model.output_node_names).do_transformation()

        # self._tmp_graph_def = FoldBatchNormNodesOptimizer(
        #     self._tmp_graph_def).do_transformation()

        if 'scale_propagation_concat' in self.recipes and self.recipes['scale_propagation_concat']:
            self._tmp_graph_def = RerangeQuantizedConcat(self._tmp_graph_def,
                                                     self.device).do_transformation()

        self._tmp_graph_def = MetaInfoChangingMemOpOptimizer(
            self._tmp_graph_def).do_transformation()

        if self.advance_config is not None and \
           deep_get(self.advance_config, 'bias_correction') is not None:
            self._tmp_graph_def = BiasCorrection(
                self._tmp_graph_def, self.model.graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

        #Debug
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            '/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat',
            'int8_fused_quantized_model.pb',
            as_text=False)

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if os.path.exists(self._int8_logged_model_path) and \
            os.path.isdir(self._int8_logged_model_path):
            import shutil
            shutil.rmtree(self._int8_logged_model_path)

        elif gfile.Exists(self._int8_logged_model_path + '.pb'):
            os.remove(self._int8_logged_model_path + '.pb')
