#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer

class RemoveFakeQuantOpOptimizer(GraphRewriterBase):
    """Remove fake quantize operations.
    """
    def __init__(self, model):
        super().__init__(model)

        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

    def _remove_all_fake_quants(self):
        _const_node = []

        for node_name in list(self.graph_info.keys()):
            node = self.graph_info[node_name].node
            if node.op == 'FakeQuantWithMinMaxVars':
                origin_outputs = list(self.graph_info[node_name].outputs)
                min_node_name = self.graph_info[node.input[1]].node.name
                max_node_name = self.graph_info[node.input[2]].node.name
                _const_node.append(min_node_name)
                _const_node.append(max_node_name)

                self.graph_analyzer.remove_node_with_single_input_output(node_name)

                for j in origin_outputs[1:]:
                    output_node = self.graph_info[j].node
                    if len(output_node.input) == 1 and \
                        output_node.op == 'Const' and output_node.input[0] == '^' + node.name:
                        self.graph_info[j].node.ClearField('input')
                    elif output_node.op == 'NoOp' :
                        new_noop_input = [
                            noop_input for noop_input in output_node.input \
                                if noop_input != '^' + node.name]
                        output_node.ClearField('input')
                        output_node.input.extend(new_noop_input)

        # remove those left const nodes used by FakeQuantWithMinMaxVars
        for node_name in list(self.graph_info.keys()):
            if node_name in _const_node:
                self.graph_analyzer.remove_node(node_name)

    @dump_elapsed_time("Pass RemoveFakeQuantOpOptimizer")
    def do_transformation(self):
        self._remove_all_fake_quants()
        return GraphAnalyzer().dump_graph()
