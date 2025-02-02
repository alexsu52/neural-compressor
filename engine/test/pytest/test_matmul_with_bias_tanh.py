import unittest
from collections import OrderedDict
from engine.converter.ops.op import OPERATORS, Operator
from engine.converter.ops.tensor import Tensor
from engine.converter.graph import Graph
from engine.converter.sub_graph.matmul_with_bias_tanh import MatmulWithBiasTanh
import numpy as np


class TestMatmulWithBiasTanh(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_matmul_with_bias_tanh_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mat_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(data=np.array(1)), Tensor(data=np.array(1)), 
                            Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['tanh'])]
        mat_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'src1_perm': '1,0'}))
        
        tanh_node = OPERATORS['Tanh']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['tanh'])]
        output_tensors = [Tensor(name='tanh:0', source_op=['tanh'],
                                dest_op=[])]
        tanh_node.construct('tanh', 'Tanh', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, mat_node, tanh_node])
        graph = MatmulWithBiasTanh()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('1,0', graph.nodes[1].attr['src1_perm'])
        self.assertEqual('tanh', graph.nodes[1].name)
        self.assertEqual('tanh', graph.nodes[1].attr['append_op'])


if __name__ == "__main__":
    unittest.main()