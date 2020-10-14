"""Tests for tuner"""
import numpy as np
import unittest
import os
import yaml
import tensorflow as tf
import importlib

def build_fake_yaml():
    fake_yaml = '''
        framework: 
          - name: tensorflow
            inputs: x
            outputs: op_to_store
        device: cpu
        tuning:
            strategy:
               name: tpe
            accuracy_criterion:
              - relative: 0.01
        snapshot:
          - path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml2():
    fake_yaml = '''
        framework: 
          - name: tensorflow
            inputs: x
            outputs: op_to_store
        device: cpu
        tuning:
            strategy:
                name: tpe
            max_trials: 5
            accuracy_criterion:
              - relative: -0.01
        snapshot:
          - path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filter=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.compat.v1.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

def build_dataloader():
    from ilit.data import DataLoader
    from ilit.data import DATASETS
    dataset = DATASETS('tensorflow')['dummy']
    dataloader = DataLoader('tensorflow', dataset)
    return dataloader

def accuracy_check(self, input_graph=None):
    return 100

class TestTuner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml()
        build_fake_yaml2()
        self.dataloader = build_dataloader()
        os.mkdir('saved')

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml2.yaml')
        os.remove('saved/ilit-tpe.snapshot')
        os.remove('saved/tpe_best_result.csv')
        os.remove('saved/tpe_trials.csv')
        os.rmdir('saved')

    def test_run_tpe_one_trial(self):
        from ilit.strategy import strategy
        from ilit import tuner as iLit

        at = iLit.Tuner('fake_yaml.yaml')
        at.tune(
            self.constant_graph,
            q_dataloader=self.dataloader,
            eval_dataloader=self.dataloader,
            eval_func=accuracy_check
        )

    def test_run_tpe_max_trials(self):
        from ilit.strategy import strategy
        from ilit import tuner as iLit

        at = iLit.Tuner('fake_yaml2.yaml')
        at.tune(
            self.constant_graph,
            q_dataloader=self.dataloader,
            eval_dataloader=self.dataloader,
            eval_func=accuracy_check
        )

    def test_loss_calculation(self):
        from ilit.strategy.tpe import TpeTuneStrategy
        from ilit import tuner as iLit

        at = iLit.Tuner('fake_yaml.yaml')
        testObject = TpeTuneStrategy(self.constant_graph, at.conf, self.dataloader)
        testObject._calculate_loss_function_scaling_components(0.01, 2, testObject.loss_function_config)
        # check if latency difference between min and max corresponds to 10 points of loss function
        tmp_val = testObject.calculate_loss(0.01, 2, testObject.loss_function_config)
        tmp_val2 = testObject.calculate_loss(0.01, 1, testObject.loss_function_config)
        self.assertTrue(True if int(tmp_val2 - tmp_val) == 10 else False)
        # check if 1% of acc difference corresponds to 10 points of loss function
        tmp_val = testObject.calculate_loss(0.02, 2, testObject.loss_function_config)
        tmp_val2 = testObject.calculate_loss(0.03, 2, testObject.loss_function_config)
        self.assertTrue(True if int(tmp_val2 - tmp_val) == 10 else False)

if __name__ == "__main__":
    unittest.main()
