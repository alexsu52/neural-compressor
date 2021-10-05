import resource

import tensorflow as tf
from tensorflow import keras

tf.__version__ = '2.4.2'

_, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Load MNIST dataset
#mnist = keras.datasets.mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
#train_images = train_images / 255.0
#test_images = test_images / 255.0

#model = tf.keras.models.load_model("baseline_model")
#model = tf.keras.applications.MobileNet(input_shape=(224,224,3))

import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model

from nncf import NNCFConfig
from nncf.tensorflow.quantization.algorithm import QuantizationBuilder

def nncf_quantization(model):
    print('NNCF - quantize!!!!')
    print('!-' * 20)

    nncf_config = NNCFConfig({
        "compression": {
            "algorithm": "quantization",
            "disable_saturation_fix": True,
            "activations": {
                "per_channel": False
            }
        }
    }
    )

    #nncf_config = register_default_init_args(nncf_config, )

    q_builder = QuantizationBuilder(nncf_config, should_init=False)
    q_model = q_builder.apply_to(model)

    q_model.save("nncf_trained_qat_model.h5", save_format='h5')

    return q_model


def tfmot_quantization(model):
    print('TFMOT - quantize!!!!')
    print('!-' * 20)
    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    #q_aware_model.compile(optimizer='adam',
    #              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #              metrics=['accuracy'])

    q_aware_model.summary()

    #train_images_subset = train_images[0:1000] # out of 60000
    #train_labels_subset = train_labels[0:1000]

    #q_aware_model.fit(train_images_subset, train_labels_subset,
    #                  batch_size=500, epochs=1, validation_split=0.1)

    #_, q_aware_model_accuracy = q_aware_model.evaluate(
    #   test_images, test_labels, verbose=0)

    #print('Quant test accuracy:', q_aware_model_accuracy)
    q_aware_model.save("tfmot_trained_qat_model")
    q_aware_model.save("tfmot_trained_qat_model.h5", save_format='h5')

    return q_aware_model

def lpot_convert(q_aware_model, model_name):
    print('LPOT - convert!!!!')
    print('!-' * 20)
    from lpot.experimental import ModelConversion, common
    conversion = ModelConversion()
    conversion.source = 'QAT'
    conversion.destination = 'default'
    conversion.model = q_aware_model#common.Model('trained_qat_model')
    q_model = conversion()
    q_model.save(model_name)

def lpot_quantization(model, model_dir):
    print('LPOT - quantization!!!!')
    print('!-' * 20)
    from lpot.experimental import Quantization, common
    quantizer = Quantization('/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat_conversion/dummy.yaml')
    quantizer.model = common.Model(model)
    q_model = quantizer()
    q_model.save(model_dir)

#model = tf.keras.applications.MobileNet(input_shape=(224,224,3))
#model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3))
#model = tf.keras.applications.MobileNetV3Small(input_shape=(224,224,3))
model = tf.keras.applications.ResNet50(input_shape=(224,224,3))
#model_dir = 'lpot_quantized_model'
#lpot_quantization(model, model_dir)
#model_dir = 'tfmot_quantized_model'
#lpot_convert(tfmot_quantization(model), model_dir)
model_dir = 'nncf_quantized_model'
lpot_convert(nncf_quantization(model), model_dir)

from lpot.experimental import Benchmark, common

print('LPOT - benchmark!!!!')
print('!-' * 20)
evaluator = Benchmark('/home/alexsu/work/projects/algo/nncf_tf/source/nncf-tf/lpot/examples/tensorflow/qat_conversion/dummy.yaml')
evaluator.model = common.Model(model_dir)
# evaluator.b_dataloader = dataloader()
evaluator()