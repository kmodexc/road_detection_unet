import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf

SAVEDMODEL_PATH = "tensorrt"

root = tf.saved_model.load(SAVEDMODEL_PATH)

infer = root.signatures['serving_default']
output_tensorname = list(infer.structured_outputs.keys())[0]

probs = infer(features)[output_tensorname].numpy()