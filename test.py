from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import cv2
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

graph = load_graph("digit_model.pb")
input_layer = "conv2d_1_input"
output_layer = "k2tfout_0"
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)
img = cv2.imread("test.jpg",0)
img = img * (1./255)
img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)
print(img.shape)
sess = tf.Session(graph=graph)
input_tensor = input_operation.outputs[0]
output_tensor = output_operation.outputs[0]
res = sess.run(output_tensor, feed_dict={input_tensor:img})
print(res)
print(np.argmax(res))