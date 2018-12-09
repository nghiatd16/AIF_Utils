#Khai báo thư viện
import tensorflow as tf
import numpy as np

#Khai báo tham số cho model
model_params = {
    'n_classes': 10,
    'input_shape': (28, 28, 1),
    'batch_size': 100
}

hyper_params = {
    'learning_rate': 0.0001,
    'drop_out': 0.25
}

tf.logging.set_verbosity(tf.logging.INFO)

def conv2d_fn(input_tensor, k_size, n_out):
    return tf.layers.conv2d(inputs= input_tensor, \
                            filters= n_out, \
                            kernel_size= k_size, \
                            activation= tf.nn.relu, \
                            use_bias= True)

def maxpool2d_fn(input_tensor, p_size, strides):
    return tf.layers.max_pooling2d(inputs= input_tensor, pool_size= p_size, strides= strides)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.cast(features, tf.float32)
    net = conv2d_fn(input_tensor= input_layer, k_size= 2, n_out = 32)
    net = maxpool2d_fn(input_tensor= net, p_size= 2, strides= 2)
    net = conv2d_fn(input_tensor= input_layer, k_size= 2, n_out= 64)
    net = maxpool2d_fn(input_tensor= net, p_size= 2, strides= 2)
    net = tf.layers.flatten(inputs= net)
    net = tf.layers.dense(inputs= net, units= 1024)
    net = tf.layers.dropout(inputs= net, \
                            rate = hyper_params['drop_out'], \
                            training= mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs= net, units= model_params['n_classes'])

    prediction = {
        'clasess': tf.argmax(input= logits, axis=1), \
        'probabilities': tf.nn.softmax(logits, name= 'softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = prediction)
    
    labels = tf.cast(labels, tf.int32)
    loss = tf.losses.sparse_softmax_cross_entropy(labels= labels, logits= logits)
    tf.summary.scalar('loss', loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate= hyper_params['learning_rate'])
        train_op = optimizer.minimize(loss = loss, global_step= tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels= labels, predictions= prediction['clasess'])
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)