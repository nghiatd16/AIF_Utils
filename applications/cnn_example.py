#Khai báo thư viện
import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from tf_records_loader import TFDataset_to_InputFn, TFRecords_to_TFDataset, Folders_to_TFRecords
#Khai báo tham số cho model
model_params = {
    'n_classes': 10,
    'input_shape': (50, 50, 1),
    'batch_size': 64
}

hyper_params = {
    'learning_rate': 0.001,
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
    # 1st block
    # chỉ đơn giản có một convolutional layer và 1 max pooling layer
    net = conv2d_fn(input_tensor= input_layer, k_size= 2, n_out = 32)
    net = maxpool2d_fn(input_tensor= net, p_size= 2, strides= 2)
    # 2nd block
    # tương tự như 1st block
    net = conv2d_fn(input_tensor= input_layer, k_size= 2, n_out= 64)
    net = maxpool2d_fn(input_tensor= net, p_size= 2, strides= 2)
    # đập phẳng feature map
    # sau 2 block bây giờ chúng ta thu được 1 feature map dạng 3D
    # mà logits layer của chúng ta thực chất là 1 mạng neural network cơ bản
    # do đó chúng ta phải đập phẳng feature map từ 3D về 2D
    net = tf.layers.flatten(inputs= net)
    # dense layer (tên gọi khác là fully connected layer) - là mạng neural network cơ bản
    net = tf.layers.dense(inputs= net, units= 1024)
    # drop out layer
    net = tf.layers.dropout(inputs= net, \
                            rate = hyper_params['drop_out'], \
                            training= mode == tf.estimator.ModeKeys.TRAIN)
    # logits layer
    # đây thực ra cũng là dense layer không có gì đặc biệt
    # nhưng vì nó là output layer nên mình đặt cái tên khác biệt 1 tí :v
    logits = tf.layers.dense(inputs= net, units= model_params['n_classes'])
    prediction = {
        'classes': tf.argmax(input= logits, axis=1), \
        'probabilities': logits
    }
    # nếu mode == PREDICT thì return lại EstimatorSpec theo mode PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = logits)
    
    # Tính toán loss của model
    labels = tf.cast(labels, tf.int32, name="labels_tensor")
    # sử dụng cross entropy để tính loss với output là ma trận được tính bởi softmax
    loss = tf.losses.softmax_cross_entropy(onehot_labels= labels, logits= prediction['probabilities'])
    # Ghi lại loss tại các checkpoint để tensorboard vẽ đồ thị loss
    tf.summary.scalar('loss', loss)
    
    # Cấu hình Train Optimizer để tiến hành train (chỉ dành cho mode == TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # chọn thuật toán để optimize loss
        optimizer = tf.train.AdamOptimizer(learning_rate= hyper_params['learning_rate'])
        # mục đích train là làm cho loss cực tiểu do đó chọn minimize
        train_op = optimizer.minimize(loss = loss, global_step= tf.train.get_global_step())
        # trả về estimator theo mode TRAIN
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    # Khởi tạo thước đánh giá model (evaluation metrics) cho EVAL mode, ở đây như thường lệ mình sử dụng accuracy làm thước đo
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels= tf.argmax(input= labels, axis=1), predictions= prediction['classes'])
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
Folders_to_TFRecords(train_directory="/home/nghiatd/workspace/dataset/digit/train", 
                    validation_directory="/home/nghiatd/workspace/dataset/digit/test", 
                    output_directory="/home/nghiatd/workspace/dataset/digit/digit_TFR")
train_dataset = TFRecords_to_TFDataset(directory="/home/nghiatd/workspace/dataset/digit/digit_TFR", format_file = "train-*")
train_input_fn = TFDataset_to_InputFn(train_dataset, (-1,50,50,1), 10, 32, repeat_times=None)
test_dataset = TFRecords_to_TFDataset(directory="/home/nghiatd/workspace/dataset/digit/digit_TFR", format_file = "validation-*")
test_input_fn = TFDataset_to_InputFn(dataset=test_dataset, shape=(-1,50,50,1), num_classes=10, batch_size=32, repeat_times=1)
classifier = tf.estimator.Estimator(
    model_fn = cnn_model_fn, \
    model_dir= 'CheckPoint')
classifier.train(input_fn = train_input_fn,steps= 2000)
classifier.evaluate(input_fn=test_input_fn)
