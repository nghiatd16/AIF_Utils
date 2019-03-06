import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
sys.path.append("..")
from sound_tf_records_loader import TFDataset_to_InputFn, TFRecords_to_TFDataset, Folders_to_TFRecords

tf.logging.set_verbosity(tf.logging.INFO)

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_width = feature_bin_count
    fingerprint_size = fingerprint_width * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'fingerprint_width': fingerprint_width,
        'fingerprint_size': fingerprint_size,
        'fft_length': 1024,
        'sample_rate': sample_rate,
        'n_classes': label_count
}

def extract_mfcc_fn(input_tensor, model_settings):
    stfts = tf.contrib.signal.stft(
            input_tensor, 
            frame_length=model_settings['window_size_samples'], 
            frame_step=model_settings['window_stride_samples'],
            fft_length=model_settings['fft_length'])
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = int(model_settings['fft_length']/2+1) 
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, model_settings['sample_rate'], lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :model_settings['fingerprint_width']]
    input_time_size = model_settings['spectrogram_length']
    input_frequency_size = model_settings['fingerprint_width']
    mfccs = tf.reshape(mfccs, [-1, input_time_size, input_frequency_size, 1])
    return mfccs

def conv2d_fn(input_tensor, k_size, n_out):
    return tf.layers.conv2d(inputs=input_tensor,
                            filters=n_out,
                            kernel_size=k_size,
                            activation=tf.nn.relu,
                            padding='same',
                            use_bias=True)

def maxpool2d_fn(input_tensor, p_size, strides):
    return tf.layers.max_pooling2d(inputs=input_tensor, pool_size=p_size, strides=strides)

def cnn_model_fn(features, labels, mode, params):
    model_settings = params['model_settings']
    hyper_params = params['hyper_params']
    
    input_layer = features
    net = extract_mfcc_fn(input_tensor=input_layer, model_settings=model_settings)

    net = conv2d_fn(input_tensor=net, k_size=[8,20], n_out=64)
    net = maxpool2d_fn(input_tensor=net, p_size=2, strides=2)

    net = conv2d_fn(input_tensor=net, k_size=[4,10], n_out=64)
    net = maxpool2d_fn(input_tensor=net, p_size=2, strides=2)

    net = tf.layers.flatten(inputs=net)
    net = tf.layers.dense(inputs=net, units=1024)
    net = tf.layers.dropout(inputs=net, rate=hyper_params['dropout_prob'], training= mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=net, units=model_settings['n_classes'])
    
    prediction = {
        'classes': tf.argmax(input= logits, axis=1),
        'probabilities': logits
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
    
    labels = tf.cast(labels, tf.int32, name="labels_tensor")
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=prediction['probabilities'])
    tf.summary.scalar('loss', loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=hyper_params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels= tf.argmax(input=labels, axis=1), predictions=prediction['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

WORKER = 2
N_CLASS = 2
BATCH_SIZE = 64
N_EPOCH = 5

SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 10
CLIP_DURATION_MS = 1000
MFCC_BIN = 40

model_settings = prepare_model_settings(
    label_count = N_CLASS, 
    sample_rate = SAMPLE_RATE, 
    clip_duration_ms = CLIP_DURATION_MS,
    window_size_ms = WINDOW_SIZE_MS, 
    window_stride_ms = WINDOW_STRIDE_MS, 
    feature_bin_count = MFCC_BIN
)

hyper_params = {
    'learning_rate': 0.001,
    'dropout_prob': 0.25
}

TRAIN_DATA = "/home/quyendb/workspace/dataset/keyword/train"
TEST_DATA = "/home/quyendb/workspace/dataset/keyword/test"
TFRECORD_DIR = "/home/quyendb/workspace/dataset/keyword/TFRecord"
LABELS_FILE = "/home/quyendb/workspace/dataset/keyword/labels.txt"

Folders_to_TFRecords(train_directory=TRAIN_DATA, 
                    validation_directory=TEST_DATA, 
                    output_directory=TFRECORD_DIR,
                    labels_file=LABELS_FILE,
                    train_shards=10,
                    validation_shards=2,
                    worker=WORKER)

def process_fn(wav_decoder):
    audio = tf.squeeze(wav_decoder.audio)
    return audio

train_dataset = TFRecords_to_TFDataset(
    directory=TFRECORD_DIR, 
    format_file = "train-*",
    process_fn=process_fn,
    worker=WORKER)
train_input_fn = TFDataset_to_InputFn(
    dataset=train_dataset, 
    num_classes=N_CLASS, 
    batch_size=BATCH_SIZE, 
    repeat_times=N_EPOCH)
test_dataset = TFRecords_to_TFDataset(
    directory=TFRECORD_DIR, 
    format_file = "validation-*",
    worker=WORKER)
test_input_fn = TFDataset_to_InputFn(
    dataset=test_dataset, 
    num_classes=N_CLASS, 
    batch_size=BATCH_SIZE, 
    repeat_times=1)

classifier = tf.estimator.Estimator(
    model_fn = cnn_model_fn,
    params = {
        'model_settings': model_settings,
        'hyper_params': hyper_params
    }, 
    model_dir= 'CheckPoint')

classifier.train(input_fn=train_input_fn)
classifier.evaluate(input_fn=test_input_fn)