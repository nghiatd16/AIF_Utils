from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
import math
import json
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

FLAGS = None

def _check_compatible_worker(train_shards, validation_shards, worker):
    if worker is None:
        return False
    if train_shards is None:
        return False
    if validation_shards is None:
        return False
    if not train_shards % worker:
        return False
    if not validation_shards % worker:
        return False
    return True


def init_flags(train_directory, validation_directory, output_directory, train_shards, validation_shards, worker=None):
    def gcd(x, y):
        while y != 0:
            (x, y) = (y, x % y)
        return x

    if not _check_compatible_worker(train_shards, validation_shards, worker):
        shards_gcd = gcd(train_shards, validation_shards)
        worker = gcd(shards_gcd, os.cpu_count())
    tf.app.flags.DEFINE_string('train_directory', train_directory,
                           'Training data directory')
    tf.app.flags.DEFINE_string('validation_directory', validation_directory,
                            'Validation data directory')
    tf.app.flags.DEFINE_string('output_directory', output_directory,
                            'Output data directory')

    tf.app.flags.DEFINE_integer('train_shards', train_shards,
                                'Number of shards in training TFRecord files.')
    tf.app.flags.DEFINE_integer('validation_shards', validation_shards,
                                'Number of shards in validation TFRecord files.')

    tf.app.flags.DEFINE_integer('num_threads', worker,
                            'Number of threads to preprocess the images.')
    tf.app.flags.DEFINE_string('labels_file', labels_file, 'Labels file')
    return tf.app.flags.FLAGS
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _encode_example(filename, buffer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'sound/buffer': _bytes_feature(tf.compat.as_bytes(buffer)),
        'sound/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'sound/label': _bytes_feature(tf.compat.as_bytes(label))
        }))
    return example

def _process_sound_files_batch(thread_index, ranges, name, filenames,
                               labels, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    
    shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            try:
                with tf.gfile.GFile(filename, 'rb') as f:
                    audio_binary = f.read()
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected error while reading %s.' % filename)
                continue
            example = _encode_example(filename, audio_binary, label)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d sounds in thread batch.' %
                        (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d sounds to %s' %
            (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d sounds to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_sound_files(name, filenames, labels, num_shards):
    """Process and save list of sounds as TFRecord of Example protos.
    Args:
        name: string, unique identifier specifying the data set ('validation' or 'train)
        filenames: list of strings; each string is a path to an image file
        labels: label for dataset
        num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_sound_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
            (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_dataset():
    raise NotImplementedError

def Folders_to_TFRecords(train_directory, validation_directory, output_directory, train_shards, validation_shards, worker=None):
    init_flags(train_directory, validation_directory, output_directory, train_shards, validation_shards,worker)
    filenames_train, labels_train = _find_dataset(train_directory)
    filenames_valid, labels_valid = _find_dataset(validation_directory)
    _process_sound_files('train', filenames_train, labels_train, FLAGS.train_shards)
    _process_sound_files('train', filenames_valid, labels_valid, FLAGS.validation_shards)

def TFRecords_to_TFDataset(directory, format_file, process_fn=None, worker=None, buffer_size=1024):
    def _decode_example(example):
        feature={
            'sound/buffer': tf.FixedLenFeature([], tf.string),
            'sound/filename': tf.FixedLenFeature([], tf.string),
            'sound/label': tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(example, feature)
        # write sound decoder
        sound_buffer = parsed["sound/buffer"]
        wav_data = contrib_audio.decode_wav(sound_buffer, desired_channels=1)
        if process_fn is not None:
            wav_data = process_fn(wav_data)
        label = parsed["sound/label"]
        return wav_data, label

    if worker is None:
        worker = os.cpu_count()
    file_names = tf.data.Dataset.list_files(os.path.join(directory, format_file))
    dataset = tf.data.TFRecordDataset(file_names, buffer_size=buffer_size, num_parallel_reads=worker).map(_decode_example)
    return dataset

def TFDataset_to_InputFn(dataset, shape, num_classes, batch_size, shuffle_size=1024, prefetch_size=32, repeat_times=None):
    def get_inputs(dataset, shape, num_classes, batch_size, shuffle_size, prefetch_size, repeat):
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat(repeat_times)  # repeat indefinitely
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)
        image, label = dataset.make_one_shot_iterator().get_next()
        image = tf.reshape(image, shape)
        label = tf.one_hot(label, num_classes)
        return image, label
    return lambda:get_inputs(dataset,shape, num_classes, batch_size, shuffle_size, prefetch_size, repeat_times)