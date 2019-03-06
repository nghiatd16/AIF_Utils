from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import random
import sys
import threading
from multiprocessing import Pool, Value

import numpy as np
import tensorflow as tf
import math
import json
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

FLAGS = None
counter = None

def _observe_data(path):
    assert os.path.isdir(path), ("No such directory")
    cwd = os.getcwd()
    os.chdir(path)
    num_files = 0
    sub_directories = sorted(os.listdir())
    labels = []
    for subdir in sub_directories:
        sub_dir_path = os.path.join(path, subdir)
        num_files += len(os.listdir(sub_dir_path))
        labels.append(os.path.basename(sub_dir_path))
    os.chdir(cwd)
    return num_files, labels

def _check_compatible_worker(train_shards, validation_shards, worker):
    if worker is None:
        return False
    if train_shards is None:
        return False
    if validation_shards is None:
        return False
    if train_shards % worker != 0:
        return False
    if validation_shards % worker != 0:
        return False
    return True


def init_flags(train_directory, validation_directory, output_directory, train_shards, validation_shards, labels_file=None, worker=None):
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
        'sound/label': _bytes_feature(tf.compat.as_bytes(str(label)))
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

def _find_dataset(data_dir, labels_file):
    """Build a list of all images files and labels in the data set.
    Args:
        data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.
            data_dir/dog/another-image.JPEG
            data_dir/dog/my-image.jpg
        where 'dog' is the label associated with these images.
        labels_file: string, path to the labels file.
        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
            dog
            cat
            flower
        where each line corresponds to a label. We map each label contained in
        the file to an integer starting with the integer 0 corresponding to the
        label contained in the first line.
    Returns:
        filenames: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.GFile(
        labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []
    # Leave label index 0 empty as a background class.
    label_index = 0

    # Construct the list of WAV files and labels.
    for text in unique_labels:
        file_path = '%s/%s/*.wav' % (data_dir, text)
        matching_files = tf.gfile.Glob(file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        # if not label_index % 100:
        #     print('Finished finding files in %d of %d classes.' % (label_index, len(labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d WAV files across %d labels inside %s.' %
            (len(filenames), len(unique_labels), data_dir))

    return filenames, texts, labels

def _create_TFRecord_miniprocess(args):
    global counter
    state, filenames, labels, num_shards, shard_idx, file_begin, file_end = args
    output_filename = '%s-%.5d-of-%.5d' % (state, shard_idx, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(file_begin, file_end):
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
    with counter.get_lock():
        counter.value += 1
    print('%s: (%.5d/%.5d) Write %d file to shard %s' %
                        (datetime.now(), counter.value, num_shards, file_end-file_begin, output_filename))
    sys.stdout.flush()

def _create_TFRecord(state, filenames, texts, labels, num_shards):# Launch a thread for each batch.
    global counter
    print('%s: Create TFRecord from %d files, write into %d shards, using %d threads' % (state, len(filenames), num_shards, FLAGS.num_threads))
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    thread_args = []
    spacing = np.linspace(0, len(filenames), num_shards + 1).astype(np.int)
    counter = Value('i', 0)
    for i in range(num_shards):
        thread_args.append((state, filenames, labels, num_shards, i+1, spacing[i], spacing[i + 1]))
    p = Pool(FLAGS.num_threads)
    p.map(_create_TFRecord_miniprocess, thread_args)
    sys.stdout.flush()

def Folders_to_TFRecords(train_directory, validation_directory, output_directory, labels_file, train_shards, validation_shards, worker=None):
    global FLAGS
    FLAGS = init_flags(train_directory, validation_directory, output_directory, train_shards, validation_shards, labels_file, worker)
    filenames_train, texts_train, labels_train = _find_dataset(train_directory, FLAGS.labels_file)
    filenames_valid, texts_valid, labels_valid = _find_dataset(validation_directory, FLAGS.labels_file)
    _create_TFRecord('train', filenames_train, texts_train, labels_train, FLAGS.train_shards)
    _create_TFRecord('valid', filenames_valid, texts_valid, labels_valid, FLAGS.validation_shards)

def TFRecords_to_TFDataset(directory, format_file, process_fn=None, worker=None, buffer_size=None):
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

def TFDataset_to_InputFn(dataset, num_classes, batch_size, shuffle_size=1024, prefetch_size=32, repeat_times=None):
    def get_inputs(dataset, num_classes, batch_size, shuffle_size, prefetch_size, repeat):
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat(repeat_times)  # repeat indefinitely
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)
        wav_data, label = dataset.make_one_shot_iterator().get_next()  
        label = tf.one_hot(tf.dtypes.cast(tf.string_to_number(label), tf.int32), num_classes)
        return wav_data, label
    return lambda:get_inputs(dataset, num_classes, batch_size, shuffle_size, prefetch_size, repeat_times)

def TFRecords_to_InputFn(directory, format_file, num_classes, process_fn=None, worker=1, 
                            batch_size=4 ,buffer_size=None, shuffle_size=1024, repeat_times=1):
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

    def get_inputs(dataset, num_classes):
        wav_data, label = dataset.make_one_shot_iterator().get_next()  
        label = tf.one_hot(tf.dtypes.cast(tf.string_to_number(label), tf.int32), num_classes)
        return wav_data, label
    if buffer_size is None:
        buffer_size=batch_size*worker
    file_names = tf.data.Dataset.list_files(os.path.join(directory, format_file))
    dataset = tf.data.TFRecordDataset(file_names, buffer_size=buffer_size, num_parallel_reads=worker*batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat(repeat_times)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_decode_example, 
        batch_size=batch_size,
        num_parallel_batches=worker))
    dataset = dataset.prefetch(batch_size)
    return lambda:get_inputs(dataset, num_classes)
