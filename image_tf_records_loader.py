'''
An package that help you storage efficient TFRecord for image datasets,
and fit TFRecord object to your model easily\n
AIF_Utils \n
Author: nghiatd_16 \n
Improvement: quyendb_14 \n
Reference: https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
'''
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


# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.

flags = tf.app.flags
FLAGS = flags.FLAGS
counter = None
def _calculate_directory_size(path):
    assert os.path.isdir(path), ("No such a directory")
    cwd = os.getcwd()
    os.chdir(path)
    directory_size = sum([os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f)])
    os.chdir(cwd)
    return directory_size

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

def _write_list_string_to_file(path, lst):
    f = open(path, "w")
    for text in lst:
        f.write("{}\n".format(text))
    f.close()

def init_flags(train_directory=None, validation_directory=None, output_directory=None, labels_file=None, train_shards=None, validation_shards=None, worker=None):
    global FLAGS
    if train_directory is None and validation_directory is None:
        Warning("Both train_directory and validation_directory are None")
        return
    if validation_directory is None:
        train_num_files, train_labels = _observe_data(train_directory)
        num_classes = len(train_labels)
        if train_shards is None:
            train_shards = int(min(math.ceil(train_num_files/64), 64))
    if train_directory is None:
        validation_num_files, validation_labels = _observe_data(validation_directory)
        num_classes = len(validation_labels)
        if validation_shards is None:
            validation_shards = int(min(math.ceil(validation_num_files/32), 32))
        
    if labels_file is None:
        if train_directory is None:
            labels_file = "{}/labels_file.txt".format(os.path.join(validation_directory, os.pardir))
            _write_list_string_to_file(labels_file, validation_labels)
        else:
            labels_file = "{}/labels_file.txt".format(os.path.join(train_directory, os.pardir))
            _write_list_string_to_file(labels_file, train_labels)
    if worker is None:
        worker = os.cpu_count()
    if train_directory is not None:
        flags.DEFINE_string('train_directory', train_directory,
                            'Training data directory')
        flags.DEFINE_integer('train_shards', train_shards,
                            'Number of shards in training TFRecord files.')
    if validation_directory is not None:
        flags.DEFINE_string('validation_directory', validation_directory,
                                'Validation data directory')
        flags.DEFINE_integer('validation_shards', validation_shards,
                            'Number of shards in validation TFRecord files.')
    flags.DEFINE_string('output_directory', output_directory,
                            'Output data directory')

    flags.DEFINE_integer('num_threads', worker,
                            'Number of threads to preprocess the images.')
    flags.DEFINE_string('labels_file', labels_file, 'Labels file')
    flags.DEFINE_integer('num_classes', num_classes, 'Number of classes')

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_image_to_example(filename, image_buffer, label, text):
    """Build an Example proto for an example.
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
    Returns:
        Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
        }))
    return example

def _process_image(filename, coder):
    """Process a single image file.
    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
    """
    # Read the image file.
    with tf.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    return image_data

def _create_TFRecord_miniprocess(args):
    
    state, filenames, texts, labels, num_shards, shard_idx, file_begin, file_end, output_directory = args
    output_filename = '%s-%.5d-of-%.5d' % (state, shard_idx, num_shards)
    output_file = os.path.join(output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    for i in range(file_begin, file_end):
        filename = filenames[i]
        label = labels[i]
        text = texts[i]
        try:
            with tf.gfile.GFile(filename, 'rb') as f:
                image_buffer = f.read()
        except Exception as e:
            print(e)
            print('SKIPPED: Unexpected error while reading %s.' % filename)
            continue
        example = _convert_image_to_example(filename, image_buffer, label, text)
        writer.write(example.SerializeToString())
    if 'win' not in sys.platform:
        global counter
        with counter.get_lock():
            counter.value += 1
        
        print('%s: (%.5d/%.5d) Write %d file to shard %s' %
                            (datetime.now(), counter.value, num_shards, file_end-file_begin, output_filename))
        sys.stdout.flush()

def _create_TFRecord(state, filenames, texts, labels, num_shards):
    global counter, FLAGS
    """Process and save list of images as TFRecord of Example protos.
    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    spacing = np.linspace(0, len(filenames), num_shards + 1).astype(np.int)

    # Create a generic TensorFlow-based utility for converting all image codings.

    thread_args = []
    for i in range(num_shards):
        args = (state, filenames, texts, labels, num_shards, i+1, spacing[i], spacing[i + 1], FLAGS.output_directory)
        thread_args.append(args)
    counter = Value('i', 0)
    # Wait for all the threads to terminate.
    p = Pool(FLAGS.num_threads)
    p.map(_create_TFRecord_miniprocess, thread_args)
    # print('%s: Finished writing all %d images in data set.' %
    #         (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir, labels_file):
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

    # Construct the list of JPEG files and labels.
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(labels)))
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

    print('Found %d JPEG files across %d labels inside %s.' %
            (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels

def _process_dataset(name, directory, num_shards, labels_file):
    """Process a complete data set and save it as a TFRecord.
    Args:
        name: string, unique identifier specifying the data set.
        directory: string, root path to the data set.
        num_shards: integer number of shards for this data set.
        labels_file: string, path to the labels file.
    """
    filenames, texts, labels = _find_image_files(directory, labels_file)
    mapping = {}
    for i in range(len(texts)):
        mapping[labels[i]] = texts[i]
    if name == 'train':
        out_path = os.path.join(FLAGS.output_directory, "_mapping.json")
        fo = open(out_path, "w")
        json.dump(mapping, fo)
        fo.close()
    _create_TFRecord(name, filenames, texts, labels, num_shards)

def Folders_to_TFRecords(train_directory=None, validation_directory=None, output_directory=None, labels_file=None, train_shards=None, validation_shards=None, worker=None):
    if train_directory is None and validation_directory is None:
        Warning("Both train_directory and validation_directory are None")
        return
    if output_directory is None:
        time_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        if validation_directory is None:
            output_directory = os.path.join(os.path.dirname(os.path.abspath(train_directory)), os.path.basename(train_directory)+"_TFRecords_{}".format(time_str))
        else:
            output_directory = os.path.join(os.path.dirname(os.path.abspath(validation_directory)), os.path.basename(validation_directory)+"_TFRecords_{}".format(time_str))
    try:
        os.makedirs(output_directory)
    except Exception as e:
        print(e)
        exit(1)
    init_flags(train_directory, validation_directory, output_directory, labels_file, train_shards, validation_shards, worker)
    if validation_directory is not None:
        _process_dataset('validation', FLAGS.validation_directory,
                        FLAGS.validation_shards, FLAGS.labels_file)
    if train_directory is not None:
        _process_dataset('train', FLAGS.train_directory,
                        FLAGS.train_shards, FLAGS.labels_file)

def _carrier(image, process_fn, Tout=tf.float32):
    return tf.py_function(process_fn, image, [Tout])

def TFRecords_to_TFDataset(directory, filename_pattern, worker=None, buffer_size=2048, process_fn=None, Tout=tf.float32):
    '''
        Args: 
            filename_pattern : one of ['train-*', 'validation-*'] - choose loading dataset
    '''
    def parse_fn(example):
        "Parse TFExample records and perform simple data augmentation."
        feature={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(example, feature)
        image = tf.image.decode_image(parsed["image/encoded"])
        label = parsed["image/class/label"]
        if process_fn is not None:
            image = _carrier([image], process_fn, Tout)[0]  # augments image using slice, reshape, resize_bilinear
        return image, label
    if worker is None:
        worker = os.cpu_count()
    file_names = tf.data.Dataset.list_files(os.path.join(directory, filename_pattern))
    dataset = tf.data.TFRecordDataset(file_names, buffer_size=buffer_size, num_parallel_reads=worker).prefetch(tf.contrib.data.AUTOTUNE).map(parse_fn)
    return dataset

def TFDataset_to_InputFn(dataset, shape, num_classes, batch_size, shuffle_size=1024, prefetch_size=32, repeat_times=1):
    def get_inputs(dataset, shape, num_classes, batch_size, shuffle_size, prefetch_size, repeat):
        # global dataset
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat(repeat_times)  # repeat indefinitely
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)
        image, label = dataset.make_one_shot_iterator().get_next()
        image = tf.reshape(image, shape)
        label = tf.one_hot(label, num_classes)
        # print(features)
        # print(label)
        return image, label
    return lambda:get_inputs(dataset,shape, num_classes, batch_size, shuffle_size, prefetch_size, repeat_times)

def create_ImageGenerator(TFIterator, process_fn=None, expected_Tout=None):
    list_Tout = expected_Tout
    while True:
        try:
            next_batch = TFIterator.get_next()
            if process_fn is not None:
                if list_Tout is None:
                    list_Tout = []
                    for ele in next_batch:
                        list_Tout.append(ele.dtype)
                next_batch = tf.py_function(process_fn, next_batch, list_Tout)
            yield next_batch
        except Exception as e:
            raise e

if __name__ == "__main__":
    location = 'E:\\Workspace\\AIF\\AIF_Challenge\\test_tf_records_loader\\init_train'
    output = 'E:\\Workspace\AIF\\AIF_Challenge\\test_tf_records_loader\\train_tf_record'
    Folders_to_TFRecords(location, location, output)