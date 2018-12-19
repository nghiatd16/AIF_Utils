import numpy as np
import cv2
import tensorflow as tf
import os
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _decode_data(serialized_example):
    features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    })
    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # channel = tf.cast(features['channel'], tf.int32)
    # print(sess.run(channel))
    # exit(-1)
    image = tf.image.decode_image(features['img'])
    label = tf.cast(features['label'], tf.string)
    return image, label
directory = "train"
file_names = tf.data.Dataset.list_files(os.path.join(directory, "train-*"))
dataset = tf.data.TFRecordDataset(file_names).map(_decode_data).batch(2).repeat().make_one_shot_iterator()
# print(dataset)
# print(files)
#session
config = tf.ConfigProto(device_count = {"GPU":0})
sess = tf.Session(config=config)
# print(sess.run(files))
# # np_img = sess.run(img)
# # tfr_file_names = 'test.tfrecord'
# file_names = ['test.tfrecord','test2.tfrecord']
# dataset = tf.data.TFRecordDataset(file_names).map(_decode_data).batch(2).repeat().make_one_shot_iterator()
sample = dataset.get_next()
img, label = sess.run(sample)
# img = img[0]
label = np.array(list(map(lambda x: x.decode('utf-8'), label)))
print(label)
# print(img)
# print(type(img))
# cv2.imshow("tmp", img)
# cv2.waitKey(0)
# sample = dataset.get_next()
# img, label = sess.run(sample)
# img = img[0]
# # print(img)
# # print(type(img))
# cv2.imshow("tmp", img)
# cv2.waitKey(0)
# label = label[0].decode('utf-8')
# print(img)
# print(type(img))
# print(label)
# print(type(label))
# writer = tf.python_io.TFRecordWriter(tfr_file_names)
# # original_img = cv2.imread("test2.jpg")
# label = "test"
# # h,w,c = original_img.shape
# # img_string = original_img.tostring()


# filename = "test.jpg"
# with tf.gfile.GFile(filename, 'rb') as f:
#     image_data = f.read()
# img = tf.image.decode_image(image_data)
# img = sess.run(img)
# h, w, c = img.shape
# feature = {
#     'width': _int64_feature(w),
#     'height': _int64_feature(h),
#     'channel': _int64_feature(c),
#     'img': _bytes_feature(image_data),
#     'label': _bytes_feature(tf.compat.as_bytes(label))
# }
# example = tf.train.Example(features=tf.train.Features(feature=feature))
# writer.write(example.SerializeToString())
# writer.close()
# img = tf.image.decode_image(image_data, channels=3)

# cv2.imshow("tmp",np_img)
# cv2.waitKey(0)