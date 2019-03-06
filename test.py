from image_processor import discover_image_dataset
from image_processor import histogram_expose_img, multi_processing_dataset
from image_tf_records_loader import Folders_to_TFRecords, TFRecords_to_TFDataset
import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings
from skimage import exposure
import cv2
tf.enable_eager_execution()

def preprocess_data(image):
    image = image.numpy()
    # image = cv2.resize(image, (40, 40))
    image = keras.preprocessing.image.random_shift(image, 0.2, 0.2)
    image = keras.preprocessing.image.random_rotation(image, 20)
    return image

def model():
    net = keras.models.Sequential()
    # net.add(keras.layers.Lambda(augment_2d,
    #                 input_shape=(64,64,3),
    #                 arguments={'rotation': 8.0, 'horizontal_flip': True}))
    net.add(keras.layers.Conv2D(32, (3, 3), input_shape=IMAGE_SHAPE, padding='same'))
    net.add(keras.layers.Activation('relu'))
    net.add(keras.layers.Conv2D(32, (3, 3)))
    net.add(keras.layers.Activation('relu'))
    net.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.25))

    net.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    net.add(keras.layers.Activation('relu'))
    net.add(keras.layers.Conv2D(64, (3, 3)))
    net.add(keras.layers.Activation('relu'))
    net.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.25))

    net.add(keras.layers.Flatten())
    net.add(keras.layers.Dense(units=512))
    net.add(keras.layers.Activation('relu'))
    net.add(keras.layers.Dropout(0.5))
    net.add(keras.layers.Dense(units=N_CLASSES))
    net.add(keras.layers.Activation('softmax'))
    return net

BATCH_SIZE = 32
N_CLASSES = 43
LR = 0.001
N_EPOCHS = 30
N_UNITS = 128
IMAGE_SHAPE = (40,40,3)
import json
if __name__ == "__main__":
    
    sample_data_path = 'E:\\Workspace\\AIF\AIF_Challenge\\test_tf_records_loader\\init_train'
    processed_sample_path = 'E:\\Workspace\\AIF\AIF_Challenge\\test_tf_records_loader\\processed_train'
    
    TFRecord_path = 'E:\\Workspace\\AIF\AIF_Challenge\\test_tf_records_loader\\train_tfrecord'
    maping_file = 'E:\\Workspace\\AIF\\AIF_Challenge\\test_tf_records_loader\\train_tfrecord\\_mapping.json'
    # dataset = discover_image_dataset(sample_data_path)
    # multi_processing_dataset(dataset, process_fn = histogram_expose_img, output_dir=processed_sample_path)
    # Folders_to_TFRecords(train_directory=processed_sample_path, output_directory=TFRecord_path, train_shards=1)
    TFDataset = TFRecords_to_TFDataset(TFRecord_path, filename_pattern="train-*", process_fn=preprocess_data, worker=12)
    TFIterator = TFDataset.batch(1).repeat(N_EPOCHS).make_one_shot_iterator()
    # images, labels = TFIterator.get_next()
    mapping_label = json.load(open(maping_file, 'r'))
    while True:
        img, lb = TFIterator.get_next()
        img = img[0].numpy().astype(np.uint8)
        print(img)
        lb = int(lb[0])
        print(type(img))
        cv2.imshow("tmp", img)
        print(mapping_label[str(lb)])
        if cv2.waitKey(0) == ord('q'):
            break
    # my_model = keras.applications.DenseNet201(include_top=False, weights=None, input_shape=IMAGE_SHAPE, classes=N_CLASSES)
    # my_model.compile(loss=keras.losses.sparse_categorical_crossentropy,
    #           optimizer=keras.optimizers.Adadelta(),
    #           metrics=['accuracy'])
    # my_model.fit(TFIterator, epochs=10, steps_per_epoch=1300//BATCH_SIZE)
    # my_model.save("my_model.hdf5")