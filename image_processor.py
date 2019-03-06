'''
An package that help you easily process image datasets efficiently and quickly with multiprocessing\n
AIF_Utils \n
Author: nghiatd_16 \n
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import os
import subprocess
import time
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import cpuinfo
from skimage import exposure
from skimage import data, img_as_float
import warnings
import numpy as np
import logging
logging.basicConfig(format='[%(levelname)s|%(asctime)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
level=logging.DEBUG)
def find_images(path):
    path = os.path.abspath(path)
    assert os.path.isdir(path), ("No such a directory")
    cwd = os.getcwd()
    os.chdir(path)
    lst_imgs = []
    for fil in os.listdir():
        if not (fil.endswith(".png") or fil.endswith(".jpg") or fil.endswith(".jpeg")):
            continue
        lst_imgs.append(os.path.join(path, fil))
    os.chdir(cwd)
    return lst_imgs

def discover_image_dataset(path):
    path = os.path.abspath(path)
    assert os.path.isdir(path), ("No such a directory")
    cwd = os.getcwd()
    os.chdir(path)
    dir_name = path
    dataset = {}
    dataset['root_path'] = dir_name
    for folder in os.listdir():
        subdir_path = os.path.join(dir_name, folder)
        dataset[os.path.basename(subdir_path)] = find_images(subdir_path)
    os.chdir(cwd)
    return dataset

def _multi_run_wrapper(args):
    try:
        _execute_image_profn(*args)
    except ChildProcessError as e:
        print(e)
        exit(-1)

def _execute_cmd(command):
    subprocess.call(command, shell=True)

def _execute_image_profn(process_fn, input_path, output_path):
    img = cv2.imread(input_path)
    img = process_fn(img)
    if img is not None:
        cv2.imwrite(output_path, img)
    del img

def resize_dataset(lst_imgs, desired_size, output_dir=None, worker=None):
    if worker is None:
        worker = os.cpu_count()
    if output_dir is None:
        tmp_path = lst_imgs[0]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(tmp_path)), "{}_OutFolder".format(time.time()))
        os.makedirs(output_dir)
    w, h = desired_size
    total = len(lst_imgs)
    lst_cmd = []
    for i in range(total):
        name = os.path.basename(lst_imgs[i]) + "_{}.jpg".format(i)
        out_path = os.path.join(output_dir, name)
        cmd = "python _resize_img.py -in {} -out {} -size {}x{}".format(lst_imgs[i], out_path, w, h)
        lst_cmd.append(cmd)
    p = Pool(worker)
    p.map(_execute_cmd, lst_cmd)

def rescale_dataset(lst_imgs, desired_ratio, output_dir=None, worker=None):
    if worker is None:
        worker = os.cpu_count()
    if output_dir is None:
        tmp_path = lst_imgs[0]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(tmp_path)), "{}_OutFolder".format(time.time()))
        os.makedirs(output_dir)
    total = len(lst_imgs)
    lst_cmd = []
    for i in range(total):
        name = os.path.basename(lst_imgs[i]) + "_{}.jpg".format(i)
        out_path = os.path.join(output_dir, name)
        cmd = "python _rescale_img.py -in {} -out {} -ratio {}".format(lst_imgs[i], out_path, desired_ratio)
        lst_cmd.append(cmd)
    p = Pool(worker)
    p.map(_execute_cmd, lst_cmd)

def multi_processing_list(lst_imgs, process_fn, output_dir=None, worker=None):
    if worker is None:
        worker = os.cpu_count()
    if output_dir is None:
        tmp_path = lst_imgs[0]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(tmp_path)), "{}_OutFolder".format(time.time()))
    os.makedirs(output_dir)
    total = len(lst_imgs)
    lst_args = []
    print("Num of parallel workers: {}".format(worker))
    for i in range(total):
        name = os.path.basename(lst_imgs[i]) + "_{}.jpg".format(i)
        out_path = os.path.join(output_dir, name)
        lst_args.append((process_fn, lst_imgs[i], out_path))
    p = Pool(worker)
    p.map(_multi_run_wrapper, lst_args)

def multi_processing_dataset(dataset, process_fn, output_dir=None, worker=None):
    if worker is None:
        worker = os.cpu_count()
    if output_dir is None:
        dir_name = dataset['root_path']
        output_dir = os.path.join(os.path.dirname(dir_name), "{}_OutFolder".format(time.time()))
        os.makedirs(output_dir)
    print("Num of parallel workers: {}".format(worker))
    lst_args = []
    for label in dataset:
        if label == "root_path": continue
        lst_imgs = dataset[label]
        
        class_path = os.path.join(output_dir, label)
        os.makedirs(class_path)
        total = len(lst_imgs)
        
        for i in range(total):
            name = os.path.basename(lst_imgs[i]) + "_{}.jpg".format(i)
            out_path = os.path.join(class_path, name)
            lst_args.append((process_fn, lst_imgs[i], out_path))
    p = Pool(worker)
    p.map(_multi_run_wrapper, lst_args)
def __deffn(img):
    return img
def show_cpu_info():
    logging.info("Getting processor info")
    cpu_info = cpuinfo.get_cpu_info()
    brand = cpu_info['brand']
    count = cpu_info['count']
    logging.info("Processor Info:\nBrand: {}\nNum cores: {}\n".format(brand, count))
def split_train_test_dataset(dataset, test_size, output_dir=None, worker=None):
    
    if output_dir is None:
        dir_name = dataset['root_path']
        output_dir = os.path.join(os.path.dirname(dir_name), "{}_OutFolder".format(time.time()))
        os.makedirs(output_dir)
    for label in dataset:
        if label == 'root_path':
            continue
        img_paths = dataset[label]
        img_labels = [label]*len(img_paths)
        X_train, X_test, y_train, y_test = train_test_split(img_paths, img_labels, test_size=test_size)
        # print(X_test[0])
        # break
        lst_train_output_dir = os.path.join(output_dir, os.path.dirname(os.path.dirname(label)), "{}_train".format(label))
        print(lst_train_output_dir)
        multi_processing_list(X_train, __deffn, lst_train_output_dir, worker)
        lst_test_output_dir = os.path.join(output_dir, os.path.dirname(os.path.dirname(label)), "{}_test".format(label))
        print(lst_test_output_dir)
        multi_processing_list(X_test, __deffn, lst_test_output_dir, worker)

def histogram_expose_img(img):
    img = cv2.resize(img, (40, 40))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_eq = exposure.equalize_hist(img)

        # Adaptive Equalization
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    show_img = img_adapteq*255
    show_img = show_img.astype(np.uint8)
    return show_img

if __name__ == "__main__":
    location = 'E:\\Workspace\AIF\\AIF_Challenge\\TrafficSignClassification\\data\\public_test'
    destination = 'E:\\Workspace\\AIF\\AIF_Challenge\\TrafficSignClassification\\data\\processed_test'
    show_cpu_info()
    images = find_images(location)
    multi_processing_list(images, output_dir=destination, process_fn=histogram_img)