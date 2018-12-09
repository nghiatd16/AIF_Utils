import cv2
import os
import subprocess
import time
from multiprocessing import Pool
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
    cv2.imwrite(output_path, img)

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
    for label in dataset:
        if label == "root_path": continue
        lst_imgs = dataset[label]
        multi_processing_list(lst_imgs, process_fn, os.path.join(output_dir, label))
