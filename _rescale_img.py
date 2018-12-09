import cv2
import argparse
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument("-inp", help="-inp : Input file path", dest="inp", action="store")
parser.set_defaults(inp=None)
parser.add_argument("-out", help="-out : Out file path", dest="out", action="store")
parser.set_defaults(out=None)
parser.add_argument("-ratio", help="-ratio (height/width) : desired ratio of image", dest="ratio", action="store")
parser.set_defaults(ratio=None)

def main(args):
    inp = args.inp
    out = args.out
    ratio = float(args.ratio)
    assert os.path.isfile(inp), ("File Not Found")
    assert os.path.isdir(os.path.dirname(out)), ("No such a directory")
    img = cv2.imread(inp)
    height, width, channel = img.shape
    mean = int((height+width)/2)
    desired_height = int(mean*ratio)
    desired_width = int(mean/ratio)
    img = cv2.resize(img, (desired_width, desired_height))
    original_ratio = int(height/width)
    # print(img.shape)
    if original_ratio < ratio:
        padding_width = desired_height/original_ratio - desired_width
        left = int(padding_width/2)
        right = int(padding_width/2)
        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, (255, 255, 255))
    if original_ratio > ratio:
        padding_height = original_ratio * desired_width - desired_height
        top = int(padding_height/2)
        bot = int(padding_height/2)
        img = cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, (255, 255, 255))
    img = cv2.resize(img, (width, height))
    # print(img.shape)
    cv2.imwrite(out, img)
if __name__ == '__main__':
    main(parser.parse_args())