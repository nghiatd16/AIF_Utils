import cv2
import argparse
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument("-inp", help="-inp : Input file path", dest="inp", action="store")
parser.set_defaults(inp=None)
parser.add_argument("-out", help="-out : Out file path", dest="out", action="store")
parser.set_defaults(out=None)
parser.add_argument("-size", help="-size : desired size of image", dest="size", action="store")
parser.set_defaults(size=None)

def main(args):
    inp = args.inp
    out = args.out
    assert os.path.isfile(inp), ("File Not Found")
    assert os.path.isdir(os.path.dirname(out)), ("No such a directory")
    width, height = args.size.split("x")
    width = int(width)
    height = int(height)
    img = cv2.imread(inp)
    img = cv2.resize(img, (width,height))
    cv2.imwrite(out, img)

if __name__ == '__main__':
    main(parser.parse_args())