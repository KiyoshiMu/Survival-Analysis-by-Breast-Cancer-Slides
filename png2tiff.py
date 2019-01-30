from tqdm import tqdm
import cv2
import argparse
import os
from tools import get_name, gen_logger, get_files

def convert(img_p: str, out_path: str) -> None:
    img = cv2.imread(img_p)
    name = get_name(img_p)
    cv2.imwrite(os.path.join(out_path, f'{name}.tif'), img)

def convert_dir(dir_p, out_path):
    for img_p in get_files(dir_p, suffix='png'):
        convert(img_p, out_path)

def batch_convert(png_dir: str, out_dir: str) -> None:
    for dir_n in tqdm(os.listdir(png_dir)):
        dir_p = os.path.join(png_dir,dir_n)
        out_path = os.path.join(out_dir, dir_n)
        os.makedirs(out_path, exist_ok=True)
        convert_dir(dir_p, out_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='A patch to convert .png files to .tif files.')
    parse.add_argument('-i', required=True)
    parse.add_argument('-o', required=True)
    command = parse.parse_args()
    batch_convert(command.i, command.o)