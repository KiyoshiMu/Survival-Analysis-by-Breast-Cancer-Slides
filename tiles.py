import openslide
import os
import sys
import numpy as np
from tqdm import tqdm
from tools import get_files, get_name, gen_logger
import argparse
# from zipfile import ZipFile
import pickle

def base_10x(properties:dict) ->int and bool:
    ori_mag = int(properties.get('openslide.objective-power', 0))
    level_count = int(properties.get('openslide.level-count', 0))
    assert ori_mag != 0 and level_count != 0
    for level in range(1, level_count, 1):
        ratio = round(float(properties.get(f'openslide.level[{str(level)}].downsample', 0)))
        assert ratio != 0
        if ori_mag / ratio == 10:
            return level, True
        elif ori_mag / ratio < 10:
            return level-1, False

def divide_prepare(slide_path, width:int):
    slide = openslide.OpenSlide(slide_path)
    properties = slide.properties
    try:
        level, base10 = base_10x(properties)
    except AssertionError:
        logger.exception(f'{get_name(slide_path)} magnitude power information loses')
        return

    dimensions = slide.level_dimensions
    dimension_ref = dimensions[0]
    ratio = round(float(properties.get(f'openslide.level[{str(level)}].downsample')))
    ori_mag = int(properties.get('openslide.objective-power'))
    
    if not base10:
        width = width * (ori_mag / ratio / 10)

    width = int(width)
    d_step = width * ratio
    size = (width, width)

    return slide, level, base10, dimension_ref, d_step, size

def divide_certain(slide_path: str, out_dir: str, width=96) -> None:
    """The origin slide is too large, the function can segment the large one into small tiles.
    In this project, we set the height equals to the width. It aims to tile images in 10X power using
    less resource as possibel"""
    slide, level, base10, dimension_ref, d_step, size = divide_prepare(slide_path, width)
    # begin segment tiles
    cwp = os.getcwd()
    case_name = get_name(slide_path)
    out_path = os.path.join(cwp, out_dir, case_name)
    os.makedirs(out_path, exist_ok=True)

    # set start points of tiles
    for i, x in enumerate(range(0, dimension_ref[0], d_step)):
        for j, y in enumerate(range(0, dimension_ref[1], d_step)):
            loc = (x, y)
            # print(loc, level, size)
            small_image = slide.read_region(location=loc, level=level, size=size)
            if is_useless(small_image):
                continue
            if not base10:
                small_image = small_image.resize((width, width))
            fp = os.path.join(out_path, '{:010d}{:010d}.tiff'.format(j, i))
            small_image.save(fp)
        
def divide(slide_path: str, out_dir: str, level=0, width_rel=96, mag=10) -> None:
    """The origin slide is too large, the function can segment the large one into small tiles.
    In this project, we set the height equals to the width.
    Slide_path: the path of the target slide; 
    level: varying definition of images, 0 is the largest, int;
    width_rel: the width and the length of output images, int.
    mag: the magnitude or the object power, float"""
    # Read slide. level 0 is mag X40 or X20.
    # here we downsize ((level 0 's mag / mag)**2) times  
    large_image = openslide.OpenSlide(slide_path)
    ori_mag = int(large_image.properties.get('openslide.objective-power'))
    time = ori_mag / mag
    tile = int(width_rel * time)
    # get reference and target location, use a reference level instead of the maxiumn power level may make reduce the cose of resize
    dimensions = large_image.level_dimensions
    dimension_ref = dimensions[0]
    dimension = dimensions[level]
    ratio = dimension[0] / dimension_ref[0]
    # set start points of tiles
    widths_point = list(range(0, dimension_ref[0], tile))
    heights_point = list(range(0, dimension_ref[1], tile))
    # begin segment tiles

    case_name = get_name(slide_path)
    # print(case_name)
    out_path = os.path.join(out_dir, case_name)
    os.makedirs(out_path, exist_ok=True)
    for i, x in enumerate(widths_point):
        for j, y in enumerate(heights_point):
            # locate start point
            loc = (x, y)
            # calculate individual size
            width, height = tile, tile
            if i == len(widths_point) - 1:
                width = dimension_ref[0] - x
            if j == len(heights_point) - 1:
                height = dimension_ref[1] - y
            size = (int(width * ratio), int(height * ratio))
            # get the small image
            small_image = large_image.read_region(location=loc, level=level, size=size)
            # filter the useless image
            if is_useless(small_image):
                continue
            # save the small image
            height_rel = width_rel
            resize_image = small_image.resize((width_rel, height_rel))
            fp = os.path.join(out_path, '{:010d}{:010d}.tiff'.format(j, i))
            resize_image.save(fp)
    
def is_useless(image) -> bool:
    """Help to judge whether the small image is informative.
    If a image has more information, it should be darker in gray mode.
    image: a Pillow object"""
    # # if the width different from the height, it's the marginal part.
    # if image.width != image.height:
    #     return True
    gray = image.convert("L")
    # 230 is a magic number, and it is not good. However, currently, I haven't found a better way
    # to select the informative images.
    return np.mean(gray) > 230

def batch_tiling(path, out_dir):
    filter_func = None
    # if os.path.isdir(out_dir):
    cache = os.listdir('c:\special')
    cache.extend(os.listdir('e:\special'))
    filter_func = lambda x:get_name(x) not in cache
    slides = list(filter(filter_func, get_files(path)))
    work_load = len(slides)

    pbar = tqdm(total=work_load)
    for slide_path in slides:
        try:
            divide(slide_path, out_dir)
        except:
            logger.exception(f'{get_name(slide_path)} encountered error in batch')
        pbar.update(1)
    pbar.close()

logger = gen_logger()
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='A patch to separate large .svs file into 10X 96*96 .tif files.')
    parse.add_argument('-i', required=True)
    parse.add_argument('-o', required=True)
    command = parse.parse_args()
    batch_tiling(command.i, command.o)
