import os
import sys
import argparse
# from zipfile import ZipFile
import pickle
import numpy as np
from tqdm import tqdm
import openslide
from tools import get_files, get_name, gen_logger

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

def divide_prepare(slide_path, width:int, logger):
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

def divide_certain(slide_path: str, out_dir: str, logger, width=96) -> None:
    """The origin slide is too large, the function can segment the large one into small tiles.
    In this project, we set the height equals to the width. It aims to tile images in 10X power using
    less resource as possibel"""
    slide, level, base10, dimension_ref, d_step, size = divide_prepare(slide_path, width, logger)
    # begin segment tiles
    cwp = os.getcwd()
    case_name = get_name(slide_path)
    out_path = os.path.join(cwp, out_dir, case_name)
    os.makedirs(out_path, exist_ok=True)

    # set start points of tiles
    height = width
    x_bound = dimension_ref[0] - width
    y_bound = dimension_ref[1] - height
    for x in tqdm(range(0, x_bound, d_step)):
        for y in range(0, y_bound, d_step):
            loc = (x, y)
            # print(loc, level, size)
            small_image = slide.read_region(location=loc, level=level, size=size)
            if is_useless(small_image):
                continue
            if not base10:
                small_image = small_image.resize((width, height))
            fp = os.path.join(out_path, '{}-{}-{}-{}.tiff'.format(x, y, width, height))
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
    # begin segment tiles
    case_name = get_name(slide_path)
    # print(case_name)
    out_path = os.path.join(out_dir, case_name)
    os.makedirs(out_path, exist_ok=True)
    # calculate individual size
    height_rel = width_rel
    width = int(tile * ratio)
    height = int(width)
    size = (width, height)
    x_bound = dimension_ref[0] - width
    y_bound = dimension_ref[1] - height
    for x in tqdm(range(0, x_bound, tile)):
        for y in (range(0, y_bound, tile)):
            # locate start point
            loc = (x, y)
            # get the small image
            small_image = large_image.read_region(location=loc, level=level, size=size)
            # filter the useless image
            if is_useless(small_image):
                continue
            # save the small image
            resize_image = small_image.resize((width_rel, height_rel))
            fp = os.path.join(out_path, '{}-{}-{}-{}.tiff'.format(x, y, width, height))
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

def batch_tiling(path, out_dir, logger):
    filter_func = None
    # filter_func = lambda x:get_name(x) not in cache
    slides = list(filter(filter_func, get_files(path)))

    for slide_path in slides:
        name = get_name(slide_path)
        try:
            logger.info(f'start {name}')
            divide(slide_path, out_dir)
        except:
            logger.exception(f'{name} encountered error in batch')

if __name__ == '__main__':
    logger = gen_logger('tile')
    parse = argparse.ArgumentParser(description='A patch to separate large .svs file into 10X 96*96 .tif files.')
    parse.add_argument('i')
    parse.add_argument('o')
    command = parse.parse_args()
    batch_tiling(command.i, command.o, logger)
