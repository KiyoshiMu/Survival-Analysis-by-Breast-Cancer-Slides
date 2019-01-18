import openslide
import os
import sys
import numpy as np
import logging

def read_files(path: str) -> list:
    """From a dir read the .svs files, then we can divide the large slide into small ones
    path: the path of all .svs files."""
    # for f in os.listdir(path):
    #     if f[-4:] == '.svs':
    #         yield os.path.join(path, f)
    return [os.path.join(path, i) for i in os.listdir(path) if i[-4:] == '.svs']

def base_10x(properties:dict) ->int and bool:
    
    ori_mag = int(properties.get('openslide.objective-power', 0))
    level_count = int(properties.get('openslide.level-count', 0))
    assert ori_mag != 0 and level_count != 0
    for level in range(1, level_count, 1):
        ratio = int(properties.get(f'openslide.level[{str(level)}].downsample', 0))
        assert ratio != 0
        if ori_mag / ratio == 10:
            return level, True
        elif ori_mag / ratio < 10:
            return level-1, False

def get_name(slide_path):
    case_name = os.path.splitext(os.path.basename(slide_path))[0]
    return case_name

def logger():
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='dividing.log',
                    filemode='w')

def divide_prepare(slide_path, width:int):
    slide = openslide.OpenSlide(slide_path)
    properties = slide.properties
    try:
        level, base10 = base_10x(properties)
    except AssertionError:
        print(f'{get_name(slide_path)}')
        return

    dimensions = slide.level_dimensions
    dimension_ref = dimensions[0]
    ratio = int(properties.get(f'openslide.level[{str(level)}].downsample'))
    ori_mag = int(properties.get('openslide.objective-power'))
    d_step = width * ratio

    if base10:
        size = (width, width)
    else:
        w_new = width * (ori_mag / ratio / 10)
        size = (w_new, w_new)

    return slide, level, base10, dimension_ref, d_step, size

def devide_certain(slide_path: str, out_dir: str, width=96) -> None:
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
            small_image = slide.read_region(location=loc, level=level, size=size)
            if is_useless(small_image):
                continue
            if not base10:
                small_image = small_image.resize((width, width))
            fp = os.path.join(out_path, '{:010d}{:010d}.png'.format(j, i))
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
    cwp = os.getcwd()
    case_name = get_name(slide_path)
    # print(case_name)
    out_path = os.path.join(cwp, out_dir, case_name)
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
            fp = os.path.join(out_path, '{:010d}{:010d}.png'.format(j, i))
            resize_image.save(fp)
    
def is_useless(image) -> bool:
    """Help to judge whether the small image is informative.
    If a image has more information, it should be darker in gray mode.
    image: a Pillow object"""
    # if the width different from the height, it's the marginal part.
    if image.width != image.height:
        return True
    gray = image.convert("L")
    # 230 is a magic number, and it is not good. However, currently, I haven't found a better way
    # to select the informative images.
    return np.mean(gray) > 230

def show_progress(cur_done: int, total: int, status='', bar_length=60):
    """Show the progress on the terminal.
    cur_done: the number of finished work;
    totoal: the number of overall work;
    status: trivial words, str;
    bar_length: the length of bar showing on the screen, int."""
    percent = cur_done / total
    done = int(percent * bar_length)
    show = '=' * done + '/' * (bar_length - done)
    sys.stdout.write('[{}] {:.2f}% {}'.format(show, percent*100, status))
    sys.stdout.flush()

def main():
    path = sys.argv[1]
    slides = read_files(path)
    work_load = len(slides)
    done = 0
    out_dir = sys.argv[2]
    for slide_path in slides:
        devide_certain(slide_path, out_dir)
        done += 1
        show_progress(done, work_load, status='Please wait')

if __name__ == '__main__':
    main()
