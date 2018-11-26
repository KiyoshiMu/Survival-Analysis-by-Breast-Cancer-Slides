import openslide
import os
import sys
import numpy as np

def read_files(path):
    """From a dir read the .svs files, then we can divide the large slide into small ones
    path: the path of all .svs files, str."""
    # for f in os.listdir(path):
    #     if f[-4:] == '.svs':
    #         yield os.path.join(path, f)
    return [os.path.join(path, i) for i in os.listdir(path) if i[-4:] == '.svs']

def divide(slide_path, out_dir, level=1, width_rel=256, mag=0.5):
    """The origin slide is too large, the function can segment the large one into small tiles.
    In this project, we set the height equals to the width.
    Slide_path: the path of the target slide, str; 
    level: varying definition of images, 0 is the largest, int;
    width_rel: the width and the length of output images, int.
    mag: the magnitude or the object power, float"""
    # Read slide. level 0 is mag X40, level 1 is mag X10 and so on.
    # here we downsize 400 (20*20) times  (20 = level 1 's mag X10 / mag)
    large_image = openslide.OpenSlide(slide_path)
    time = 10 / mag
    tile = width_rel * time
    # get reference and target location
    dimensions = large_image.level_dimensions
    dimension_ref = dimensions[0]
    dimension = dimensions[level]
    ratio = dimension[0] / dimension_ref[0]
    # set start points of tiles
    widths_point = list(range(0, dimension_ref[0], tile))
    heights_point = list(range(0, dimension_ref[1], tile))
    # begin segment tiles
    cwp = os.getcwd()
    case_name = os.path.basename(slide_path)[:-4]
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
            fp = os.path.join(out_path, '{:02d}{:02d}.png'.format(j, i))
            resize_image.save(fp)
    
def is_useless(image):
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

def show_progress(cur_done, total, status='', bar_length=60):
    """Show the progress on the terminal.
    cur_done: the number of finished work, int;
    totoal: the number of overall work, int;
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
        divide(slide_path, out_dir)
        done += 1
        show_progress(done, work_load, status='Please wait')

if __name__ == '__main__':
    main()
