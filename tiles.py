# import numpy as np
import openslide
import os
import cv2
import sys
import numpy as np

def read_files(path):
    """from a dir read the .svs files, then we can divide the large slide into small ones"""
    for f in os.listdir(path):
        if f[-4:] == '.svs':
            yield os.path.join(path, f)

def divide(slide, level=1, width_rel=250):
    """the origin image is too large, the function can segment the large one into small tiles.
    slide is the path of the target slide; level points varying definition of images, 0 is the largest;
    width_rel is the width of output images. In this project, we set the height equals to the width. """
    # read slide
    large_image = openslide.OpenSlide(slide)
    tile = width_rel * 20
    # get reference and target location
    dimensions = large_image.level_dimensions
    dimension_ref = dimensions[0]
    dimension = dimensions[level]
    ratio = dimension[0] / dimension_ref[0]
    # set start points of tiles
    widths_point = list(range(0, dimension_ref[0], tile))
    heights_point = list(range(0, dimension_ref[1], tile))
    # begin segment tiles
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
                break
            # save the small image
            height_rel = width_rel
            resize_image = small_image.resize((width_rel, height_rel))
            fp = os.path.join(os.getcwd(), '{:02d}{:02d}.png'.format(j, i))
            resize_image.save(fp)
    
def is_useless(image):
    # 
    if image.width != image.height:
        return True
    gray = image.convert("L")
    return np.mean(gray) > 230

def main():
    slide = sys.argv[1]
    print(slide)
    divide(slide)

if __name__ == '__main__':
    main()