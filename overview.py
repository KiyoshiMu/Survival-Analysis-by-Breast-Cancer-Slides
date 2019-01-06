import openslide
import os
import sys

def get_files(dir_p):
    return [os.path.join(item[0], fn) for item in os.walk(dir_p) 
    for fn in item[2] if item[2] and fn[-4:]=='.svs']

def get_name(p):
    return os.path.splitext(os.path.basename(p))[0]

def read_lowest(svs_p):
    slide = openslide.OpenSlide(svs_p)
    level = slide.level_count - 1
    dimention = slide.level_dimensions[level]
    image = slide.read_region((0, 0), level, dimention)
    slide.close()
    return image

def output_lowest(svs_p, dst):
    image = read_lowest(svs_p)
    fn = get_name(svs_p)
    fp = os.path.join(dst, fn)
    image.save(fp)

def process_dir(dir_p, dst):
    os.makedirs(dst, exist_ok=True)
    for slide_p in get_files(dir_p):
        output_lowest(slide_p, dst)

if __name__ == "__main__":
    dir_p = sys.argv[1]
    dst = sys.argv[2]
    process_dir(dir_p, dst)