import openslide
import os
import argparse
from tools import save_pickle, gen_logger
from tqdm import tqdm

funcs = {}
def record_funcs(func):
    funcs[func.__name__.split('_')[-1][:3]] = func
    return func

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
    fn = f'{get_name(svs_p)}.png'
    fp = os.path.join(dst, fn)
    image.save(fp)

@record_funcs
def process_dir(dir_p, dst):
    os.makedirs(dst, exist_ok=True)
    for slide_p in get_files(dir_p):
        output_lowest(slide_p, dst)

@record_funcs
def collect_power(dir_p, dst):
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, 'power.txt'), 'w+') as recorder:
        recorder.write('Name\tPower\t\Size\n')
        for slide_p in get_files(dir_p):
            try:
                properties = openslide.OpenSlide(slide_p).properties
                power = properties.get('openslide.objective-power', 0)
                size = (properties.get('openslide.level[0].height', 0), properties.get('openslide.level[0].width', 0))
                line = f'{get_name(slide_p)}\t{power}\t{size}\n'
                recorder.write(line)
            except:
                logger.exception(f'{slide_p} encounter errors')

@record_funcs
def collect_properties(dir_p, dst):
    container = {}
    for slide_p in tqdm(get_files(dir_p)):
        try:
            container[get_name(slide_p)] = openslide.OpenSlide(slide_p).properties
        except:
            logger.exception(f'{slide_p} encounter errors')
    save_pickle(container, '..', 'properties')

logger = gen_logger('overview')
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('dir')
    parse.add_argument('-f')
    parse.add_argument('-o', required=True)
    command = parse.parse_args()
    func = funcs.get(command.f, collect_properties)
    dir_p = command.dir
    dst = command.o
    func(dir_p, dst)