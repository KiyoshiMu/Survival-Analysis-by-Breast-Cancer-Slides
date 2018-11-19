import random
import pandas as pd
import os
import shutil
from tiles import show_progress

"""The 'Target.xlsx' shounld under the current work directory.
When random select 10 images from each case, generate a .xlsx which consists of
the correspondent y values, i.e. 2D arrays.
"""
def helper(path, dst, step=10, start=0):
    """Under the case directory, we randomly select files to construct data."""
    f_list = os.listdir(path)[1:]
    random.shuffle(f_list)
    for i, f in enumerate(f_list[:step], start):
        old = os.path.join(path, f)
        new = os.path.join(dst, '{:05d}.png'.format(i))
        shutil.copy(old, new)

def case_search(works, step=10):
    """Cases should be put in different directories. 
    After runnung, it formed data and created a y_value.txt.
    works: a list including all the dir names, list;
    step: the number of small images used from a slide, It ranges from 1 to 20, int."""
    # prepare directory and reference data
    parent = os.path.dirname(cwp)
    images_path = os.path.join(parent, 'images')
    os.makedirs(images_path, exist_ok=True)
    refer = pd.read_excel('Target.xlsx', index_col=[0])
    # begin work
    count = 0
    total = len(works)
    for done, fp in enumerate(works, start=1):
        f = os.path.split(fp)[-1]
        # the case name is formed by the front 12 characters of each slide name
        case_name = f[:12]
        try:
            duration, event = refer.loc[case_name]
        except KeyError:
            print('Case {} does not belong to the project.'.format(case_name))
            continue
        helper(fp, images_path, step=step, start=count)
        # prepare a file to record the relationship between cases and selected small images
        with open('case_to_num.txt', 'a') as c2n:
            line = '{} {} {:05d}\n'.format(f, case_name, count)
            c2n.write(line)
        # form the y value for each small image
        with open('y_values.txt', 'a') as y:
            
            for _ in range(step):
                count += 1
                line = '{:05d} {} {}\n'.format(count, duration, event)
                y.write(line)
        
        show_progress(done, total, status='y_dataset is under preparation')

def find_dir():
    res = []
    for f in os.listdir(cwp):
        fp = os.path.join(cwp, f)
        # all small images are under directories
        if os.path.isdir(fp):
            res.append(fp)
    return res
            
def main():
    works = find_dir()
    case_search(works)

if __name__ == "__main__":
    cwp = os.getcwd()
    main()
