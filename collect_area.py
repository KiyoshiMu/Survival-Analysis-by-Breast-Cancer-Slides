import os
import shutil
from collections import defaultdict
import numpy as np
from tools import load_pickle, gen_logger
import sys

logger = gen_logger(name='collect.txt')

def weird_load(fp):
    container = {}
    with open(fp, 'rb') as file:
        while True:
            chunk = file.read(3408)
            if not chunk:
                break
            with open('_', 'wb') as temp:
                temp.write(chunk)
            data = load_pickle('_')
            container.update(data)
    return container

def to_case(result):
    outcome = defaultdict(list)
    for area in result.keys():
        case_n = area.rsplit('/', 2)[-2]
        outcome[case_n].append(area)
    return outcome

def profile_threshold(case_dict, result, threshold=0.98):
    outcome = defaultdict(list)
    threshes = np.arange(0.96, 1.00, 0.01)
    thresh_temp = {thresh:defaultdict(int) for thresh in threshes}
    for case_n, areas in case_dict.items():
        for area in areas:
            p = result[area]
            loc = round(p-0.004, 2)
            if loc in thresh_temp.keys():
                thresh_temp[loc][case_n] += 1
            if p >= threshold:
                outcome[case_n].append(area)
                
    return outcome, thresh_temp

def download_fps(selected_area):
    try:
        os.makedirs('../download', exist_ok=True)
        for case_n, areas in selected_area.items():
            case_p = f'../download/{case_n}'
            os.makedirs(case_p, exist_ok=True)
            for area in areas:
                shutil.copy(area, case_p)
    except:
        logger.exception('encountered error in batch')
        
if __name__ == "__main__":
    result = weird_load(sys.argv[1])
    case_dict = to_case(result)
    selected_area, thresh_temp = profile_threshold(case_dict, result)
    download_fps(selected_area)