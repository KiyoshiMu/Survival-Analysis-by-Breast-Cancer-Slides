import os
import pickle
import shutil
from collections import defaultdict
import numpy as np
from bisect import bisect_right
from copy import deepcopy

def pkl_dir_load(dir_p):
    container = {}
    for f_n in os.listdir(dir_p):
        if f_n[-4:] == '.pkl':
            f_p = os.path.join(dir_p, f_n)
            with open(f_p, 'rb') as case:
                data = pickle.load(case)
                container.update(data)
    return container

def to_case(result):
    outcome = defaultdict(list)
    for area in result.keys():
        if '.zip' in area:
            case_n = area.split('.zip', 1)[0]
        else:
            case_n = os.path.basename(os.path.dirname(area)) # for moving
            # case_n = os.path.basename(os.path.dirname(area)).split('.')[0]
        outcome[case_n].append(area)
    return outcome

def profile_count(case_dict):
    outcome = {}
    for case_n, areas in case_dict.items():
        count = len(areas)
        outcome[case_n] = count
    return outcome

def profile_threshold(case_dict, result, threshold=0.99, threshes=None):
    if threshes is None:
        threshes=np.arange(0.96, 1.00, 0.01)
    outcome = defaultdict(list)
    compare = defaultdict(list)
    thresh_num_temp = {thresh:defaultdict(int) for thresh in threshes}
    for case_n, areas in case_dict.items():
        for area in areas:
            p = result[area]
            idx = bisect_right(threshes, p)-1
            if idx >= 0:
                loc = threshes[idx]
                thresh_num_temp[loc][case_n] += 1
            if p >= threshold:
                outcome[case_n].append(area)
            elif p < 0.5:
                compare[case_n].append(area)
    return outcome, thresh_num_temp, compare

def refine_thresh_num_temp(thresh_num_temp, threshes, case_dict):
    thresh_num = deepcopy(thresh_num_temp)
    cases = case_dict.keys()
    threshes = threshes[::-1]
    for idx, thresh in enumerate(threshes[1:], 1):
        pre_thresh = threshes[idx-1]
        for case_n in cases:
            thresh_num[thresh][case_n] += thresh_num[pre_thresh][case_n]
    return thresh_num

def case_p_study(result, query):
    container = {}
    for k in result.keys():
        if query in k:
            container[k] = result[k]
    return container

def case_bound_search(case_ps, num=50, start=0.50):
    key_idx = np.arange(start, 0.96, 0.01)
    counts = [0] * len(key_idx)
    for p in case_ps.values():
        idx = bisect_right(key_idx, p)-1
        if idx >= 0:
            counts[idx] += 1
    base = counts[-1]
    loc = len(key_idx) - 1
    while base < num:
        loc -= 1
        if loc < 0:
            print('Search fail')
            print(counts)
            return None
        base += counts[loc]
    threshold = key_idx[loc]
    print(f'use threshold that is {key_idx[loc]}')
    return threshold

def case_select(case_ps, threshold=None):
    if threshold is None:
        return
    container = set()
    for k, p in case_ps.items():
        if p >= threshold:
            container.add(k)
    return container

def case_supply(result, case_names, num=50):
    recorder = {}
    for case_name in case_names:
        # print(case_name)
        case_ps = case_p_study(result, case_name)
        threshold = case_bound_search(case_ps, num=num, start=0.5)
        recorder[case_name] = case_select(case_ps, threshold)
    return recorder

def sel_move(selected_area, dst):
    need_patch = []
    for case_n, areas in selected_area.items():
        case_p = os.path.join(dst, case_n)
        os.makedirs(case_p, exist_ok=True)
        for area in areas:
            shutil.copy(area, case_p)
        if len(areas) < 50:
            need_patch.append(case_n)
    return need_patch

def lose_move(patch_supply, dst):
    for case, ps in patch_supply.items():
        dst = os.path.join(dst, case)
        os.makedirs(dst, exist_ok=True)
        for file in ps:
            try:
                shutil.copy(file, dst)
            except:
                print(case, file)
                break

def pkl_select(pkl_dir_p, dst):
    result = pkl_dir_load(pkl_dir_p)
    case_dict = to_case(result)
    threshes = np.arange(0.90, 1.01, 0.01)
    selected_area, _, _ = profile_threshold(case_dict, result, threshes=threshes)
    need_patch = sel_move(selected_area, dst)
    need_patch.extend(list(set(case_dict.keys()) - set(selected_area.keys())))
    patch_supply = case_supply(result, need_patch)
    lose_move(patch_supply, dst)