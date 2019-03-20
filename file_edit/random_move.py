import os
import shutil
import random
import sys
sys.path.append('..')
from tools import load_pickle
from tqdm import tqdm
import argparse
random.seed(42)

def get_link(pkl_p='', selected_p=''):
    if pkl_p:
        try:
            link = load_pickle(pkl_p)
        except FileNotFoundError:
            print(f'No {pkl_p}')
        else:
            return link
    if selected_p:
        return img_search(selected_p)

def img_search(selected_p):
    link = {}
    for case in os.listdir(selected_p):
        cp = os.path.join(selected_p, case)
        link[case] = [os.path.join(cp, fn) for fn in os.listdir(cp)]
    return link

def random_select(link, num=500):
    random_result = {}
    for case, ps in link.items():
        if len(ps) > num:
            ps = random.choices(ps, k=num)
        random_result[case] = ps
    return random_result

def img_move(random_result, dst='c:/selected'):
    for case, ps in tqdm(random_result.items()):
        case_dst = os.path.join(dst, case)
        os.makedirs(case_dst, exist_ok=True)
        for p in ps:
            shutil.copy(p, case_dst)

def main(pkl_p='', selected_p='', dst='c:/selected', num=500):
    link = get_link(pkl_p=pkl_p, selected_p=selected_p)
    random_result = random_select(link, num=num)
    img_move(random_result, dst=dst)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='This stript is used for random selection of certain \
number of samll images. Then copy them to SDS disk.')
    parse.add_argument('i')
    parse.add_argument('-o', default='c:/selected')
    parse.add_argument('-n', default=500, type=int, help='the number of imgs for selection')
    command = parse.parse_args()
    in_p = command.i
    if in_p[-3:] == 'pkl':
        main(pkl_p=in_p, dst=command.o, num=command.n)
    else:
        main(selected_p=in_p, dst=command.o, num=command.n)