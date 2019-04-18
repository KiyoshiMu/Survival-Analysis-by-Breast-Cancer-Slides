import argparse
import os
import json
from snas import SNAS, SNAS_predictor
from tools import gen_logger, load_locs, mark
from tiles import batch_tiling, marking
from area_judge import judge_area
from area_move import pkl_select

if __name__ == "__main__":
    logger = gen_logger('from_svs')
    parse = argparse.ArgumentParser(description='A pipeline from .svs files to survival models')
    parse.add_argument('i', help='the path of directory that saves imgs for cases')
    parse.add_argument('o', help='the path for output')
    parse.add_argument('-n', default='outcome', help='the name of .pkl file')
    parse.add_argument('-m', default='train', help='the working mode, if you want to use the prediction mode, just type "val"')
    command = parse.parse_args()
    dst = command.o
    tile_dst = os.path.join(dst, 'tiles')
    batch_tiling(command.i, tile_dst, logger)
    judge_area(tile_dst, dst=dst, pkl_name=command.n, logger=logger)
    sel_dst = os.path.join(dst, 'sel')
    pkl_select(os.path.join(dst, command.n), sel_dst)
    if command.o == 'train':
        model_dst = os.path.join(dst, 'models')
        model = SNAS(sel_dst, model_dst, logger=logger)
        model.whole_train()
    else:
        model = SNAS_predictor(sel_dst, dst, command.i, logger=logger)
        model.work()
        mark_info = load_locs(dst)
        mark_dst = os.path.join(dst, 'mark')
        marking(mark_info, mark_dst)