import argparse
import os
import json
from snas import SNAS, SNAS_predictor
from tools import gen_logger, load_locs, marking, move_model_val
from tiles import batch_tiling
from area_judge import judge_area
from area_move import pkl_select

if __name__ == "__main__":
    logger = gen_logger('from_svs')
    parse = argparse.ArgumentParser(description='A pipeline from .svs files to survival models')
    parse.add_argument('i', help='the path of directory that saves imgs for cases')
    parse.add_argument('o', help='the path for output')
    parse.add_argument('-m', default='train', help='the working mode, if you want to use the prediction mode, just type "val"')
    command = parse.parse_args()
    dst = command.o
    tile_dst = os.path.join(dst, 'tiles')
    # batch_tiling(command.i, tile_dst, logger)
    # logger.info('tiling done')
    pkl_dst = os.path.join(dst, 'pkl')
    # judge_area(tile_dst, dst=pkl_dst, logger=logger)
    # logger.info('judge done')
    sel_dst = os.path.join(dst, 'sel')
    # pkl_select(pkl_dst, sel_dst)
    # logger.info('selection done')
    if command.o == 'train':
        model_dst = os.path.join(dst, 'models')
        model = SNAS(sel_dst, model_dst, logger=logger)
        model.whole_train()
    else:
        model = SNAS_predictor(sel_dst, dst, logger=logger)
        model.work()
        mark_info = load_locs(dst)
        mark_dst = os.path.join(dst, 'mark')
        marking(mark_info, command.i, mark_dst)
        used_dst = os.path.join(dst, 'used')
        move_model_val(sel_dst, dst, used_dst)