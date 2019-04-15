from snas import SNAS
from tools import gen_logger
import argparse
import os
from tiles import batch_tiling
from area_judge import judge_area
from area_move import pkl_select

if __name__ == "__main__":
    logger = gen_logger('from_svs')
    parse = argparse.ArgumentParser(description='A pipeline from .svs files to survival models')
    parse.add_argument('i', help='the path of directory that saves imgs for cases')
    parse.add_argument('o', help='the path for output')
    parse.add_argument('-n', default='outcome', help='the name of .pkl file')
    command = parse.parse_args()
    dst = command.o
    tile_dst = os.path.join(dst, 'tiles')
    batch_tiling(command.i, tile_dst)
    judge_area(command.o, dst=tile_dst, pkl_name=command.n)
    sel_dst = os.path.join(dst, 'sel')
    pkl_select(os.path.join(dst, command.n), sel_dst)
    model_dst = os.path.join(dst, 'models')
    model = SNAS(command.i, model_dst, logger=logger)