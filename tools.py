import logging
import os
import random
import sys
import pickle
from imgaug import augmenters as iaa
import imgaug as ia
import pandas as pd
import openslide
import cv2
import shutil

class Train_table_creator:
    def __init__(self, selected_p, dst, train_ratio=0.8, target_p='data/Target.xlsx', logger=None):
        self.selected_p = selected_p
        self.dst = dst
        self.train_table_p = None
        self.test_table_p = None
        self.train_table = None
        self.test_table = None
        self.logger = logger if logger is not None else gen_logger('tools')
        self.create(train_ratio, target_p=target_p)

    def create(self, train_ratio=0.8, target_p='data/Target.xlsx'):
        """Change the ratio, then a new train table and a test table will be created"""
        self.train_table_p = os.path.join(self.dst, f'{train_ratio}train_table.xlsx')
        self.test_table_p = os.path.join(self.dst, f'{train_ratio}test_table.xlsx')
        if self.cache():
            self.logger.info('Use Cache')
            return
        if self.havefile():
            self._read_file()
            self.logger.info('Read from Files')
            return
        try:
            case_path_df = self._case2path(self.selected_p)
            merge_table = self._merge_table_creator(case_path_df, target_p=target_p)
            self.train_table, self.test_table = self._train_table_creator(merge_table, train_ratio)
        except:
            self.logger.exception('check here')

    def __call__(self, train_ratio=0.8):
        return self.create(train_ratio)

    def __repr__(self):
        print(f'Current img directory is {self.selected_p}, Cache is {self.cache()}')

    def get_gene_table(self, gene_p='data/gene.xlsx'):
        assert self.train_table is not None, 'train gene samples error'
        gene = pd.read_excel(gene_p, index_col=0)
        return gene.loc[self.train_table['sample']], gene.loc[self.test_table['sample']]

    def cache(self):
        return self.train_table is not None and self.test_table is not None

    def havefile(self):
        return os.path.isfile(self.train_table_p) and os.path.isfile(self.test_table_p)

    def _read_file(self):
        self.train_table = pd.read_excel(self.train_table_p)
        self.test_table = pd.read_excel(self.test_table_p)
            
    def _case2path(self, x_p):
        cur_dirs = os.listdir(x_p)
        result = {}
        for dir_n in cur_dirs:
            case = dir_n[:12]
            if case in result:
                continue
            result[case] = os.path.join(x_p, dir_n)
        return pd.DataFrame.from_dict(result, orient='index', columns=['path'])

    def _merge_table_creator(self, case_path_df, target_p='data/Target.xlsx'):
        target = pd.read_excel(target_p)
        merge_table = (case_path_df.merge(target, left_index=True, right_on='sample')).reset_index(drop=True)
        # merge_table.reset_index(drop=True, inplace=True)
        return merge_table

    def _train_table_creator(self, merge_table, train_ratio):
        idx = merge_table.index.tolist()
        random.shuffle(idx)

        train_size = round(len(idx) * train_ratio)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]

        train_table = ((merge_table.iloc[train_idx]).sort_values('duration')).reset_index(drop=True)
        test_table = ((merge_table.iloc[test_idx]).sort_values('duration')).reset_index(drop=True)

        train_table.to_excel(self.train_table_p)
        test_table.to_excel(self.test_table_p)
        self.logger.info('Searching Succeed')
        return train_table, test_table
        
def save_pickle(data, dst, name='record'):
    with open(os.path.join(dst, f'{name}.pkl'), 'ab') as record:
        pickle.dump(data, record, pickle.HIGHEST_PROTOCOL)

def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as temp:
        container = pickle.load(temp)
    return container

def get_files(path: str, suffix='svs') -> list:
    """From a dir read the .svs files, then we can divide the large slide into small ones
    path: the path of all .svs files."""
    result = [os.path.join(path, i) for i in os.listdir(path) if i.rsplit('.', 1)[-1]==suffix]
    return result

def get_name(slide_path):
    case_name = os.path.splitext(os.path.basename(slide_path))[0]
    return case_name

def gen_logger(name='', stream=True):
    name=f'{name}.log'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(f'../{name}')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

def marking(mark_info:list, slides_p, dst):
    os.makedirs(dst, exist_ok=True)
    for case, makrers in mark_info:
        slide_path = os.path.join(slides_p, f'{case}.svs')
        large_image = openslide.OpenSlide(slide_path)
        size = tuple(large_image.level_dimensions[2])
        ratio = large_image.level_dimensions[0][0] / size[0]
        assert len(size) == 2, 'size error'
        raw_image = large_image.read_region(location=(0,0), level=2, size=size).convert("RGB")
        fp = f'{os.path.join(dst, case)}.jpg'
        raw_image.save(fp)
        raw_image = cv2.imread(fp)
        for marker in makrers:
            p1 = (int(marker[0]/ratio), int(marker[1]/ratio))
            p2 = (int((marker[0]+marker[2])/ratio), int((marker[1]+marker[3])/ratio))
            cv2.rectangle(raw_image, p1, p2, (0,255,0), 5)
            # cv2.line(raw_image, p1, (int(size[1]*0.9), p1[1]), (0,255,0), 5)
        cv2.imwrite(fp, raw_image)

def move_model_val(sel_p, loc_dst, dst):
    with open(os.path.join(loc_dst, 'locs.txt'), 'r') as locs:
        for line in locs:
            case, sels = line.split('\t')
            val_p = os.path.join(dst, case)
            os.makedirs(val_p, exist_ok=True)
            for info in sels.split(','):
                shutil.copy(os.path.join(sel_p, case, f'{info.strip()}.tiff'), val_p)

def load_locs(dst):
    mark_info = []
    with open(os.path.join(dst, 'locs.txt'), 'r') as locs:
        for line in locs:
            case, sels = line.split('\t')
            markers = [[int(n) for n in info.split('-')] for info in sels.split(',')]
            mark_info.append((case, markers))
    return mark_info

def get_seq():
    """From kaggle, augment an array of images. However, in this study, the performance of SNAS is better without augmentation"""
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                    ]),
                    iaa.Invert(0.01, per_channel=True), # invert color channels
                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq