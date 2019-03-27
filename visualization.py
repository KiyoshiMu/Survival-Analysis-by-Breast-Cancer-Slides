# learn from https://github.com/jacobgil/keras-cam
import os
import argparse
from models import model_nas
from keras.applications.nasnet import preprocess_input
import cv2
import numpy as np
from keras import backend as K
from tools import gen_logger

def read_dir(dir_p, sel_num) -> list:
    pool = os.listdir(dir_p)
    pool_size = len(pool)
    cur, end = 0, 0
    ori_carrier, pro_carrier = [], []
    while end < pool_size:
        end += sel_num
        while cur < end:
            ori_img = cv2.imread(os.path.join(dir_p, pool[cur]))
            ori_carrier.append(ori_img)
            pro_carrier.append(preprocess_input(ori_img))
            cur += 1
        yield np.array(ori_carrier), np.array(pro_carrier)
        ori_carrier.clear()
        pro_carrier.clear()

def visualize_class_activation_map(dir_p, output_path, sel_num=1):
    model = model_nas()
    
    for ori_imgs, imgs in read_dir(dir_p, sel_num):
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()
        final_conv_layer = model.get_layer(name="NASNet")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, _] = get_output(imgs)
        for idx, ori_img in enumerate(ori_imgs):
            conv_output = conv_outputs[idx, :, :, :]

            #Create the class activation map.
            cam = np.zeros(dtype = np.float32, shape = conv_output.shape[1:3])
            for i, w in enumerate(class_weights[:, 1]):
                cam += w * conv_output[i, :, :]
            cam /= np.max(cam)
            cam = cv2.resize(cam, (96, 96))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.2)] = 0
            img = heatmap*0.5 + ori_img
            cv2.imwrite(output_path, img)



if __name__ == "__main__":
    logger = gen_logger('vis')
    parse = argparse.ArgumentParser()
    parse.add_argument('i', help='the path of directory that saves imgs for cases')
    parse.add_argument('-o', default='..', help='the path for output')
    parse.add_argument('-s', type=int, default=1, help='the num of imgs used for visualization')

    command = parse.parse_args()
    logger.info(f'Begin train on {command}')
    dst = command.o
    os.makedirs(dst, exist_ok=True)
    try:
        visualize_class_activation_map(command.o, dst, sel_num=command.s)
    except:
        logger.Exception('debug')