import argparse as ag
import numpy as np
import model.config as _model_c
import train.config as _train_c
_train_layer=_train_c.parse_layer_args()
_model_layer=_model_c.parse_layer_args()
"""parsing and configuration"""
def parse_layer_args():
    desc = "layer segmention application config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--dtype',type=type,default=np.float32)
    parser.add_argument('--model_dir', type=str, default=_model_layer.model_dir)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--image_dir', type=str, default='../data/银屑4倍')
    parser.add_argument('--image_ext', type=str, default='bmp')
    parser.add_argument('--batch_size',type=int, default=_train_layer.batch_size)
    return parser.parse_args()