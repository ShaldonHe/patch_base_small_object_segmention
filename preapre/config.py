import argparse as ag
import os
import model.config as m_c
model_layer_c=m_c.parse_layer_args()
"""parsing and configuration"""
def parse_args():
    desc = "skin slice train config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--layer_num',type=int,default=model_layer_c.layer_num,help='分类的数量, default class_num: 2')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--model_saved_dir', type=str, default='./data/model/Saved/')

    return parser.parse_args()
