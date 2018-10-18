import argparse as ag
import os
import numpy as np
import model.config as _model_c
_model=_model_c.parse_args()
"""parsing and configuration"""
def parse_args():
    desc = "patch based tiny object segmentation train config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The Learning Rate')
    parser.add_argument('--dtype',type=type,default=np.float32)
    parser.add_argument('--model_dir', type=str, default=_model.model_dir)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--n_epoch', type=int, default=100)
    return parser.parse_args()