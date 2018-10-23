import argparse as ag
import os
import numpy as np
import model.config as _model_con
_model_c_g=_model_con.parse_g_args()
_model_c_d=_model_con.parse_d_args()
_model_c_s=_model_con.parse_args()
"""parsing and configuration"""
def parse_args():
    desc = "patch based tiny object segmentation train config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The Learning Rate')
    parser.add_argument('--dtype',type=type,default=np.float32)
    parser.add_argument('--model_g_dir', type=str, default=_model_c_g.model_dir)
    parser.add_argument('--model_s_dir', type=str, default=_model_c_s.model_dir)
    parser.add_argument('--model_d_dir', type=str, default=_model_c_d.model_dir)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--n_epoch', type=int, default=50)
    return parser.parse_args()