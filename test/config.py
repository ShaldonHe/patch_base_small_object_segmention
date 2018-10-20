import argparse as ag
import numpy as np
import model.config as _model_con
_model_c=_model_con.parse_g_args()
"""parsing and configuration"""
def parse_args():
    desc = "layer segmention test config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=10, help='The size of batch')
    parser.add_argument('--dtype',type=type,default=np.float32)
    parser.add_argument('--model_dir', type=str, default=_model_layer.model_dir)
    parser.add_argument('--shuffle', type=bool, default=False)
    return parser.parse_args()