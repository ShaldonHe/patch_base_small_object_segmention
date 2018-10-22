import argparse as ag
import os
import model.config as _m_con
import libs.common.config_interface as libci
# _model_c=model_con.
"""parsing and configuration"""
_m_c_g=_m_con.parse_g_args()
def parse_args():
    desc = "data config"
    parser = ag.ArgumentParser(description=desc)
    
    parser.add_argument('--origin_image_dir', type=str, default='/home/xiaodonghe/Documents/Data/CHXD/segmentation/Train/Images/Cropped')
    parser.add_argument('--origin_label_dir', type=str, default='/home/xiaodonghe/Documents/Data/CHXD/segmentation/Train/Labels/Cropped')

    parser.add_argument('--model_image_dir', type=str, default='./data/model/data/prep_image')
    parser.add_argument('--model_label_dir', type=str, default='./data/model/data/prep_label')


    parser.add_argument('--data_ext', type=str, default='png|jpeg|bmp|jpg')

    parser.add_argument('--data_info_file', type=str, default='./data/model/data/data_layer_info.npz')
    
    parser.add_argument('--radius', type=int, default=_m_c_g.input_shape[0]//2,help='block radius')
    parser.add_argument('--pyramid', type=list, default=_m_c_g.pyramid)
    parser.add_argument('--stride', type=int, default=max(_m_c_g.input_shape[0]//32,8),help='block stride in train progress')

    parser.add_argument('--patch_shape', type=list, default=_m_c_g.input_shape[0:2])
    
    parser.add_argument('--data_shape', type=list, default=_m_c_g.input_shape)
    parser.add_argument('--label_shape', type=list, default=_m_c_g.output_shape)
    parser.add_argument('--layer_num', type=int, default=_m_c_g.layer_num)
    # parser.add_argument('--train_batch_size', type=int, default=_train.batch_size)
    return libci.check_config(parser.parse_args())