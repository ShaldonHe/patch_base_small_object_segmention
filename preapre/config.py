import argparse as ag
import data.config as d_con
import libs.common.file_interface as libfi
import libs.common.config_interface as libci
# _model_g=m_c.parse_g_args()
_data_c=d_con.parse_args()
# model_d=m_c.parse_d_args()
"""parsing and configuration"""
def parse_args():
    desc = "config"
    parser = ag.ArgumentParser(description=desc)
    
    # parser.add_argument('--layer_num',type=int,default=_model_g.layer_num)

    # patch data path
    parser.add_argument('--model_image_dir', type=str, default=_data_c.model_image_dir)
    parser.add_argument('--model_label_dir', type=str, default=_data_c.model_label_dir)

    # origin data path
    parser.add_argument('--origin_image_path', type=str, default=_data_c.origin_image_dir)
    parser.add_argument('--origin_label_path', type=str, default=_data_c.origin_label_dir)

    # data info file
    parser.add_argument('--data_info_file', type=str, default='./data/model/data/data_layer_info.npz')

    # data ext
    parser.add_argument('--data_ext', type=str, default=_data_c.data_ext)

    # input_shape
    parser.add_argument('--patch_shape', type=list, default=_data_c.patch_shape)

    _
    args=parser.parse_args()
    return libci.check_config(args)

