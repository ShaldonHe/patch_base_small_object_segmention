import argparse as ag
import os
import train.config as _train_con
import test.config as _test_con
_train_layer=_train_con.parse_layer_args()
_test_layer=_test_con.parse_layer_args()
_model_layer=_train_con._model_layer
"""parsing and configuration"""
def parse_args():
    desc = "skin slice train config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='../data/',help='The data dirs')
    parser.add_argument('--model_data_path', type=str, default='./data/model_data')
    parser.add_argument('--data_info', type=str, default='./data/model_data/data_info.npz')
    parser.add_argument('--data_layer_info', type=str, default='./data/model_data/data_layer_info.npz')
    parser.add_argument('--data_feature_info', type=str, default='./data/model_data/data_feature_info.npz')
    parser.add_argument('--data_slice_info', type=str, default='./data/model_data/data_slice_info.npz')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='Directory name to save the images')
    parser.add_argument('--radius', type=int, default=16,help='block radius')
    parser.add_argument('--stride', type=int, default=16,help='block stride in train progress')
    parser.add_argument('--layers_label',type=list, default=['_background_','角质层','棘层','基层','真皮层'],help='分层模型的标签')

    return parser.parse_args()

def parse_layer_args():
    desc = "skin layers model train config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--label_path', type=str, default=_model_layer.label_dir,help='The label dirs')
    parser.add_argument('--model_data_path', type=str, default='./data/model_data')
    parser.add_argument('--data_info', type=str, default='./data/model_data/data_layer_info.npz')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='Directory name to save the images')
    parser.add_argument('--radius', type=int, default=_model_layer.input_shape[0]//2,help='block radius')
    parser.add_argument('--pyramid', type=list, default=_model_layer.pyramid)
    parser.add_argument('--stride', type=int, default=max(_model_layer.input_shape[0]//32,8),help='block stride in train progress')
    parser.add_argument('--data_shape', type=list, default=_model_layer.input_shape)
    parser.add_argument('--label_shape', type=list, default=_model_layer.output_shape)
    parser.add_argument('--layer_num', type=int, default=_model_layer.layer_num)
    parser.add_argument('--train_batch_size', type=int, default=_train_layer.batch_size)
    parser.add_argument('--layers_label',type=list, default=['_background_','角质层','棘层','基层','真皮层'],help='分层模型的标签')
    parser.add_argument('--test_batch_size', type=int, default=_test_layer.batch_size)
    return parser.parse_args()

def parse_feature_args():
    desc = "skin feature model train config"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='../data/',help='The data dirs')
    parser.add_argument('--model_data_path', type=str, default='./data/model_data')
    parser.add_argument('--data_info', type=str, default='./data/model_data/data_info.npz')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='Directory name to save the images')
    parser.add_argument('--radius', type=int, default=32,help='block radius')
    parser.add_argument('--reg', type=list, default=[1,2])
    parser.add_argument('--stride', type=int, default=16,help='block stride in train progress')
    parser.add_argument('--layers_label',type=list, default=['_background_','角质层','棘层','基层','真皮层'],help='分层模型的标签')

    return parser.parse_args()