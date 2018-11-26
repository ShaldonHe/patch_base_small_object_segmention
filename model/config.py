import argparse as ag
import libs.common.config_interface as libci
im_size=512
im_channel=3
layer_num=1
class_num=1
def parse_d_args():
    parser = ag.ArgumentParser()
    parser.add_argument('--label_dir', type=str, default='./data/label_data/patch')
    parser.add_argument('--class_num',type=int,default=class_num)
    parser.add_argument('--input_shape',type=list,default=[im_size,im_size,im_channel+layer_num])#原图×2个尺寸
    parser.add_argument('--output_shape',type=list,default=[class_num])
    parser.add_argument('--model_dir', type=str, default='./data/model/gan_d/',help='Directory name to save the model')
    con=parser.parse_args()
    return libci.check_config(con)

def parse_g_args():
    desc = "gan g model"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--label_dir', type=str, default='./data/label_data/patch')
    parser.add_argument('--layer_num',type=int,default=layer_num)
    parser.add_argument('--input_shape',type=list,default=[im_size,im_size,im_channel])
    parser.add_argument('--output_shape',type=list,default=[im_size,im_size,im_channel+layer_num])#Label 1 channel
    parser.add_argument('--result_dir', type=str, default='./result/gan_g',help='Directory name to save the images')
    parser.add_argument('--model_dir', type=str, default='./data/model/gan_g/',help='Directory name to save the model')
    parser.add_argument('--pyramids', type=list, default=[1])
    con=parser.parse_args()
    return libci.check_config(con)

def parse_args():
    desc = "gan g model"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--label_dir', type=str, default='./data/label_data/patch')
    parser.add_argument('--layer_num',type=int,default=layer_num)
    parser.add_argument('--input_shape',type=list,default=[im_size,im_size,im_channel])
    parser.add_argument('--output_shape',type=list,default=[im_size,im_size,layer_num])#Label 1 channel
    parser.add_argument('--result_dir', type=str, default='./result/segmentation',help='Directory name to save the images')
    parser.add_argument('--model_dir', type=str, default='./data/model/segmentation/',help='Directory name to save the model')
    parser.add_argument('--pyramids', type=list, default=[1])
    con=parser.parse_args()
    return libci.check_config(con)

def parse_multitask_args():
    desc = "gan g model"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--label_dir', type=str, default='./data/label_data/patch')
    parser.add_argument('--layer_num',type=int,default=layer_num)
    parser.add_argument('--input_shape',type=list,default=[im_size,im_size,im_channel])
    parser.add_argument('--output_shape',type=list,default=[im_size,im_size,layer_num])#Label 1 channel
    parser.add_argument('--result_dir', type=str, default='./result/multitask',help='Directory name to save the images')
    parser.add_argument('--model_dir', type=str, default='./data/model/multitask/',help='Directory name to save the model')
    parser.add_argument('--pyramids', type=list, default=[1])
    con=parser.parse_args()
    return libci.check_config(con)

def parse_class_args():
    desc = "gan g model"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--label_dir', type=str, default='./data/label_data/patch')
    parser.add_argument('--layer_num',type=int,default=layer_num)
    parser.add_argument('--input_shape',type=list,default=[im_size,im_size,im_channel])
    parser.add_argument('--output_shape',type=list,default=[im_size,im_size,layer_num])#Label 1 channel
    parser.add_argument('--result_dir', type=str, default='./result/multitask',help='Directory name to save the images')
    parser.add_argument('--model_dir', type=str, default='./data/model/multitask/',help='Directory name to save the model')
    parser.add_argument('--pyramids', type=list, default=[1])
    con=parser.parse_args()
    return libci.check_config(con)