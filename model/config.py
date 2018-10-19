import argparse as ag

def patch_d_args():
    k_size=512
    layer_num=5
    parser = ag.ArgumentParser()
    parser.add_argument('--label_dir', type=str, default='./data/label_data/layer')
    parser.add_argument('--size',type=int,default=k_size)
    parser.add_argument('--layer_num',type=int,default=layer_num)
    parser.add_argument('--input_shape',type=list,default=[k_size,k_size,3])#原图×2个尺寸
    parser.add_argument('--output_shape',type=list,default=[k_size,k_size,layer_num])#Label 1 channel
    parser.add_argument('--pyramid', type=list, default=[1])
    parser.add_argument('--key_matcher',type=dict,default={
        'background':0,
        '真皮层':1,
        '基层':2,
        '真皮乳头层':2,
        '棘层':3,
        '棘细胞层':3,
        '角质层':4
    })
    parser.add_argument('--layer_label',type=list,default=[
        'background',
        '真皮层',
        '真皮乳头层',
        '棘层',
        '角质层'
    ])
    parser.add_argument('--result_dir', type=str, default='./result/layer_model',
                        help='Directory name to save the images')
    parser.add_argument('--model_dir', type=str, default='./data/model/layer/',
                        help='Directory name to save the model')
    con=parser.parse_args()
    return con

def parse_feature_args():
    desc = "Feature model"
    parser = ag.ArgumentParser(description=desc)
    parser.add_argument('--label_dir', type=str, default='./data/label_data/feature')

    parser.add_argument('--feature_num',type=int,default=7)
    parser.add_argument('--input_shape',type=list,default=[64,64,(3+1)*3])#(原图+layer分割结果)*3个尺寸
    parser.add_argument('--result_dir', type=str, default='./result/feature_model',
                        help='Directory name to save the images')
    parser.add_argument('--model_dir', type=str, default='./data/model/feature/',
                        help='Directory name to save the model')
    con=parser.parse_args()
    return con

def parse_slice_args():
    desc = "slice model"
    parser = ag.ArgumentParser(description=desc)

    parser.add_argument('--class_num',type=int,default=5)
    parser.add_argument('--key_matcher',type=dict,default={
        '正常':0,
        '单纯':0,
        '海绵':1,
        '苔藓':2,
        '血管炎':3,
        '银屑':4
    })
    parser.add_argument('--class_label',type=list,default=[
        '正常',
        '海绵',
        '苔藓',
        '血管炎',
        '银屑'
    ])
    parser.add_argument('--input_shape',type=list,default=[256,256,3+5+7])#原图+Layer channel+Feature Channel
    parser.add_argument('--label_dir', type=str, default='./data/label_data/slice')
    parser.add_argument('--result_dir', type=str, default='./result/slice_model')
    parser.add_argument('--model_dir', type=str, default='./data/model/slice/')
    con=parser.parse_args()
    return con
