from . import config as con
import libs.common.file_interface as libfi
import numpy as np
import tensorflow as tf
import libs.common.data_interface as libsdi
from libs.collection.prep_utility import imread,imwrite
def MA_segmention_input():
    c=con.parse_args()
    
    image_dicts=libfi.getfiledicbyext(c.model_image_dir,ext='jpg')
    label_dicts=libfi.getfiledicbyext(c.model_label_dir,ext='png')

    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img

    data=libsdi.mutil_file_reader(image_dicts,block_shape=c.data_shape,transform_fn=_transform_image_data)
 

    def _transform_label_data(file_path):
        img=imread(file_path,c=0)
        if img.max()>1:
            img=img.astype(np.float)/255.0
            
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        if c.layer_num==1:
            label_img[:,:,0]=img.astype(np.float)
        else:
            for i in range (c.layer_num):
                label_img[:,:,i]=(img==i).astype(np.float)
        
        return label_img

    label=libsdi.mutil_file_reader(label_dicts,block_shape=c.label_shape,transform_fn=_transform_label_data)
    return tf.estimator.inputs.numpy_input_fn(
    x={"images": data},
    y=label,
    batch_size=c.train_batch_size,
    num_epochs=None,
    shuffle=True),data,label

def MA_segmention_data():
    c=con.parse_args()
    # data_info=np.load(c.data_info_file)['data_info'].item()
    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img
    
    image_dicts=libfi.getfiledicbyext(c.model_image_dir,ext='jpg')
    label_dicts=libfi.getfiledicbyext(c.model_label_dir,ext='png')

    data=libsdi.rank_file_reader(image_dicts,block_shape=c.data_shape,transform_fn=_transform_image_data)
 
    def _transform_label_data(file_path):
        img=imread(file_path,c=0)
        if img.max()>1:
            img=img.astype(np.float)/255.0
            
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        if c.layer_num==1:
            label_img[:,:,0]=img.astype(np.float)
        else:
            for i in range (c.layer_num):
                label_img[:,:,i]=(img==i).astype(np.float)
        return label_img

    label=libsdi.rank_file_reader(label_dicts,block_shape=c.label_shape,transform_fn=_transform_label_data)
    return data,label



def MA_segmention_debug_data():
    c=con.parse_args()
    data_info=np.load(c.data_info_file)['data_info'].item()

    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img
    
    data_dict=data_info['patch_image_dict']
    keys=list(data_dict.keys())[:40]
    _data_dict={}
    for k in keys:
        _data_dict[k]=data_dict[k]
    data=libsdi.rank_file_reader(_data_dict,block_shape=c.data_shape,transform_fn=_transform_image_data)
 
    def _transform_label_data(file_path):
        img=imread(file_path,c=0)
        if img.max()>1:
            img=img.astype(np.float)/255.0
            
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        if c.layer_num==1:
            label_img[:,:,0]=img.astype(np.float)
        else:
            for i in range (c.layer_num):
                label_img[:,:,i]=(img==i).astype(np.float)
        return label_img
    label_dict=data_info['patch_label_dict']
    _label_dict={}
    for k in keys:
        _label_dict[k]=label_dict[k]

    label=libsdi.rank_file_reader(_label_dict,block_shape=c.label_shape,transform_fn=_transform_label_data)
    return data,label

def MA_segmention_train_input():
    c=con.parse_args()
    data_info=np.load(c.data_info_file)['data_info'].item()
    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img

    data=libsdi.mutil_file_reader(data_info['patch_image_dict'],block_shape=c.data_shape,transform_fn=_transform_image_data)
 

    def _transform_label_data(file_path):
        img=imread(file_path,c=0)
        if img.max()>1:
            img=img.astype(np.float)/255.0
            
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        for i in range (c.layer_num):
            label_img[:,:,i]=(img==i).astype(np.float)
        return label_img

    label=libsdi.mutil_file_reader(data_info['patch_label_dict'],block_shape=c.label_shape,transform_fn=_transform_label_data)
    return data._index2ids,data,label





    




