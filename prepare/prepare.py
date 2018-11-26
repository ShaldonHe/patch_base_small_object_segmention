import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
from . import config as _con
# import data.config as _data_con
# import model.config as _model_con
_c=_con.parse_args()
#prepare train file
import libs.common.file_interface as libfi
import matplotlib.pyplot as plt
import libs.common.class_interface as libci
import libs.common.view_interface as libvi
import libs.collection.prep_utility as prep
# 1. Crop the images and labels to patch image and label
# 2. Write data info to data_info file
#       data information   
#       (1) the dict of origin images
#       (2) the dict of origin labels
#           (the number of images, the number of labels may be
#           different, because some images may not contain any 
#           lesions, there is no labels for these images)
#       (3) parms: radius, pyramid, stride, angles
#       (4) the dict of patch images
#       (5) the number of patch labels

def load_data_dcit(root,data_structure,data_ext):
    result={}
    data_dicts={}
    for key in data_structure:
        data_dicts[key]=libfi.getfiledicbyext('{root}/{subfolder}'.format(root=root,subfolder=data_structure[key]),ext=data_ext)
    
    d_keys=list(data_structure.keys())
    file_keys=list(data_dicts[d_keys[0]].keys())
    file_keys.sort()
    for key in file_keys:
        tmp_d={}
        for d_key in d_keys:
            tmp_d[d_key]=data_dicts[d_key][key]
        result[key]=libci.dict2object(tmp_d)
    
    return result

def load_image(data,tar_size=(2048,2048)):
    image=prep.imread(data.image)
    label=prep.imread(data.label,c=0)
    vessel=prep.imread(data.vessel,c=0)


    if image.shape[:2]!=tar_size:
        image=prep.resize_image(image,tar_size=tar_size)
    if label.shape[:2]!=tar_size:
        label=prep.resize_image(label,tar_size=tar_size)
    if vessel.shape[:2]!=tar_size:
        vessel=prep.resize_image(vessel,tar_size=tar_size)
        
    return image,label,vessel

 
def progress():
    if not osp.exists(_c.data_info_file):
        data_info={}
    else:
        loaded=np.load(_c.data_info_file)
        data_info=loaded['data_info'].item()

    data_dict=load_data_dcit(_c.origin_data_root,_c.origin_data_structure,_c.data_ext)
    parms=dict(radius=_c.radius, pyramids=_c.pyramids, stride=_c.stride, angles=_c.angles)
    data_info['parms']=parms
    for key in data_dict:
        ob=data_dict[key]
        image,label,vessel=load_image(ob)
        ids,_=libvi.image_to_blocksinfo(image,
            radius=parms['radius'],
            angles=parms['angles'],
            stride=parms['stride'],
            pyramid=parms['pyramids'])
        for _i in ids:
            p_image=libvi.get_block_fromids(image,_i,block_size=_c.patch_shape)
            p_label=libvi.get_block_fromids(label,_i,block_size=_c.patch_shape)
            p_vessel=libvi.get_block_fromids(vessel,_i,block_size=_c.patch_shape)
            patch_id=_i.get_id()
            p_image_path='{root}/{key}_{patch_id}/{data_type}.{ext}'.format(
                root=_c.model_data_dir,
                key=key,
                data_type='image',
                patch_id=patch_id,
                ext='jpg')
            p_label_path='{root}/{key}_{patch_id}/{data_type}.{ext}'.format(
                root=_c.model_data_dir,
                key=key,
                data_type='label',
                patch_id=patch_id,
                ext='png')
            p_vessel_path='{root}/{key}_{patch_id}/{data_type}.{ext}'.format(
                root=_c.model_data_dir,
                key=key,
                data_type='vessel',
                patch_id=patch_id,
                ext='png')

            if not libfi.exist('{root}/{key}_{patch_id}'.format(root=_c.model_data_dir,key=key,patch_id=patch_id)):
                libfi.make_dir('{root}/{key}_{patch_id}'.format(root=_c.model_data_dir,key=key,patch_id=patch_id))
            if np.sum(p_label)<=2:
                continue

            prep.imwrite(p_image_path,p_image.astype(np.uint8))
            prep.imwrite(p_label_path,p_label.astype(np.uint8))
            prep.imwrite(p_vessel_path,p_vessel.astype(np.uint8))
            print(key,patch_id)

    data_info['patch_data_dict']=data_dict
    np.savez_compressed(_c.data_info_file,data_info=data_info)

import shutil as s
import os
def clear_images():
    image_dicts=libfi.getfiledicbyext(_c.model_image_dir,ext='jpg')
    label_dicts=libfi.getfiledicbyext(_c.model_label_dir,ext='png')
    for k,v in label_dicts.items():
        label=prep.imread(v,0)
        if np.sum(label)<=1:
            os.remove(v)
            os.remove(image_dicts[k])
            print(k)

    for k,v in image_dicts.items():
        if k not in label_dicts:
            os.remove(v)

