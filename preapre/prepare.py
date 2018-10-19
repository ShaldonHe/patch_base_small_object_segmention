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
import libs.common.img_pair_interface as libimg
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

def _loop_equal(a,b):
    if type(a)==type(b):
        if isinstance(a,list) or isinstance(a,tuple) or isinstance(a,set):
            if len(a)!=len(b):
            return False
            _a=sorted(a)
            _b=sorted(b)
            for _a_i,_b_i in zip(a,b):
                if not _loop_equal(_a_i,_b_i):
                    return False
        elif isinstance(a,dict):
            if len(a)!=len(b):
                return False
            _key_a=sorted(a.keys())
            _key_b=sorted(b.keys())
            if not _loop_equal(_key_a,_key_b):
                return False
            else:
                for k in _key_a:
                    if not _loop_equal(a[key],b[key]):
                        return False
        else:
            return a==b
        return True
    else:
        return False

 
def progress():
    if not osp.exists(_c.data_info):
        data_info={}
    else:
        loaded=np.load(_c.data_info)
        data_info=loaded['data_info']
    
    need_repatch=False
    if not need_repatch:
        key='origin_image_dict'
        if key not in data_info:
            need_repatch=True
        else:
            _tmp=libfi.getfiledicbyext(_c.origin_image_path,_c.data_ext)
            if not _loop_equal(_tmp,data_info[key]):
                need_repatch=True
    if not need_repatch:
        key='origin_label_dict'
        if key not in data_info:
            need_repatch=True
        else:
            _tmp=libfi.getfiledicbyext(_c.origin_label_path,_c.data_ext)
            if not _loop_equal(_tmp,data_info[key]):
                need_repatch=True
    if not need_repatch:
        key='patch_image_dict'
        if key not in data_info:
            need_repatch=True
        else:
            _tmp=libfi.getfiledicbyext(_c.patch_image_path,_c.data_ext)
            if not _loop_equal(_tmp,data_info[key]):
                need_repatch=True
    if not need_repatch:
        key='patch_label_dict'
        if key not in data_info:
            need_repatch=True
        else:
            _tmp=libfi.getfiledicbyext(_c.patch_label_path,_c.data_ext)
            if not _loop_equal(_tmp,data_info[key]):
                need_repatch=True
    if not need_repatch:
        key='parms'
        if key not in data_info:
            need_repatch=True
        else:
            _tmp=dict(radius=_c.radius, pyramids=_c.pyramids, stride=_c.stride, angles=_c.angles)
            if not _loop_equal(_tmp,data_info[key]):
                need_repatch=True
    
    if need_repatch:
        # crop origin images to patch images
        origin_image_dict=libfi.getfiledicbyext(_c.origin_image_path,_c.data_ext)
        data_info['origin_image_dict']=origin_image_dict
        origin_label_dict=libfi.getfiledicbyext(_c.origin_label_path,_c.data_ext)
        data_info['origin_label_dict']=origin_label_dict
        patch_image_dict={}
        patch_label_dict={}
        parms=dict(radius=_c.radius, pyramids=_c.pyramids, stride=_c.stride, angles=_c.angles)
        data_info['parms']=parms
        for key in origin_image_dict:
            image_path=origin_image_dict[key]
            image=prep.imread(image_path)
            if key not in origin_label_dict:
                label_path=None
            else:
                label_path=origin_label_dict[key]
            
            if label_path is None:
                label=np.zeros(shape=image.shape[0:2],dtype=np.float)
            else:
                label=prep.imread(label_path,c=0)
            
            ids=libvi.image_to_blocksinfo(image,
                radius=parms['radius'],
                angles=parms['angles'],
                stride=parms['stride'],
                pyramid=parms['pyramids'])
            for _i in ids:
                p_image=libvi.get_block_fromids(image,_i,block_size=_c.patch_shape)
                p_label=libvi.get_block_fromids(label,_i,block_size=_c.patch_shape)
                patch_id='{i.centroid}-{i.radius}-{i.pyramid}-{i.angle}'.format(i=_i)
                p_image_path='{root}/{key}_{patch_id}.{ext}'.format(
                    root=_c.patch_image_path,
                    key=key,
                    patch_id=patch_id,
                    ext='png')
                p_label_path='{root}/{key}_{patch_id}.{ext}'.format(
                    root=_c.patch_label_path,
                    key=key,
                    patch_id=patch_id,
                    ext='png')
                prep.imwrite(p_image_path,p_image)
                prep.imwrite(p_label_path,p_label)
                print(key,patch_id)
        data_info['patch_image_dict']=libfi.getfiledicbyext(_c.patch_image_path,_c.data_ext)
        data_info['patch_label_dict']=libfi.getfiledicbyext(_c.patch_label_path,_c.data_ext)
        np.savez_compressed(_c.data_info_file,data_info=data_info)