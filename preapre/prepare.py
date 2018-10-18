import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
from . import config as con
import data.config as data_con
import model.config as model_con
c=con.parse_args()
data_c=data_con.parse_args()
slice_c=model_con.parse_slice_args()
layer_c=model_con.parse_layer_args()
#prepare train file
import libs.common.file_interface as libfi
import matplotlib.pyplot as plt
import libs.common.img_pair_interface as libimg
import libs.collection.prep_utility as prep
import PIL.Image as pilm
def progress():
    if not osp.exists(data_c.data_info):
        com_lines=[]
    else:
        loaded=np.load(data_c.data_info)
        com_lines=loaded['com_lines']
    json_files=libfi.getfilesbyext(data_c.data_path,'json')
    for j_file in json_files:
        r=libfi.path_search(j_file,'({})'.format('|'.join(list(slice_c.key_matcher.keys()))))
        if r is None:
            continue
        class_label=slice_c.class_label[slice_c.key_matcher[r[0]]]
        file_id=libfi.getfilename(j_file)
        tar_dir='{model_data_path}/{class_label}/{file_name}'.format(class_label=class_label,file_name=file_id,model_data_path=data_c.model_data_path)
        if not libfi.exist(tar_dir):
            libfi.make_dir(tar_dir)
        com_line='labelme_json_to_dataset {src_file} -o {tar_folder}'.format(
            src_file=j_file,tar_folder=tar_dir)
        if com_line not in com_lines:
            com_lines.append(com_line)
            print(com_line)
            os.system(com_line)
    # np.savez_compressed(data_c.data_info,com_lines=com_lines)
    print('='*100)
    data_list=libfi.getfilesbyext(data_c.model_data_path,'png|txt')
    data_dics={}
    for f in data_list:
        folder,file_name,file_ext=libfi.getfiledetails(f)
        class_label,file_id=folder.split('/')[-2:]
        id ='{class_label}-{file_id}'.format(class_label=class_label,file_id=file_id).lower()   
        if id not in data_dics:
            data_dics[id]=libimg.data_pair(['label_names','img','label'])
        if file_name in ['label_viz']:
            continue
        if file_name=='label_names':
            file=open(f)
            'x'.strip('\n')
            data_dics[id].label_names=list(file.readlines())
            data_dics[id].label_names=[x.strip('\n') for x in data_dics[id].label_names]
            file.close()
        elif file_name=='img':
            data_dics[id].img=f
        elif file_name=='label':
            data_dics[id].label=f
        else:
            print(f)

        print('processed:',f)
    # np.savez_compressed(data_c.data_info,com_lines=com_lines,data_dics=data_dics)
    print('Generated Data Info File:%s'%(data_c.data_info))
    print('='*100)
    layer_dict=filte_layer_labels(data_dics)
    np.savez_compressed(data_c.data_info,com_lines=com_lines,data_dics=data_dics)
    np.savez_compressed(data_c.data_layer_info,layer_dict=layer_dict)
    print('normalize labels')
    print('='*100)
    normalize_label(layer_dict,layer_c.key_matcher,layer_c.label_dir)

def filte_layer_labels(data_dicts):
    result_dicts={}
    for k in data_dicts:
        label_img=prep.imread(data_dicts[k].label,c=0)
        if len(set(label_img.flatten()))<layer_c.layer_num-1:
            continue
        i=0
        for l in layer_c.key_matcher.keys():
            if l in data_dicts[k].label_names:
                i=i+1

        if i >=layer_c.layer_num-2:
            print(k)
            result_dicts[k]=data_dicts[k]
    return result_dicts        

        
def normalize_label(dicts,matchers,output_dir):
    num=max(matchers.items(),key=lambda x:x[1])[1]
    if not libfi.exist(output_dir):
        libfi.make_dir(output_dir)
    for key in dicts:
        item=dicts[key]
        label_img=np.asarray(pilm.open(item.label))
        r_label=np.zeros(shape=label_img.shape[0:2],dtype=np.uint8)
        for num,s in enumerate(item.label_names):
            if s in matchers:
                r_label[label_img==num]=matchers[s]
        print('{dir}/{key}.{ext}'.format(dir=output_dir,key=key,ext='png'))
        prep.imwrite('{dir}/{key}.{ext}'.format(dir=output_dir,key=key,ext='png'),r_label)
