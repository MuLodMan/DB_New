import torch.utils.data as data
from concern.config import Configurable, State
from data.data_control import TrainData
import os


#only for debug
from visual_debug import visual_featuremap
#only for debug

import json

class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        if cmd.get('demo',False):
            self.__available = False
            return
        self.__available = True
        self.__is_training = False
        self.load_all(**kwargs) #载入和初始化相关的类
        if type(data_dir) == list and type(data_list) == list:
            self.data_dir = [os.path.join(*ed_dir) for ed_dir in data_dir]
            self.data_list = [os.path.join(*el_dir) for el_dir in data_list]
        else:
             self.data_dir = self.data_dir
             self.data_list = self.data_list
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.__training_data = TrainData()
        if 'train' in self.data_list[0]:
            groupLevel = cmd.get('groupLevel',-1)
            assert groupLevel > -1
            self.__is_training = True
            self.get_train_samples(groupLevel)
        elif 'test' in self.data_list[0]:
            self.get_test_samples()
        else:
            raise Exception()

    def get_train_samples(self,groupLevel):
        for i in range(len(self.data_dir)): #file list array
            with open(self.data_list[i], mode='r',encoding='utf-8') as fid:
                    file_json_list = json.load(fid)
                    cur_list = file_json_list['img_list'][groupLevel]
                    self.imgl = imgl = cur_list.get('imgl',None)
                    self.c_size = cur_list.get('c_size',None)
                    assert type(imgl) == list

    def get_test_samples(self):
        image_paths = self.image_paths
        gt_paths = self.gt_paths
        for i in range(len(self.data_dir)): #file list array
            with open(self.data_list[i], mode='r',encoding='utf-8') as fid:
                if 'icdar2019' in self.data_list[i]:
                    file_json_list = json.load(fid)
                    for file_str in file_json_list['img_list']:
                        re_img_path = os.path.join(self.data_dir[i],'test_images',file_str.strip())
                        re_ann_path = os.path.join(self.data_dir[i],'test_gts',file_str.strip().split('.')[0]+'.json')
                        image_paths.append(re_img_path)
                        gt_paths.append(re_ann_path)
                elif 'icdar2015' in self.data_list[i]:
                    image_list = fid.readlines()
                    for file_str in image_list:
                        re_img_path = os.path.join(self.data_dir[i],'test_images',file_str.strip())
                        re_ann_path = os.path.join(self.data_dir[i],'test_gts',file_str.strip().split('.')[0]+'.txt')
                        image_paths.append(re_img_path)
                        gt_paths.append(re_ann_path)
                    self.targets = self.load_point_ann()
        
    def load_point_ann(self):
            res = []
            if 'icdar2019' in self.data_dir[0]: #icdar2019
                for gt in self.gt_paths:
                    lines = []
                    with open(gt,mode='r',encoding='utf-8') as raw_str:
                        ann_dict = json.load(raw_str)
                        ann_lines = ann_dict['lines']
                        for ann_line in ann_lines:
                            item = {}
                            points = ann_line['points']
                            temp_points = []
                            for p_i in range(0,len(points),2):
                                temp_points.append([int(points[p_i]),int(points[p_i+1])]) #[x,y]
                            item['poly'] = temp_points
                            if ann_line['ignore']: item['text'] = '###' #ignore
                            else: item['text'] = ann_line['transcription']
                            lines.append(item)
                    res.append(lines)
                return res

            elif 'icdar2015' in self.data_dir[0]:
                    for gt in self.gt_paths:
                        lines = []
                        with open(gt,mode='r',encoding='utf-8') as raw_str:
                            ann_lines = raw_str.readlines()
                            for ann_line in ann_lines:
                                item = {}
                                parts = ann_line.strip().strip('\ufeff\xef\xbb\xbf').split(',')
                                label = parts[-1]
                                poly = []
                                for p_i in range(0,len(parts[:8]),2):
                                    poly.append([int(parts[p_i]),int(parts[p_i+1])]) #[x,y]
                                item['poly'] = poly
                                item['text'] = label
                                lines.append(item)
                        res.append(lines)
                    return res     

        
    def __getitem__(self, index):
        data_dir = self.data_dir[0]
        if self.__is_training:
           img_dir = [data_dir , 'train_images']
           mask_dir = [data_dir , 'train_mask']
           data = self.__training_data.load_data(img_dir=img_dir,
                                                 mask_dir = mask_dir,
                                                 file_name=self.imgl[index],
                                                 c_size = self.c_size,
                                                 processes=self.processes)
           return data
        else:
           return data

    def __len__(self):
       if self.__available == False:
           return 1
       if self.__is_training:
          return len(self.imgl)
       else:
          return 1
