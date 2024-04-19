import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np

from concern.config import State
from .data_process import DataProcess
import cv2
import math
#for visual debug
from PIL import Image

class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args:list, root=True):
        method_list = []
        for arg in args:
          method = getattr(iaa,arg['method'])
          paras =  arg['paras']
          if type(paras) == dict:
             keys = paras.keys()
             for key in keys:
                 paras[key] = self.to_tuple_if_list(paras[key]) 
             method_list.append(method(**paras))
          elif type(paras) == list:
            tparas = self.to_tuple_if_list(paras)
            method_list.append(method(tparas))
          elif isinstance(paras,(int,str,float)):
             method_list.append(method(paras))
        return iaa.Sequential(method_list)

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj
            

class AugmentData(DataProcess):
    augmenter_args = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.keep_ratio = kwargs.get('keep_ratio')
        self.only_resize = kwargs.get('only_resize')
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        #for visual debug
        # punish = Image.fromarray(data['c_mask'][3]*80,mode='L')

        # pil_img = Image.fromarray(obj = data['image'],mode = 'RGB')
        #for visual debug end

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True 
        else:
            data['is_training'] = False 
        return data
    
class AugumentMap(AugmentData):
      def may_augment_annotation(self, aug, data,shape):
          if aug is None:
              return data
          
          data['c_mask'] = self.may_augment_mask(aug,data,shape)

          

      def may_augment_mask(self,aug,data,shape):
          

          t_mask = np.transpose(a = data['c_mask'] , axes = (1 , 2 , 0))

          segmap = SegmentationMapsOnImage(arr = t_mask , shape = shape )

          aug_segments = aug.augment_segmentation_maps(segmaps = segmap)

          au_t_mask = np.transpose(a = aug_segments.get_arr() , axes = (2,0,1))

          return au_t_mask
           
    

class AugmentDetectionData(AugmentData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

