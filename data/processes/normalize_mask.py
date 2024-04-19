from .data_process import DataProcess
import math
import numpy as np
import cv2 as cv
#for visual debug
from PIL import Image
#for visual debug
class MapMask(DataProcess):
       def process(self, data:dict):
            # data = {'image':img,'gt':mask[0],'mask':mask[1],'thresh_map':mask[2],'punish':mask[3],'thresh_mask':mask[4],'c_size':c_size} #thresh_mask will be generated by make_fit_padding
            c_mask = data.pop('c_mask')
            data['gt'] = np.expand_dims(a = c_mask[0] , axis = 0).astype(np.float32)
            data['mask'] = c_mask[1].astype(np.float32)
            data['thresh_map'] = c_mask[2].astype(np.float32)
            data['punish'] =  np.expand_dims(a = c_mask[3] , axis = 0).astype(np.float32)
            data['thresh_mask'] = c_mask[4].astype(np.float32)
            return data

       
         
                
               
            
           