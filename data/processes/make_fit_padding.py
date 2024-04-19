from .data_process import DataProcess
import math
import numpy as np
import cv2 as cv
#for visual debug
from PIL import Image
#for visual debug
class make_padding(DataProcess):
       def process(self, data:dict):
          total_mask = data.get('c_mask')
          img = data.get('image')
          self.max_size =  960 #32*30
          
          (Height,Width) = img.shape[:2]

          canva_size = data.get('c_size')
          
          
          padded_h = canva_size - Height
          padded_w = canva_size - Width
          top = math.ceil(padded_h / 2)
          bottom = padded_h - top
          left = math.ceil(padded_w / 2)
          right = padded_w - left 

          top = int( top )
          bottom = int( bottom )
          left = int( left )
          right = int( right )
          padded_image =  cv.copyMakeBorder(src = img,top = top,bottom = bottom,left = left,right = right,borderType=cv.BORDER_CONSTANT,value=0)

          padded_gt = cv.copyMakeBorder(src = total_mask[0] , top = top , bottom = bottom , left = left , right = right , borderType=cv.BORDER_CONSTANT,value=0) 

          padded_mask = cv.copyMakeBorder(src = total_mask[1],top=top,bottom=bottom,left=left,right=right,borderType=cv.BORDER_CONSTANT,value=0)

          padded_thresh = cv.copyMakeBorder(src = total_mask[2],top=top,bottom=bottom,left=left,right=right,borderType=cv.BORDER_CONSTANT,value=0) 

          padded_punish_mask = cv.copyMakeBorder(src = total_mask[3],top=top,bottom=bottom,left=left,right=right,borderType=cv.BORDER_CONSTANT,value=0)

          self.size_Limit(canva_size = canva_size,
                          padded_image = padded_image,
                          padded_gt=padded_gt,
                          padded_mask=padded_mask,
                          padded_thresh=padded_thresh,
                          padded_punish_mask= padded_punish_mask,
                          data=data
                          )
          return data
       
       def size_Limit(self,canva_size,
                      padded_image:np.ndarray,
                      padded_gt:np.ndarray,
                      padded_mask:np.ndarray,
                      padded_thresh:np.ndarray,
                      padded_punish_mask:np.ndarray,
                      data:dict):
           new_compress_mask = None
           if canva_size > self.max_size:
               new_compress_mask = np.zeros(shape=(5 , self.max_size , self.max_size),dtype = np.uint8)
               data['image'] = cv.resize(src = padded_image,dsize=(self.max_size,self.max_size),interpolation=cv.INTER_AREA)#image
               new_compress_mask[0] = cv.resize(src = padded_gt,dsize=(self.max_size,self.max_size),interpolation=cv.INTER_NEAREST)#gt
               new_compress_mask[1] = cv.resize(src = padded_mask,dsize=(self.max_size,self.max_size),interpolation=cv.INTER_NEAREST)#mask
               new_compress_mask[2] = cv.resize(src = padded_thresh,dsize=(self.max_size,self.max_size),interpolation=cv.INTER_NEAREST)#thresh_map
               new_compress_mask[3] = cv.resize(src = padded_punish_mask , dsize=(self.max_size,self.max_size) , interpolation=cv.INTER_NEAREST)#punish
           else:
               new_compress_mask = np.zeros(shape=(5 , canva_size , canva_size),dtype = np.uint8)
               data['image'] = padded_image
               new_compress_mask[0] = padded_gt
               new_compress_mask[1] = padded_mask
               new_compress_mask[2] = padded_thresh
               new_compress_mask[3] = padded_punish_mask

           new_compress_mask[4] = new_compress_mask[2]  + new_compress_mask[0] #thresh_mask
           data['c_mask'] = new_compress_mask

         #punish_img = Image.fromarray((data['punish']*80).astype(np.uint8),mode='L')
         #punish_img.show()
                
               
            
           