from collections import OrderedDict

import torch
import numpy as np
from .grab_analysis import grab_analysis
from .data_process import DataProcess
import cv2 as cv
class MakeGrabCut(DataProcess):
    def __init__(self,debug=False,**kwargs):

        self.debug = debug
    
    def process(self, data:dict):
          padded_rects =  data.get('padded_rect',[])
          padded_polygons = data.get('padded_polygons',[])
          polygons = data.get('polygons',[])
          

          padding_Len = len(padded_rects)
          assert padding_Len == len(polygons) == len(padded_polygons)
          ori_image = data.get('image',None)
          grab_mask = np.zeros(shape=ori_image.shape[:2],dtype=np.float32)
          for idx in range(0,padding_Len):
                padded_rect = padded_rects[idx]
                padded_polygon = padded_polygons[idx]
                if len(padded_rect)>0:
                    (minX,minY,maxX,maxY) = padded_rect

                    paddedCropedImage = ori_image[minY:maxY,minX:maxX]

                    (paddedCropHeight,paddedCropWidth) = paddedCropedImage.shape[:2]

                    gt_relative_p = polygons[idx] - np.array([minX,minY])

                    padded_relative_p = padded_polygon - np.array([minX,minY])

                    grab_sub_mask = self.make_grab_cutMap(paddedHeight = paddedCropHeight , paddedWidth = paddedCropWidth , gr_points = gt_relative_p , image = paddedCropedImage,pr_polygon = padded_relative_p)

                    if grab_sub_mask is None:
                        grab_analysis.uneffective_grab += 1
                    else:
                       grab_analysis.effective_grab += 1
                       grab_mask[minY : maxY,minX : maxX] = grab_sub_mask


          self.holing_padded_map(data = data,grab_map = grab_mask)
          
          return data

    def holing_padded_map(self,data:dict,grab_map:np.ndarray):
        padded_map =  data.get('thresh_mask')
        shrink_map = data.get('gt')[0]
        if np.array_equal(padded_map.shape,shrink_map.shape) and np.array_equal(padded_map.shape,grab_map.shape):
            data['thresh_mask'] = (padded_map - shrink_map) * (1 - grab_map)
            data['thresh_punish_mask'] = grab_map#shrink_map * grab_map
        

    def make_grab_cutMap(self,paddedHeight:int,paddedWidth:int,gr_points:np.ndarray,image:np.ndarray,pr_polygon:np.ndarray):
         bgdModel = np.zeros((1,65),np.float64)
         fgdModel = np.zeros((1,65),np.float64)
      #  result_mask,result_bgdMask,reslut_fgdModel = cv.grabCut(img=image,mask=probility_mask,rect=None,bgdModel=bgdModel,fgdModel=fgdModel,iterCount=5,mode=cv.GC_INIT_WITH_MASK)
      #  return np.where(((result_mask==2)|(result_mask==0)),0,1).astype(np.uint8)
      #  cv.GC_BGD = 0,cv.GC_PR_FGD = 3,cv.GC_FGD = 1,cv.GC_PR_BGD=2
            
         line_mask = np.full(shape=(paddedHeight,paddedWidth),fill_value=cv.GC_PR_BGD,dtype=np.uint8)#np.full(shape=(paddedHeight,paddedWidth),fill_value=0,dtype = np.uint8) 

         cv.fillPoly(img = line_mask , pts = [np.round(pr_polygon).astype(np.int32)] , color = cv.GC_BGD)#

         cv.fillPoly(img = line_mask , pts = [np.round(gr_points).astype(np.int32)] , color = cv.GC_PR_FGD) #

         polygon_area = (line_mask == cv.GC_PR_FGD).sum()

         if polygon_area <= 0:
             return None  

         result_mask,result_bgdMoedel,reslut_fgdModel = cv.grabCut(img=image,mask=line_mask,rect=None,bgdModel=bgdModel,fgdModel=fgdModel,iterCount=20,mode=cv.GC_INIT_WITH_MASK)
         
         new_mask = np.where((result_mask == cv.GC_BGD)|(result_mask == cv.GC_PR_BGD),0,1).astype(np.float32)

        #  pr_fgd_mask = np.where(result_mask == cv.GC_PR_FGD, 1, 0)
        #  fgd_mask = np.where(result_mask == cv.GC_FGD, 1, 0)
        #  grab_area = fgd_mask.sum()  
         
         grab_area = new_mask.sum()
         grab_ratio = grab_area/polygon_area   

        #  print("The grab map ratio is "+ str(grab_ratio))

         if grab_ratio > 0.25 and grab_ratio < 0.9375:
            return new_mask
         else : return None 
    
