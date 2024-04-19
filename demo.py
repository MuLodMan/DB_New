#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
#for single gpu or cpu model load
from PIL import Image
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='.\\demo_results\\', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    
    #for debug to choose device
    parser.add_argument('--devices',type=int , default=1,help='set gpu nums default is 1')

    #for debug insight
    # parser.add_argument('--an')
    #for debug insight

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    args['demo'] = True

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        self.structure = experiment.structure
        self.model_path = self.args['resume']
        self.devices = self.args['devices']
        self.ori_cv_image = None

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_dtype(torch.float32)#'torch.FloatTensor'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        new_states = OrderedDict()
        if self.devices <= 1:  #False is for load single gpu model
            for k,v in (states or {}).items():
                name = k[:6]+k[13:] #k[6:13] 
                new_states[name] = v
        else:
            new_states = states
        model.load_state_dict(new_states, strict=True)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        self.new_width = new_width
        self.new_height = new_height
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        # height, width, _ = img.shape
        # padded_h = 32 - height % 32
        # padded_w = 32 - width % 32
        # self.board_top = top = math.ceil(padded_h / 2)
        # self.board_bottom = bottom = padded_h - top
        # self.board_left = left = math.ceil(padded_w / 2)
        # self.board_right = right = padded_w - left
        # return cv2.copyMakeBorder(src = img,top = top,bottom = bottom,left = left,right = right,borderType=cv2.BORDER_CONSTANT,value=0) 
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        self.ori_cv_image = img
        img = img.astype('float32')
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('\\')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch) 
            #for save shrink_punish
            # self.save_heatmap(predict=pred)
            #for show shrink_punish
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('\\')[-1].split('.')[0]+'.jpg'), vis_image)
        return self
    


    #for better visualize heatmap on image
    def save_heatmap_on_image(self,predict:torch.Tensor):
        if len(predict.shape) < 4 :raise Exception('bad aug')
        heat_map = predict[0].permute(1,2,0)
        np_h_m =heat_map.to(device = torch.device('cpu')).numpy()
        norm_h_m = cv2.normalize(src = np_h_m, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(src =norm_h_m  , colormap=cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(src1 = self.ori_cv_image , alpha = 0.7, src2 = heatmap_color , beta=0.3 , gamma = 0 )
        cv2.imwrite("demo_results\\icdar2019\\thresh00042.png" , superimposed)

    def save_heatmap(self,predict:torch.Tensor):
        if len(predict.shape) < 4 :raise Exception('bad aug')
        norm_h_m = None
        heat_map = predict[0].permute(1,2,0)
        np_h_m =heat_map.to(device = torch.device('cpu')).numpy()
        #binary shrinked punished masked
        loaded = np.load('datasets\\icdar2019\\train_mask\\train_ReCTS_000042.npz')
        punish_mask = loaded['masks'][3]
        shrink_punish_map = self.set_punish_map(punish = punish_mask) 
        shrink_binary_punish = np_h_m * shrink_punish_map
        norm_h_m = cv2.normalize(src = shrink_binary_punish, dst = norm_h_m , alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        heatmapshow = cv2.applyColorMap(norm_h_m, cv2.COLORMAP_JET)
        cv2.imwrite("demo_results\\icdar2019\\thresh_binary_mask_00042.png" , heatmapshow)
        #binary shrinked punished masked

    def set_punish_thresh_map(self,punish:torch.Tensor):
        '''
        set punish for thresh
        '''
        punish = np.pad(punish , pad_width=((self.board_top , self.board_bottom) , (self.board_left , self.board_right)) , constant_values = 0)
        (h,w) = punish.shape
        punish_map = torch.zeros(size=(h , w ),dtype=torch.float32)
        punish_map[punish == 2] = 1
        punish_map[punish == 1] = 1.25
        punish_map[punish == 3] = 0.875
        return punish_map
    
    def set_punish_map(self,punish:np.ndarray):
        '''
        set punish for shrink
        '''
        punish = cv2.resize(src = punish,dsize=(self.new_width,self.new_height),interpolation=cv2.INTER_NEAREST)
        (h,w) = punish.shape
        punish_map = np.zeros(shape=(h , w),dtype=np.float32)
        punish_map[punish == 2] = 1#1
        punish_map[punish == 1] = 0.875
        punish_map[punish == 3] = 1.25
        return np.expand_dims(punish_map,axis = 2)

if __name__ == '__main__':
    main()
