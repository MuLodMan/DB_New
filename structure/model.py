import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {})) #ResNet,avgpool,fc smooth backbone_args meaningless in resnet yaml
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

        #only for debug
        # print(self.decoder)
        #only for debug

    def forward(self, data, *args, **kwargs): 
        # if (not self.__checked_graph) :#for debug
        #    SummaryWriter(log_dir="../samples/encoder_decoder_graph_resnet50").add_graph(nn.Sequential(self.backbone,self.decoder),input_to_model=data) #for debug
        #    self.__checked_graph = True
        return self.decoder(self.backbone(data))


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return model

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])
    

    def forward(self, batch):
        punish = None
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
            # if self.training: 
            #     punish = batch.pop('punish')
            #     self.model.decoder.set_punish_map(punish,self.device) #for punish threshold
        else:
            data = batch.to(self.device)
        pred = self.model(data)  #forward step for debug

        if self.training:
            thresh_punish = self.set_punish_thresh_map(punish=punish)
            batch['thresh_map'] = batch['thresh_map'] #* thresh_punish #for punish threshold
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred
    def set_punish_thresh_map(self,punish:torch.Tensor):
        '''
        set punish for thresh
        '''
        (b,c,h,w) = punish.shape
        punish_map = torch.zeros(size=(b,c,h,w),dtype=torch.float32)
        punish_map[punish == 2] = 1
        punish_map[punish == 1] = 1.25
        punish_map[punish == 3] = 0.75
        return punish_map
           # for bi in range(b):
        #     for ci in range(c):
        #         for hi in range(h):
        #             for wi in range(w):
        #                 curmask = punish[bi][ci][hi][wi]
        #                 if curmask == 2:
        #                     self.__punish_map[bi][ci][hi][wi] = 0.875
        #                 if curmask == 1:
        #                     self.__punish_map[bi][ci][hi][wi] = 0.75
        #                 if curmask == 3:
        #                     self.__punish_map[bi][ci][hi][wi] = 1.25
#        self.__unpunished = 2#0.875
#        self.__punished_weak = 1#0.75
#        self.__punished_aug = 3#1.125