from concern.config import Configurable, State
import os
import torch
#for single gpu or cpu model load
from collections import OrderedDict


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        logger.info("Resuming from " + self.resume)
        #for multiple gpus or cpus model load
        # state_dict = torch.load(self.resume, map_location=device)
        # model.load_state_dict(state_dict, strict=False)
        #for single gpu or cpu model load
        states = torch.load(
            self.resume, map_location = device)
        new_states = OrderedDict()
        for k,v in (states or {}).items():
                name = k[:6]+k[13:] #k[6:13] 
                new_states[name] = v
        model.load_state_dict(new_states, strict=True)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
