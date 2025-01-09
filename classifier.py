import torch
import torchaudio as ta
import torch.nn.functional as F
import torchaudio.functional as taf
from model import DrumClassifierModel
from omegaconf import OmegaConf

class DrumClassifier():
    def __init__(self, weights_path=None, cfg_path="cfg.yaml"):
        if weights_path is None:
            raise Exception(f"DrumClassifier: Missing model weights path")
        try:
            self.cfg = OmegaConf.load(cfg_path)
        except:
            raise Exception("DrumClassifier: Missing or invalid configuration file. If left empty, cfg_path is assumed to be 'cfg.yaml'")
        self.model = DrumClassifierModel(cfg=self.cfg)
        try:
            self.model.load_state_dict(torch.load(weights_path))
        except:
            raise Exception(f"DrumClassifier: Missing or invalid model weights file at {weights_path}")
        self.tg_dict = {target:i for i,target in enumerate(self.cfg.train.targets)}
        self.tg_num = len(self.cfg.train.targets)
        self.sr = self.cfg.audio.sr
        self.segment_l = self.cfg.audio.max_l
        self.tg_len = self.segment_l * self.sr

    def prepare_input(self, x, batched=False, sr=None):
        if sr is not None and sr!=self.sr:
            x = taf.resample(x, sr, self.sr)
        #account for mono content
        if not batched:
            x = x.unsqueeze(0)
        isMono = x.shape[1]==1
        if isMono:
            if x.shape[1]>2:
                x = x[:,:2,:]
            else:
                x = x.repeat(1,2,1)
        if(x.shape[-1]<self.tg_len):
            diff = self.tg_len - x.shape[-1]
            x = F.pad(x, (0, diff))
        elif(x.shape[-1]>self.tg_len):
            x = x[:,:,:self.tg_len]
        return x
    
    def classify(self, x, sr=None, output_pred=False, output_text_label=False):
        #x: [B,C,T] or [C,T]. If you don't provide sr, it will be assumed to be cfg.audio.sr
        x = self.prepare_input(x, batched=(len(x.shape)==3),sr=sr)
        with torch.inference_mode():
            pred = self.model(x)
            if output_pred:
                return pred
            y = pred.squeeze().argmax().item()
        if output_text_label:
            return self.cfg.train.targets[y]
        return y

    def classify_from_path(self, file_path,output_pred=False, output_text_label=False):
        x, Fs = ta.load(file_path)
        return self.classify(x,sr=Fs,output_pred=output_pred,output_text_label=output_text_label)