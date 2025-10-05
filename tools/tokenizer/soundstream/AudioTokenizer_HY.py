# authors: @shaunxsyang
# times  : @2025/06/25

"""Command-line for audio compression."""
import os
import json5
import numpy as np

from pathlib import Path
import sys
import torchaudio
import os
import torch
import typing as tp
from collections import OrderedDict
from omegaconf import OmegaConf
import logging

from tools.tokenizer.soundstream.models.hy_tokenize import TokenizerGANWrapper
from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.common import codec_specaug

class HY_Tokenizer(AbsTokenizer):
    def __init__(self, 
                 device=torch.device('cpu'), 
                 clip_length=450):
        """ hunyuan(HY) tokenizer with low frame rate is fixed bandwidth of 1kbps ~ 5kbps.
            It encodes speech with 5 fps and 512-dim vector for each frame.
            The value of each entry is in [0, 16384] , [0, 8192], [0, 4] and so on.
        """
        super(HY_Tokenizer, self).__init__()
        # [original] GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU

        self.device = device

        config_path = f'/apdcephfs/share_302533218/shaunxsyang/speech_projects/HY-GPT4O-TTS-mimi/job/exp_Hz/exp_5hz_RVQ32_c256_30wh/config.25-08-09.json'
        if not os.path.isfile(config_path):
            raise ValueError(f"{config_path} file does not exist.")
        # [YXS] need to compare config
        with open(config_path, "r") as f:
             data = f.read()
        basic_config = json5.loads(data)
        config = basic_config
        hps = JsonHParams(**config) 
        # import ipdb; ipdb.set_trace() 
        # config = OmegaConf.load(config_path) # [YXS] TODO check?
        
        # self properties
        self.hps = hps
        self.sr = 16000
        self.dim_codebook = 256
        self.n_codebook = 32 # [YXS] TODO config different rvq layers
        self.bw = 1.5 # 
        self.freq = self.n_codebook * 5
        self.mask_id = self.dim_codebook * self.n_codebook

        # [YXS] codec model path
        self.ckpt_path = f'/apdcephfs/share_302533218/shaunxsyang/speech_projects/HY-GPT4O-TTS-mimi/job/exp_Hz/exp_5hz_RVQ32_c256_30wh/checkpoints_5Hz/generator-650000.pt'
        logging.info(f"using config {config_path} and model {self.ckpt_path}")
        # [YXS] load model paras
        self.hy_tokenize = self.build_codec_model(config)

    def build_codec_model(self, config):
        model = self.create_model()
        parameter_dict = self.load_state_dict()
        model = self.load_model(parameter_dict, model)
        print(f"Parameters : {num_params(model)} M".center(100,'='))
        model = model.to(self.device)
        # model.eval() # [YXS] add 
        return model
    
    def create_model(self):
        return get_model(self.hps.model.type, self.hps.model) # [YXS] TODO load model name
 
    def load_state_dict(self):
        print(f"Restore from {self.ckpt_path}".center(100, "="))
        parameter_dict = torch.load(self.ckpt_path, map_location='cpu')
        return parameter_dict
    
    def load_model(self, state_dict, model): # [YXS] for multi gpu 'module' name
        state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
        clean_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v
        model.load_state_dict(clean_dict)
        model.remove_weight_norm()
        return model

    def _flatten_codebooks(self, arr, offset_size=1024):
        assert len(arr.shape) == 2
        arr = arr.copy()
        if offset_size is not None:
            for n in range(arr.shape[0]):
                arr[n, :] += (offset_size * n)
        flat_arr = arr.ravel("F")
        return flat_arr # [YXS] (n_q * frames,), e.g. (1395,)






    def encode(self, wav_root, sr=16000):
        wav, sr = torchaudio.load(wav_root)

        if wav.numel() == 0:
            return None

        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)

        wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
        # import ipdb; ipdb.set_trace()
        compressed = self.hy_tokenize.encode(wav) # [n_codebook, 1, n_frames]
        compressed = compressed.squeeze(1).detach().cpu().numpy() # [n_codebook, n_frames]
        # print(f"debug, {wav_root}, {compressed}")
        flat_codec = self._flatten_codebooks(compressed, self.dim_codebook)
        # flat_codec = self._flatten_codebooks(compressed, None) # [YXS]
        flat_codec = torch.from_numpy(flat_codec)
        flat_codec = flat_codec.to(torch.int16)
        return flat_codec

    def detokenize(self, codes):
        assert codes.dim() == 1
        l = len(codes)
        # logging.info(f"debug, {l}, {codes}")
        assert len(codes) % self.n_codebook == 0
        
        codes = codes.view(-1, self.n_codebook).transpose(0, 1)
        for i in range(self.n_codebook):
            codes[i] -= i * self.dim_codebook
        out = self.hy_tokenize.decode(codes.long().to(self.device).unsqueeze(1))
        out = out.detach().cpu().squeeze(0)
        return out

    @property
    def is_discrete(self):
        return True

    def tokenize(self, wav, task=None, cache=None):

        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav)

        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                # Some on-the-fly process
                if task in ['asr'] and cache['is_train']:
                    wav = codec_specaug(
                        wav.view(-1, self.n_codebook).contiguous(),
                        mask_id=self.mask_id,
                    )
                return wav
            if wav.dim() == 2: # transfer to 3 dim
                if wav.numel() == 0:
                    return None
                wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
            compressed = self.hy_tokenize.encode(wav, target_bw=self.bw) # [n_codebook, 1, n_frames]
            compressed = compressed.squeeze(1).detach().cpu().numpy() # [n_codebook, n_frames]
            flat_codec = self._flatten_codebooks(compressed, self.dim_codebook)
            flat_codec = torch.from_numpy(flat_codec)
            flat_codec = flat_codec.to(torch.int16)
            return flat_codec
        else:
            raise NotImplementedError

    @property
    def codebook_length(self):
        return self.dim_codebook * self.n_codebook + 1

    def find_length(self, x):
        return self.tokenize(x).shape[0] // self.n_codebook


def get_model(model_name, config, **kwargs):
        return eval(model_name)(config, **kwargs)

def num_params(net) :
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1024 / 1024
    return parameters

class JsonHParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            if type(v) == str and v.lower() in ["non", "none", "nil", "null"]:
                v = None
            self[k] = v

    def to_dict(self):
        return self

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def pop(self, key):
        return self.__dict__.pop(key)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def set(self, key, value):
        return setattr(self, key, value)

    def exist(self, key):
        return hasattr(self, key)



if __name__ == '__main__':
    tokenizer = HY_Tokenizer(device=torch.device('cuda:0')).cuda()
    wav_path = '/apdcephfs/share_302533218/shaunxsyang/speech_projects/LibriSpeech/test-clean/8230/279154/8230-279154-0026.wav'
    # import ipdb; ipdb.set_trace()
    
    import librosa
    audio, data_sr = librosa.load(wav_path, sr=16000, mono=True)
    x_ = torch.from_numpy(audio.reshape(1, -1)).unsqueeze(0).float().to('cuda')
    o = tokenizer.hy_tokenize(x_)
    output = o["sample_seg"]
    print(output, output.shape)
    
    code = tokenizer.tokenize(wav_path)
    print(code.shape) # [YXS] (n_q * frames,)  
    wav = tokenizer.detokenize(code)
    print(wav, wav.shape) 
