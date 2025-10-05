# authors: @shaunxsyang
# times  : @2025/06/25
from typing import Sequence, Optional, Union

import math
import random

import torch
import torch.nn as nn
import numpy as np

# from tools.tokenizer.soundstream.modules.seanet import SEANetEncoder, SEANetDecoder
# from tools.tokenizer.soundstream.quantization  import ResidualVectorQuantizer

from tools.tokenizer.hytokenize.modules.wavlm import WavLMTeacher
from tools.tokenizer.hytokenize.modules.modules import ProjectorMLP
from tools.tokenizer.hytokenize.modules.mimi import MimiCodecEncWithFactorVQ, MimiCodecDec
from tools.tokenizer.hytokenize.modules.utils import get_mask_from_lengths
from collections import OrderedDict




def exists(val):
    return val is not None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(BaseModel, self).__init__()
        self.config = config
        self.integrity_test()

    def forward(self, feature):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, feature):
        return self.forward(feature)

    def remove_weight_norm(self):
        pass

    def integrity_test(self):
        pass

class TokenizerGANWrapper(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.tokenizer = MimiCodecEncWithFactorVQ(**config.tokenizer_conf)
        self.token2wav = MimiCodecDec(**config.wave_decoder_conf)
        self.wav_input_sr = config.get("wav_input_sr", 24000)
        self.trainable_module = ["tokenizer", "token2wav"]
        
        # distill related
        self.use_distill_branch = config.get("use_distill_branch", False)
        if hasattr(config, "teacher_type") and self.use_distill_branch:
            self.teacher = WavLMTeacher(**config.teacher_conf)
            config.distill_learner_conf["output_dim"] = self.teacher.output_dim
            self.distill_learner = ProjectorMLP(**config.distill_learner_conf)
            self.trainable_module += ["distill_learner"]
            distill_criterion_conf = config.get("distill_criterion_conf", {})
            self.distill_criterion = eval(config.distill_criterion)(config, **distill_criterion_conf)
            
            for name, param in self.teacher.named_parameters():
                param.requires_grad = False
            
    def preprocess_data(self, x):
        wav_input = x
        x = wav_input.squeeze(1)
        tokenizer_out = self.tokenizer(
            x,
            input_sample_rate = self.wav_input_sr,
        )
        quantized = tokenizer_out["quantized"]
        vq_loss = tokenizer_out["vq_loss"]
        vq_loss_items = tokenizer_out["vq_loss_items"]
        
        processed_res = {
            "token": quantized,
            "vq_loss": vq_loss,
            "vq_loss_items": vq_loss_items,
        }
        
        if self.use_distill_branch:
            with torch.no_grad():
                distill_targets = self.teacher(x, input_sample_rate = self.wav_input_sr)["regressed_feats"]
                processed_res["distill_targets"] = distill_targets
        
        return processed_res
        
    def forward(self, x):
        processed_data = self.preprocess_data(x)
        token = processed_data["token"]
        res = {}
        ret = self.token2wav(token, return_hidden_feat=self.use_distill_branch)
        if self.use_distill_branch:
            distill_targets = processed_data["distill_targets"]
            res["sample_seg"], hidden_feat = ret
            if self.distill_learner.input_type == "token":
                distill_preds, distill_pred_lens = self.distill_learner(token)
            else:
                distill_preds, distill_pred_lens = self.distill_learner(hidden_feat)
            res["distill_loss"], res["distill_loss_items"] = self.distill_criterion(distill_targets.detach(), distill_preds, distill_pred_lens)
        else:
            res["sample_seg"] = ret
            
        if "vq_loss" in processed_data:
            res["vq_loss"] = processed_data["vq_loss"]
            res["vq_loss_items"] = processed_data["vq_loss_items"]
            
        return res
    
    @torch.no_grad()
    def preprocess_infer_data(self, x):
     
        wav_input = x # [YXS] (1, 1, len)

        assert exists(wav_input)
        assert wav_input.size(0) == 1, f"Only support batch_size == 1 when inference"
        if wav_input.dim() == 2:
            wav_input = wav_input.unsqueeze(1)
        tokenizer_out = self.tokenizer(wav_input.squeeze(1), 
                                       input_sample_rate=self.wav_input_sr, 
                                       apply_random_quantization=False)
        token = tokenizer_out["quantized"]
        indicies = tokenizer_out["tokens"]
        if "token_lens" in tokenizer_out:
            token_lens = tokenizer_out["token_lens"][0].item()
            indicies = indicies[:token_lens]
        before_quantize = tokenizer_out["before_quantize"]
        
        indicies_mask = indicies 
        codes = indicies_mask[:,:,-1]=0
        codes_to_quantized = self.tokenizer.vq.vq2emb(codes.premute(1, 2, 0))
        codes_to_quantized = codes_to_quantized.transpose(1, 2).masked_fill(~self.mask.unsqueeze(1), 0.0) 
        
        res = {
            "token": token,
            "before_quantize": before_quantize,
            "indicies": indicies,
            "codes_to_quantized": codes_to_quantized,
        }
        
        return res
    
    @eval_decorator
    @torch.no_grad()
    def inference(self, x): # [YXS] (1, 1, len)
        processed_data = self.preprocess_infer_data(x)
        token = processed_data["token"]
        before_quantize = processed_data["before_quantize"]
        indicies = processed_data["indicies"]

        audio = self.token2wav(token, return_hidden_feat=False)
        
        dec_hidden_feat = None
        if isinstance(audio, tuple):
            audio, dec_hidden_feat = audio

        return audio, indicies


    # [YXS] TODO modify encode -- code -- decode
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        tokenizer_out = self.tokenizer(
            x.squeeze(1),
            input_sample_rate = self.wav_input_sr,
        )
        quantized = tokenizer_out["quantized"]
        vq_loss = tokenizer_out["vq_loss"]
        vq_loss_items = tokenizer_out["vq_loss_items"]
        
        # [YXS]
        token_lens = tokenizer_out["token_lens"]
        indicies_mask = tokenizer_out["tokens"]
        before_quantize = tokenizer_out["before_quantize"] 
        
        # [YXS] second indice all -1 = 0
        codes = torch.where(indicies_mask == -1, 0, indicies_mask)
        self.mask = get_mask_from_lengths(token_lens, max_len=before_quantize.transpose(1, 2).shape[-1])

        return codes  # [n_codebook, 1, n_frames]
    

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # [YXS] 
        codes_to_quantized = self.tokenizer.vq.vq2emb(codes.permute(1, 2, 0))
        # codes_to_quantized = codes_to_quantized.transpose(1, 2).masked_fill(~self.mask.unsqueeze(1), 0.0)
        # import ipdb; ipdb.set_trace()
        # print('check codes:\n', codes, '\n', codes_to_quantized)
        o = self.token2wav(codes_to_quantized, return_hidden_feat=False)
        return o

# test
if __name__ == '__main__':
    import os 
    import json5
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
    
    config_path = f'/apdcephfs/share_302533218/shaunxsyang/speech_projects/HY-GPT4O-TTS-mimi/job/exp_Hz/exp_5hz_RVQ8_c16384_30wh/config.25-06-13.json'
    if not os.path.isfile(config_path):
        raise ValueError(f"{config_path} file does not exist.")
        # [YXS] need to compare config
    with open(config_path, "r") as f:
        data = f.read()
    basic_config = json5.loads(data)
    config = basic_config
    hps = JsonHParams(**config)
    import ipdb; ipdb.set_trace()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
    hy_tokenize = TokenizerGANWrapper(hps.model).to(device)
    for i in range(10):
        print(f"Iter {i}: ")
        wav_path = '/apdcephfs/share_302533218/shaunxsyang/speech_projects/LibriSpeech/test-clean/8230/279154/8230-279154-0026.wav'
        import librosa
        audio, data_sr = librosa.load(wav_path, sr=16000, mono=True)
        import ipdb; ipdb.set_trace()

        x = torch.from_numpy(audio.reshape(1, -1)).unsqueeze(0).float().to('cuda')
        # x = torch.rand(1, 1, 16000)
        o = hy_tokenize(x)["sample_seg"] 
        print('output', o.shape)
        codes = hy_tokenize.encode(x)
        import ipdb; ipdb.set_trace()
        o_ = hy_tokenize.decode(codes)
        print('output', o_.shape)
