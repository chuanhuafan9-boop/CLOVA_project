'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from tools.blip_vqa.vit import VisionTransformer, interpolate_pos_embed
from tools.blip_vqa.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,   
                 init_tokenizer_path = None              
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer(init_path = init_tokenizer_path)   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 init_tokenizer_path = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer(init_path=init_tokenizer_path)   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
        
    # def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
    #     # 1. 编码图像
    #     image_embeds = self.visual_encoder(image)
    #     batch_size = image.size(0)  # 获取 batch_size
    #
    #     if not sample:
    #         image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
    #
    #     image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
    #     model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
    #     # 2. 准备 Prompt
    #     prompt = [self.prompt] * image.size(0)
    #     input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
    #     input_ids[:,0] = self.tokenizer.bos_token_id
    #     input_ids = input_ids[:, :-1]
    #
    #     # 3. 生成 (分为采样模式和Beam Search模式)
    #     if sample:
    #         #nucleus sampling
    #         outputs = self.text_decoder.generate(input_ids=input_ids,
    #                                               max_length=max_length,
    #                                               min_length=min_length,
    #                                               do_sample=True,
    #                                               top_p=top_p,
    #                                               num_return_sequences=1,
    #                                               eos_token_id=self.tokenizer.sep_token_id,
    #                                               pad_token_id=self.tokenizer.pad_token_id,
    #                                               repetition_penalty=1.1,
    #                                               # --- 新增参数 ---
    #                                               output_scores=True,  # <--- 1. 开启分数输出
    #                                               return_dict_in_generate=True,  # <--- 2. 返回字典格式
    #                                               # ----------------
    #                                               **model_kwargs)
    #     else:
    #         #beam search(VQA通常走这里)
    #         outputs = self.text_decoder.generate(input_ids=input_ids,
    #                                               max_length=max_length,
    #                                               min_length=min_length,
    #                                               num_beams=num_beams,
    #                                               eos_token_id=self.tokenizer.sep_token_id,
    #                                               pad_token_id=self.tokenizer.pad_token_id,
    #                                               repetition_penalty=repetition_penalty,
    #                                               # --- 新增参数 ---
    #                                               output_scores=True,  # <--- 1. 开启分数输出
    #                                               return_dict_in_generate=True,  # <--- 2. 返回字典格式
    #                                               # ----------------
    #                                               **model_kwargs)
    #
    #         # ==================== [核心修改：收集概率数据] ====================
    #         # 初始化一个结构来存储详细概率： [Batch_Size, Steps, Top_K]
    #         # 注意：如果是 Beam Search，outputs.scores 的维度通常包含 beam，这里我们只取第一个 beam (最简单的处理)
    #
    #         batch_detailed_scores = [[] for _ in range(batch_size)]  # 为每个样本创建一个空列表
    #
    #         print(f"\n======== [BLIP VQA] 答案生成概率详情 ========")
    #
    #         # 遍历生成的每一个时间步 (Step)
    #         for step_i, step_scores in enumerate(outputs.scores):
    #             # step_scores 形状: (batch_size * beams, vocab_size)
    #
    #             # 计算该步所有词的概率
    #             probs = torch.softmax(step_scores, dim=-1)
    #
    #             # 遍历 batch 中的每一张图
    #             for b_i in range(batch_size):
    #                 # 确定当前样本在 logits 中的索引
    #                 # 如果是 Beam Search，每个样本有 num_beams 个结果，我们只取第 1 个 (最好的那个)
    #                 # 如果是 Sample，num_beams 通常为 1
    #                 if not sample:
    #                     score_idx = b_i * num_beams
    #                 else:
    #                     score_idx = b_i
    #
    #                 # 获取当前样本、当前步的概率分布
    #                 current_probs = probs[score_idx]
    #
    #                 # 取 Top 5
    #                 top_k = 5
    #                 top_probs, top_indices = torch.topk(current_probs, top_k)
    #
    #                 # 存入临时列表
    #                 step_info = []
    #
    #                 # 只有打印第0个样本时才输出到控制台(避免刷屏)，但所有数据都会被存下
    #                 if b_i == 0:
    #                     print(f"--- Step {step_i + 1} (Batch 0) ---")
    #
    #                 for k in range(top_k):
    #                     token_id = top_indices[k].item()
    #                     prob = top_probs[k].item()
    #                     word = self.tokenizer.decode([token_id])
    #
    #                     # 存数据：字典格式
    #                     step_info.append({"word": word, "prob": prob})
    #
    #                     if b_i == 0:
    #                         print(f"  排名 {k + 1}: '{word}' (概率: {prob:.4f})")
    #
    #                 # 将这一步的 Top-K 信息加入到对应样本的结果中
    #                 batch_detailed_scores[b_i].append(step_info)
    #
    #         print("===================================================\n")
    #
    #         # 1. 恢复 outputs 格式以便解码
    #         outputs = outputs.sequences
    #
    #         # 2. [核心修改] 将 文本 和 分数 打包到 captions 列表里
    #         captions = []
    #         for i, output in enumerate(outputs):
    #             # A. 解码文本 (保持原来的逻辑)
    #             text_full = self.tokenizer.decode(output, skip_special_tokens=True)
    #             # 去掉 prompt 部分，只保留生成的答案
    #             answer_text = text_full[len(self.prompt):]
    #
    #             # B. 获取对应的分数数据
    #             # 这里的 i 对应 batch 中的第 i 张图
    #             score_data = batch_detailed_scores[i]
    #
    #             # C. 封装成字典
    #             # 以前是: captions.append(answer_text)
    #             # 现在改为:
    #             captions.append({
    #                 "caption": answer_text,  # 答案文本 (例如 "yes")
    #                 "scores": score_data  # 概率详情 (例如 [{'word': 'yes', 'prob': 0.9}, ...])
    #             })
    #
    #         # 3. 返回修改后的 captions 列表
    #         return captions

    #原来的 generate 函数
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        # 1. 编码图像
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        # 2. 准备 Prompt
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        # 3. 生成 (分为采样模式和Beam Search模式)
        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search(VQA通常走这里)
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer(init_path = None):
    tokenizer = BertTokenizer.from_pretrained(init_path)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
