
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

#Import from the adapter version
from .llama_adapter_hf import LlamaForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..multimodal_projector.builder import build_vision_projector, build_vision_projector_aligner, build_vision_projector_fusion_adapter

from ..multimodal_encoder.builder import build_vision_tower

class AdaptiveLlamaVisionModule(nn.Module):

    def __init__(self, config, proj_config = None, adapter_qlen=None, inference_mode=False):
        super().__init__()
        self.config = config
        self.adapter_qlen = adapter_qlen
        # prevent it from init during training. training use initialize_vision_modules to set proper config
        if inference_mode:
            # Set proj_config when config is not provided
            self.proj_config = proj_config if proj_config is not None else config
            self.vision_tower = build_vision_tower(proj_config, delay_load=True)

            if hasattr(config, "mm_aligner_structured") and self.config.mm_aligner_structured:
                if self.adapter_qlen is not None:
                    self.mm_projector = build_vision_projector_fusion_adapter(self.config, self.config.mm_aligner_size, self.adapter_qlen)
                else:
                    self.mm_projector = build_vision_projector_aligner(self.config, self.config.mm_aligner_size)
            else:
                self.mm_projector = build_vision_projector(proj_config)
        

    def initialize_vision_modules(self, model_args, **kwargs):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_mm_mlp_adapter_size = model_args.pretrain_mm_mlp_adapter_size

        self.config.mm_aligner_structured = model_args.mm_aligner_structured
        self.config.mm_aligner_size = model_args.mm_aligner_size

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.pretrain_mm_mlp_adapter_size = pretrain_mm_mlp_adapter_size

        if not self.config.mm_aligner_structured:
            self.mm_projector = build_vision_projector(self.config)
        elif self.adapter_qlen is not None:
            self.mm_projector = build_vision_projector_fusion_adapter(self.config, self.config.mm_aligner_size, self.adapter_qlen, pretrained_mm_size=pretrain_mm_mlp_adapter_size)
        else:
            self.mm_projector = build_vision_projector_aligner(self.config, self.config.mm_aligner_size, pretrained_mm_size=pretrain_mm_mlp_adapter_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            print(f"Loading a pretrained projector from {pretrain_mm_mlp_adapter}")
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)

    # def build_vision_tower(self, *args, **kwargs):
    #     self.clip, self.clip_transform = clip.load('ViT-L/14')

    # def clip_encode_image(self, x):
    #     # modified from CLIP
    #     x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
    #     # shape = [*, width, grid ** 2]
    #     x = x.reshape(x.shape[0], x.shape[1], -1)
    #     x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    #     x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
    #                   x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    #     x = x + self.clip.visual.positional_embedding.to(x.dtype)
    #     x = self.clip.visual.ln_pre(x)

    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.clip.visual.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD

    #     # preserve all spatial tokens
    #     x = self.clip.visual.ln_post(x[:, :, :])

    #     if self.clip.visual.proj is not None:
    #         x = x @ self.clip.visual.proj

    #     return x
    
    def encode_images(self, images, blank_image_enabled=None):
        # Clip encoder
        image_features = self.vision_tower(images)

        # Proejction mix and match, fusion, all in one
        x = self.mm_projector(image_features)

        # Padding for blank images
        
        # if blank_image_enabled is not None:
            # 3 for only padding, 4 for extra layer since you get the gelu
            # pad_multiplier = self.mm_projector._modules['4'].max_base_size
            # pad = self.mm_projector._modules['4'].pad

            # pad_multiplier = self.mm_projector._modules['6'].max_base_size
            # pad = self.mm_projector._modules['6'].pad
            # for x_b in range(x.shape[0]):
            #     if blank_image_enabled[x_b]:
            #         #If it is a blank image, then pad all the way
            #         x[x_b] = torch.reshape(pad.repeat(x.shape[1] * pad_multiplier), (x.shape[1], -1))
     
        return x

class AdaptiveLlamaConfig(LlamaConfig):
    model_type = "adaptive_llama"

class AdaptiveLlamaModel(LlamaForCausalLM):
    config_class = AdaptiveLlamaConfig

    def __init__(self, config, proj_config = None, lm_adapter=None, inference_mode=False):
        
        super().__init__(config, adapter_specs=lm_adapter)

        self.lm_adapter = lm_adapter
        self.proj_config = proj_config if proj_config is not None else config
        self.vision_module = AdaptiveLlamaVisionModule(config, proj_config=self.proj_config, adapter_qlen=self.lm_adapter.adapter_base_len, inference_mode=inference_mode)

        self.post_init()
        # Initialize weights and apply final processing by calling LlamaForCausalLM
    
    # override LlamaForCausalLM forward

    def get_vision_tower(self):
        return self.vision_module.vision_tower

    def get_model(self):
        return self.vision_module
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        blank_image_enabled: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        visual_query = self.vision_module.encode_images(images, blank_image_enabled)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_query=visual_query,
            attn_adapter=self.lm_adapter,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images"),
            }
        )
        return model_inputs
    
    # Stop trainer from complaining
    def initialize_vision_tokenizer(*args, **kwargs):
        pass

AutoConfig.register("adaptive_llama", AdaptiveLlamaConfig)
AutoModelForCausalLM.register(AdaptiveLlamaConfig, AdaptiveLlamaModel)