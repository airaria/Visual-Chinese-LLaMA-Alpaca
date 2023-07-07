from typing import Optional, List, Union
import os
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.clip.modeling_clip import CLIPVisionConfig, CLIPVisionModel
from huggingface_hub import hf_hub_download
from .configuration_visualcla import VisualCLAConfig
from .modeling_visual_resampler import VisualResamplerConfig, VisualResamplerModel


def extend_position_embedding(state_dict, patch_size, after):
    """
    modify state_dict in-place for longer position embeddings
    """
    keys = {}
    for k,v in state_dict.items():
        if k.endswith('vision_model.embeddings.position_embedding.weight'):
            assert k not in keys
            keys['pe'] = (k,v)
        if k.endswith('vision_model.embeddings.position_ids'):
            assert k not in keys
            keys['pi'] = (k,v)

    pe_weight = keys['pe'][1]
    position_length_before = pe_weight.shape[0]
    embed_dim = pe_weight.shape[1]
    grid_before = position_length_before - 1
    position_length_after = (after // patch_size) ** 2 + 1 
    grid_after = position_length_after - 1

    new_pe_weight = pe_weight[1:].reshape((grid_before,grid_before,-1))
    new_pe_weight =  torch.nn.functional.interpolate(
        new_pe_weight.permute(2,0,1).unsqueeze(0),
        size = (grid_after,grid_after), mode = 'bicubic')
    new_pe_weight = new_pe_weight.squeeze(0).permute(1,2,0).reshape(grid_after*grid_after, -1)
    new_pe_weight = torch.cat((pe_weight[0:1],new_pe_weight), dim=0)
    assert new_pe_weight.shape == (grid_after*grid_after + 1, embed_dim)
    
    state_dict[keys['pe'][0]] = new_pe_weight
    state_dict[keys['pi'][0]] = torch.arange(grid_after*grid_after + 1).unsqueeze(0)
    return state_dict


class VisualCLAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = VisualCLAConfig
    base_model_prefix = "visualcla"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class VisualCLAModel(VisualCLAPreTrainedModel):
    def __init__(
        self,
        config: VisualCLAConfig = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        device=None
    ):
        super().__init__(config)
        if vision_model is None:
            if config.vision_config is not None:
                vision_config = CLIPVisionConfig(**config.vision_config)
                vision_model = CLIPVisionModel(vision_config)

        if text_model is None:
            if config.text_config is not None:
                text_config = LlamaConfig(**config.text_config)
                text_model = LlamaForCausalLM(text_config)

        self.vision_model = vision_model
        self.text_model = text_model

        self.vision_embed_dim = config.vision_config['hidden_size']
        self.text_embed_dim = config.text_config['hidden_size']

        if config.use_visual_resampler == True:
            resampler_config = VisualResamplerConfig(**config.visual_resampler_config)
            self.visual_resampler = VisualResamplerModel(resampler_config)
            if device:
                self.visual_resampler = self.visual_resampler.to(device)
        else:
            self.visual_resampler = None
        self.image_projection_layer = nn.Linear(self.vision_embed_dim,self.text_embed_dim)
        if device:
            self.image_projection_layer = self.image_projection_layer.to(device)

        # Initialize weights
        self.image_projection_layer.apply(self._init_weights)
        self.image_at_head = True

    @classmethod
    def from_pretrained(cls, 
        *args, 
        **kwargs
    ) -> PreTrainedModel:
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_merged_pretrained(cls, 
        visualcla_model_name_or_path: str = None, 
        *args, 
        **kwargs
    ) -> PreTrainedModel:
        # At the moment fast initialization is not supported
        # for merged models

        is_local = os.path.isdir(visualcla_model_name_or_path)
        if is_local:
            visualcla_model_local_path = visualcla_model_name_or_path
        else:
            visualcla_model_local_path = hf_hub_download(visualcla_model_name_or_path)

        torch_dtype = kwargs.pop('torch_dtype')
        default_device = kwargs.pop('default_device')
        device_map = kwargs.pop('device_map')
        load_in_8bit = kwargs.pop('load_in_8bit')

        visualcla_config = VisualCLAConfig.from_pretrained(visualcla_model_local_path)
        text_model_path = os.path.join(visualcla_model_name_or_path, 'text_encoder')
        vision_model_path = os.path.join(visualcla_model_name_or_path, 'vision_encoder')
        
        from glob import glob
        visualcla_model_ckpts = list(glob(visualcla_model_local_path + '/pytorch_model*.bin'))
        visualcla_model_state_dict = {}
        for ckpt in visualcla_model_ckpts:
            state_dict = torch.load(ckpt, map_location='cpu')
            visualcla_model_state_dict.update(state_dict)

        text_model = LlamaForCausalLM.from_pretrained(text_model_path,
                                                    torch_dtype=torch_dtype, 
                                                    low_cpu_mem_usage=True, 
                                                    device_map=device_map, 
                                                    load_in_8bit=load_in_8bit,
                                                    **kwargs)
        vision_model = CLIPVisionModel.from_pretrained(vision_model_path,
                                                    torch_dtype=torch_dtype,
                                                    device_map={"":default_device},
                                                    **kwargs)

        # instantiate config with corresponding kwargs
        visualcla_config.text_config = text_model.config.to_dict()
        visualcla_config.vision_config = vision_model.config.to_dict()

        # init model
        model = cls(config=visualcla_config,
                    vision_model=vision_model,
                    text_model=text_model,
                    device=default_device)

        visual_resampler_state_dict = {k.replace('visual_resampler.', ''):v for k,v in visualcla_model_state_dict.items() if k.startswith('visual_resampler')}
        model.visual_resampler.load_state_dict(visual_resampler_state_dict)
        model.visual_resampler.to(torch_dtype)

        image_projection_layer_state_dict = {k.replace('image_projection_layer.', ''):v for k,v in visualcla_model_state_dict.items() if k.startswith('image_projection_layer')}
        model.image_projection_layer.weight = torch.nn.Parameter(image_projection_layer_state_dict['weight'], False)
        model.image_projection_layer.bias = torch.nn.Parameter(image_projection_layer_state_dict['bias'], False)
        model.image_projection_layer.to(default_device).to(torch_dtype)

        return model

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        visualcla_config: Union[str,VisualCLAConfig] = None,
        torch_dtype=torch.float16,
        default_device=None,
        device_map=None,
        load_in_8bit=False,
        **kwargs,
    ) -> PreTrainedModel:

        if isinstance(visualcla_config, str):
            visualcla_config = VisualCLAConfig.from_pretrained(visualcla_config)

        kwargs_vision = {
            argument[len("vision_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = CLIPVisionConfig.from_pretrained(vision_model_name_or_path)
            else:
                vision_config = kwargs_vision["config"]
            vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path,
                                                            torch_dtype=torch_dtype,
                                                            device_map={"":default_device},
                                                            **kwargs_vision)

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = LlamaConfig.from_pretrained(text_model_name_or_path)
            else:
                text_config = kwargs_text["llama_config"]

            text_model = LlamaForCausalLM.from_pretrained(text_model_name_or_path, 
                                                              torch_dtype=torch_dtype, 
                                                              low_cpu_mem_usage=True, 
                                                              device_map=device_map, 
                                                              load_in_8bit=load_in_8bit,
                                                              **kwargs_text)

        # instantiate config with corresponding kwargs
        visualcla_config.text_config = text_config.to_dict()
        visualcla_config.vision_config = vision_config.to_dict()

        # init model
        model = cls(config=visualcla_config,
                    vision_model=vision_model,
                    text_model=text_model,
                    device=default_device)
        if torch_dtype == torch.float16:
            model.visual_resampler.half()
            model.image_projection_layer.half()
        return model


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        '''
        attention_mask: mask for the text
        '''
        input_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            # Vision Features
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = self.vision_model.vision_model.post_layernorm(vision_outputs[0])
            if self.config.use_visual_resampler:
                resample_image_embeds = self.visual_resampler(encoder_hidden_states=image_embeds)
                image_embeds = resample_image_embeds.last_hidden_state
            image_embeds = self.image_projection_layer(image_embeds)
            #image markers are <img> and </img> and position 1 and 2 if image at head
            if self.image_at_head:
                multimodal_embeds = torch.concat([input_embeds[:,:2], image_embeds, input_embeds[:,2:]],dim=1)
            else:
                new_input_embeds = []
                for cur_input_ids, cur_input_embeds, cur_image_embeds in zip(input_ids, input_embeds, image_embeds):
                    num_patch = cur_image_embeds.shape[0]
                    image_start_token_pos = torch.where(cur_input_ids == self.tokenizer.img_start_token_id)[0]
                    if len(image_start_token_pos) == 0 or not self.tokenizer.img_token_id in cur_input_ids:
                        new_input_embeds.append(cur_input_embeds)
                        continue
                    if cur_input_ids[image_start_token_pos + num_patch + 1] != self.tokenizer.img_end_token_id:
                        print(cur_input_ids)
                        raise ValueError(f"Num of patch ({num_patch}) is not equal to the length of pre-filled image patch tokens.")
                    cur_new_input_embeds = torch.cat([cur_input_embeds[:image_start_token_pos+1], cur_image_embeds, cur_input_embeds[image_start_token_pos+num_patch+1:]], dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                multimodal_embeds = torch.stack(new_input_embeds, dim=0)

            if attention_mask is not None:
                if self.image_at_head:
                    image_mask = torch.ones(image_embeds.shape[:2], dtype=input_ids.dtype, device=input_ids.device)
                    multimodal_mask = torch.concat([image_mask,attention_mask],dim=1) 
                else:
                    multimodal_mask = attention_mask
            if 'llama' in self.text_model.__module__: # decoder text model, add -100 as labels for image patchs
                if self.image_at_head:
                    labels = torch.concat([labels[:,[0]], torch.ones(image_embeds.shape[:2], dtype=labels.dtype, device=labels.device)*-100, labels[:,1:]],dim=1)

        else:
            multimodal_embeds = input_embeds
            multimodal_mask = attention_mask

        outputs = self.text_model(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            return_dict = return_dict
        )

        return outputs


    @torch.no_grad()
    def generate(
        self,
        input_ids = None,
        pixel_values = None,
        attention_mask = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = False,
        **kwargs,
    ):
        input_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            # Vision Features
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = self.vision_model.vision_model.post_layernorm(vision_outputs[0])
            if self.config.use_visual_resampler:
                resample_image_embeds = self.visual_resampler(encoder_hidden_states=image_embeds)
                image_embeds = resample_image_embeds.last_hidden_state
            image_embeds = self.image_projection_layer(image_embeds)
            # image markers are <img> and </img> and position 1 and 2 if image at head
            if self.image_at_head:
                multimodal_embeds = torch.concat([input_embeds[:,:2], image_embeds, input_embeds[:,2:]],dim=1)
            else:
                new_input_embeds = []
                for cur_input_ids, cur_input_embeds, cur_image_embeds in zip(input_ids, input_embeds, image_embeds):
                    num_patch = cur_image_embeds.shape[0]
                    image_start_token_pos = torch.where(cur_input_ids == self.tokenizer.img_start_token_id)[0]
                    if len(image_start_token_pos) == 0:
                        new_input_embeds.append(cur_input_embeds)
                        continue
                    if cur_input_ids[image_start_token_pos + num_patch + 1] != self.tokenizer.img_end_token_id:
                        raise ValueError(f"Num of patch ({num_patch}) is not equal to the length of pre-filled image patch tokens.")
                    cur_new_input_embeds = torch.cat([cur_input_embeds[:image_start_token_pos+1], cur_image_embeds, cur_input_embeds[image_start_token_pos+num_patch+1:]], dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                multimodal_embeds = torch.stack(new_input_embeds, dim=0)

            if attention_mask is not None:
                if self.image_at_head:
                    image_mask = torch.ones(image_embeds.shape[:2], dtype=input_ids.dtype, device=input_ids.device)
                    multimodal_mask = torch.concat([image_mask,attention_mask],dim=1)
                else:
                    multimodal_mask = attention_mask
        else:
            multimodal_embeds = input_embeds
            multimodal_mask = attention_mask

        outputs = self.text_model.generate(
            inputs_embeds=multimodal_embeds,
            attention_mask = multimodal_mask,
            generation_config = generation_config,
            logits_processor = logits_processor,
            stopping_criteria = stopping_criteria,
            prefix_allowed_tokens_fn = prefix_allowed_tokens_fn,
            synced_gpus = synced_gpus,
            **kwargs
        )
        return outputs

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.text_model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_model.set_output_embeddings(new_embeddings)
