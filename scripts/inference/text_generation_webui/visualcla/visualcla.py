import time
from abc import abstractmethod
from typing import List, Tuple
import os
import json

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from peft import PeftModel

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.logging_colors import logger
from modules.text_generation import encode

from .modeling_visual_resampler import VisualResamplerConfig, VisualResamplerModel

class VisualCLA_Pipeline(AbstractMultimodalPipeline):
    CLIP_REPO = "openai/clip-vit-large-patch14"

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.clip_device = self._get_device("vision_device", params)
        self.clip_dtype = self._get_dtype("vision_bits", params)
        self.visual_resampler_device = self._get_device("vision_device", params)
        self.visual_resampler_dtype = self._get_dtype("vision_bits", params)
        self.projector_device = self._get_device("projector_device", params)
        self.projector_dtype = self._get_dtype("projector_bits", params)
        self.image_processor, self.vision_tower, self.visual_resampler, self.image_projection_layer = self._load_models()

    def _load_models(self):
        start_ts = time.time()

        if 'visualcla_merged_model' not in shared.settings and 'visualcla_vision_lora_model' not in shared.settings:
            raise KeyError("Except one of 'visualcla_merged_model' and 'visualcla_vision_lora_model' is set in setting-visualcla.yaml, but neither was set.")

        if 'visualcla_merged_model' in shared.settings:
            vision_model_path = os.path.join(shared.settings['visualcla_merged_model'], 'vision_encoder')
            logger.info(f"VisualCLA - Loading CLIP from {vision_model_path} as {self.clip_dtype} on {self.clip_device}...")
            image_processor = CLIPImageProcessor.from_pretrained(shared.settings['visualcla_merged_model'], torch_dtype=self.clip_dtype)
            vision_tower = CLIPVisionModel.from_pretrained(vision_model_path, torch_dtype=self.clip_dtype).to(self.clip_device)

            visualcla_ckpt = torch.load(os.path.join(shared.settings['visualcla_merged_model'], 'pytorch_model.bin'))
           
            logger.info(f"VisualCLA - Loading visual resampler from {shared.settings['visualcla_merged_model']} as {self.visual_resampler_dtype} on {self.visual_resampler_device}...")
            visual_resampler_config = VisualResamplerConfig.from_dict(json.load(open(os.path.join(shared.settings['visualcla_merged_model'], 'config.json')))['visual_resampler_config'])
            visual_resampler = VisualResamplerModel(visual_resampler_config)
            visual_resampler_ckpt = {k.replace('visual_resampler.', ''):v for k,v in visualcla_ckpt.items() if k.startswith('visual_resampler')}
            visual_resampler.load_state_dict(visual_resampler_ckpt)
            visual_resampler = visual_resampler.to(self.visual_resampler_device)

            logger.info(f"VisualCLA - Loading projector from {shared.settings['visualcla_merged_model']} as {self.projector_dtype} on {self.projector_device}...")
            projector_path = os.path.join(shared.settings['visualcla_merged_model'], 'image_projection_layer_model.bin')
            image_projection_layer = torch.nn.Linear(*self.visualcla_projector_shape())
            projecton_ckpt = {k.replace('image_projection_layer.', ''):v for k,v in visualcla_ckpt.items() if k.startswith('image_projection_layer')}
            image_projection_layer.weight = torch.nn.Parameter(projecton_ckpt['weight'].to(dtype=self.projector_dtype), False)
            image_projection_layer.bias = torch.nn.Parameter(projecton_ckpt['bias'].to(dtype=self.projector_dtype), False)
            image_projection_layer = image_projection_layer.to(self.projector_device)
        else: 
            logger.info(f"VisualCLA - Loading CLIP from {VisualCLA_Pipeline.CLIP_REPO} as {self.clip_dtype} on {self.clip_device}...")
            image_processor = CLIPImageProcessor.from_pretrained(VisualCLA_Pipeline.CLIP_REPO, torch_dtype=self.clip_dtype)
            vision_tower = CLIPVisionModel.from_pretrained(VisualCLA_Pipeline.CLIP_REPO, torch_dtype=self.clip_dtype).to(self.clip_device)
            vision_tower = PeftModel.from_pretrained(vision_tower, shared.settings['visualcla_vision_lora_model']).to(self.clip_device)
            vision_tower = vision_tower.base_model.model

            logger.info(f"VisualCLA - Loading visual resampler from {shared.settings['visualcla_vision_lora_model']} as {self.visual_resampler_dtype} on {self.visual_resampler_device}...")
            visual_resampler_config =VisualResamplerConfig.from_json_file(os.path.join(shared.settings['visualcla_vision_lora_model'], 'visual_resampler_config.json'))
            visual_resampler = VisualResamplerModel(visual_resampler_config)
            visual_resampler_ckpt = torch.load(os.path.join(shared.settings['visualcla_vision_lora_model'], 'visual_resampler_model.bin'))
            visual_resampler.load_state_dict(visual_resampler_ckpt)
            visual_resampler = visual_resampler.to(self.visual_resampler_device)

            logger.info(f"VisualCLA - Loading projector from {shared.settings['visualcla_vision_lora_model']} as {self.projector_dtype} on {self.projector_device}...")
            image_projection_layer = torch.nn.Linear(*self.visualcla_projector_shape())
            projecton_ckpt = torch.load(os.path.join(shared.settings['visualcla_vision_lora_model'], 'image_projection_layer_model.bin'))
            image_projection_layer.weight = torch.nn.Parameter(projecton_ckpt['weight'].to(dtype=self.projector_dtype), False)
            image_projection_layer.bias = torch.nn.Parameter(projecton_ckpt['bias'].to(dtype=self.projector_dtype), False)
            image_projection_layer = image_projection_layer.to(self.projector_device)

        logger.info(f"VisualCLA supporting models loaded, took {time.time() - start_ts:.2f} seconds")
        return image_processor, vision_tower, visual_resampler, image_projection_layer

    @staticmethod
    def image_start() -> str:
        return "<img>"

    @staticmethod
    def image_end() -> str:
        return "</img>"

    def image_placeholder() -> str:
        return "<img_token>"

    @staticmethod
    def num_image_embeds() -> int:
        return 64

    @staticmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(shared.model.model, 'embed_tokens'):
            func = shared.model.model.embed_tokens
        else:
            func = shared.model.model.model.embed_tokens  # AutoGPTQ case

        return func(input_ids).to(shared.model.device, dtype=shared.model.dtype)

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        return VisualCLA_Pipeline.embed_tokens(
            encode(
                VisualCLA_Pipeline.image_placeholder() *VisualCLA_Pipeline.num_image_embeds(),
                add_bos_token=False)[0]
        )

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        images = self.image_processor(images, return_tensors='pt')['pixel_values']
        images = images.to(self.clip_device, dtype=self.clip_dtype)

        with torch.no_grad():
            image_forward_outs = self.vision_tower(images)
            image_embeds = self.vision_tower.vision_model.post_layernorm(image_forward_outs[0])

            resample_image_embeds = self.visual_resampler(encoder_hidden_states=image_embeds)
            image_embeds = resample_image_embeds.last_hidden_state

            image_features = image_embeds.to(self.projector_device, dtype=self.projector_dtype)
            image_features = self.image_projection_layer(image_features)
        return image_features.to(shared.model.device, dtype=shared.model.dtype)


    @staticmethod
    @abstractmethod
    def visualcla_projector_shape() -> Tuple[int, int]:
        pass


class VisualCLA_7B_Pipeline(VisualCLA_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @staticmethod
    def name() -> str:
        return "visualcla-7b"

    @staticmethod
    def placeholder_token_id() -> int:
        return 49957

    @staticmethod
    def visualcla_projector_shape() -> Tuple[int, int]:
        return (1024, 4096)


