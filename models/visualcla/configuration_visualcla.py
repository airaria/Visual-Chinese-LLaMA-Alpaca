""" VisualCLA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Union, Dict

logger = logging.get_logger(__name__)


class VisualCLAConfig(PretrainedConfig):

    model_type = "visualcla"
    is_composition = True

    def __init__(
        self, 
        text_config: Union[PretrainedConfig, Dict] = None,
        vision_config: Union[PretrainedConfig, Dict] = None,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_visual_resampler=False,
        visual_resampler_config=None,
        **kwargs):
        super().__init__(**kwargs)

        if text_config:
            if isinstance(text_config,PretrainedConfig):
                text_config = text_config.to_dict()
        self.text_config = text_config

        if vision_config:
            if isinstance(vision_config, PretrainedConfig):
                vision_config = vision_config.to_dict()
        self.vision_config = vision_config

        self.initializer_range=initializer_range
        self.layer_norm_eps=layer_norm_eps

        self.use_visual_resampler = use_visual_resampler
        self.visual_resampler_config = visual_resampler_config