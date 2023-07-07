from typing import Optional

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline

available_pipelines = ['visualcla-7b']


def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if name == 'visualcla-7b':
        from .visualcla import VisualCLA_7B_Pipeline
        return VisualCLA_7B_Pipeline(params)
    return None


def get_pipeline_from_model_name(model_name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if 'visualcla' not in model_name.lower():
        return None
    if '7b' in model_name.lower():
        from .visualcla import VisualCLA_7B_Pipeline
        return VisualCLA_7B_Pipeline(params)
    return None
