import logging
import math
import gc
import traceback
from queue import Queue
from threading import Thread
from .modeling_visualcla import VisualCLAModel
from .configuration_visualcla import VisualCLAConfig
from PIL import Image
from copy import deepcopy
import torch
from typing import Union

from transformers import  CLIPImageProcessor
from transformers import LlamaTokenizer
from transformers.image_utils import ImageInput
import transformers
from transformers import (
    GenerationConfig, 
    LogitsProcessorList, 
    TemperatureLogitsWarper, 
    LogitsWarper
)
from transformers.generation.logits_process import LogitNormalization

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_MULTIMODAL = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)

prompt_sep_before = '### '
prompt_sep_after = '\n\n'

DEFAULT_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=512,
    min_length=0,
    do_sample=True,
    top_p=0.9,
    top_k=40,
    num_beams=1,
    temperature=0.5,
    num_return_sequences=1,
    no_repeat_ngram_size=15,
    repetition_penalty=1.1
)

def encoding_text(history, text, num_patch, tokenizer):
    if history == []:
        prompt_text = prompt_sep_before + 'Instruction' + ': \n' + \
            '<image_placeholder>\n' + text + prompt_sep_after + \
            prompt_sep_before + 'Response' + ':'
    else:
        prompt_text = prompt_sep_before + 'Instruction' + ': \n' + \
            text + prompt_sep_after + \
            prompt_sep_before + 'Response' + ':'
            
    for hist in history[::-1]:
        if hist['type'] == 'instruction':
            if "first_instruction" in hist:
                prompt_text = prompt_sep_before + 'Instruction' + ': \n' + \
                    '<image_placeholder>\n' + hist['value'] + prompt_sep_after + \
                    prompt_text
            else:
                prompt_text = prompt_sep_before + 'Instruction' + ': \n' + \
                    hist['value'] + prompt_sep_after + \
                    prompt_text
        elif hist['type'] == 'response':
            prompt_text = prompt_sep_before + 'Response' + ':' + \
                hist['value'] + prompt_sep_after + \
                prompt_text
        else:
            raise ValueError(f"Except 'type' are 'instruction' and 'response', but get '{hist['type']}'.")
    
    prompt_text = PROMPT_TEMPLATE_MULTIMODAL + prompt_text
    prompt_text = prompt_text.replace('<image_placeholder>', tokenizer.img_start_token + num_patch*tokenizer.img_token + tokenizer.img_end_token)
    input_text = tokenizer.bos_token + prompt_text
    test_input = tokenizer(input_text, return_tensors='pt', add_special_tokens=False)
    return test_input


def get_model_and_tokenizer_and_processor(
    visualcla_model=None,
    text_model=None,
    vision_model=None,
    lora_model=None,
    torch_dtype=torch.float16,
    default_device=None,
    device_map=None,
    load_in_8bit=False
):
    # get tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(visualcla_model or lora_model)
    #special_token_dict = {'additional_special_tokens': ['<img>','</img>','<pad>','<img_token>']}
    tokenizer.pad_token = '<pad>'
    tokenizer.img_start_token = '<img>'
    tokenizer.img_start_token_id = tokenizer.convert_tokens_to_ids(tokenizer.img_start_token)
    tokenizer.img_end_token = '</img>'
    tokenizer.img_end_token_id = tokenizer.convert_tokens_to_ids(tokenizer.img_end_token)
    tokenizer.img_token = '<img_token>'
    tokenizer.img_token_id = tokenizer.convert_tokens_to_ids(tokenizer.img_token)

    # get model llama + clip-vit
    if visualcla_model is not None:
        logger.info("Init VisualCLA model from pretrained")
        model = VisualCLAModel.from_merged_pretrained(
            visualcla_model,
            torch_dtype=torch_dtype,
            default_device=default_device,
            device_map=device_map,
            load_in_8bit=load_in_8bit
        )
    else:
        assert text_model is not None and vision_model is not None
        # init VisualCLA model
        logger.info("Init VisualCLA model with pretrained text/image encoders")
        visualcla_config = VisualCLAConfig.from_pretrained(lora_model)
        model = VisualCLAModel.from_vision_text_pretrained(
            vision_model,
            text_model,
            visualcla_config=visualcla_config,
            torch_dtype=torch_dtype,
            default_device=default_device,
            device_map=device_map,
            load_in_8bit=load_in_8bit
        )


    image_processor = CLIPImageProcessor.from_pretrained(vision_model or visualcla_model)
    image_processor.patch_size = model.vision_model.config.patch_size
    model.tokenizer = tokenizer
    model.image_processor = image_processor
    model.image_at_head = False

    if model.config.visual_resampler_config['num_query_tokens'] != -1:
        model.num_patch = model.config.visual_resampler_config['num_query_tokens']
    else:
        model.num_patch = (image_processor.size["shortest_edge"] // image_processor.patch_size) ** 2 + 1

    return model, tokenizer, image_processor

@torch.inference_mode()
def chat(model, image : Union[str,ImageInput], text : str, history = [], generation_config = None):

    generation_config = generation_config or DEFAULT_GENERATION_CONFIG
    generation_config.bos_token_id = generation_config.bos_token_id or model.tokenizer.bos_token_id

    if isinstance(image, str):
        pixel_values = model.image_processor(Image.open(image), return_tensors='pt').pixel_values
    elif isinstance(image, Image.Image):
        pixel_values = model.image_processor(image, return_tensors='pt').pixel_values
    else:
        pixel_values = image
    test_input = encoding_text(history, text, model.num_patch, model.tokenizer)
    if model.device == torch.device('cpu'):
        test_input["pixel_values"] = pixel_values.float()
    else:
        test_input["pixel_values"] = pixel_values.half()
    test_input.to(model.device)

    if len(history) == 0:
        history.append({"type":"instruction", "value":text, "first_instruction": True})
    else:
        history.append({"type":"instruction", "value":text})

    outputs = model.generate(
        input_ids=test_input.input_ids,
        attention_mask=test_input.attention_mask,
        pixel_values=test_input.pixel_values,
        generation_config=generation_config,
    )
    output = outputs[0]
    response = model.tokenizer.decode(output, skip_special_tokens=True)
    history.append({"type":"response", "value":response})
    print("Response:", response)
    print("History:", history)
    return response, history

@torch.inference_mode()
def chat_in_stream(model, image : Union[str,ImageInput], text : str, history = [], generation_config = None):

    generation_config = generation_config or DEFAULT_GENERATION_CONFIG
    generation_config.bos_token_id = generation_config.bos_token_id or model.tokenizer.bos_token_id

    if isinstance(image, str):
        pixel_values = model.image_processor(Image.open(image), return_tensors='pt').pixel_values
    elif isinstance(image, Image.Image):
        pixel_values = model.image_processor(image, return_tensors='pt').pixel_values
    else:
        pixel_values = image
    test_input = encoding_text(history, text, model.num_patch, model.tokenizer)
    if model.device == torch.device('cpu'):
        test_input["pixel_values"] = pixel_values.float()
    else:
        test_input["pixel_values"] = pixel_values.half()
    test_input.to(model.device)

    if len(history) == 0:
        history.append({"type":"instruction", "value":text, "first_instruction": True})
    else:
        history.append({"type":"instruction", "value":text})

    origin_size = len(test_input.input_ids[0])
    eos_token_id = model.tokenizer.eos_token_id

    response = ''
    old_history = deepcopy(history)

    generate_params = generation_config.to_dict()
    generate_params['input_ids'] = test_input.input_ids
    generate_params['attention_mask'] = test_input.attention_mask
    generate_params['pixel_values'] = test_input.pixel_values

    def generate_with_callback(callback=None, **kwargs):
        if 'stopping_criteria' in kwargs:
            kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        else:
            kwargs['stopping_criteria'] = [Stream(callback_func=callback)]
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            next_token_ids = output
            if next_token_ids[0] == eos_token_id:
                break
            next_tokens = model.tokenizer.decode(
                next_token_ids, skip_special_tokens=True)
            if type(model.tokenizer) is LlamaTokenizer and len(next_token_ids) > 0:
                if model.tokenizer.convert_ids_to_tokens(int(next_token_ids[0])).startswith('▁'):
                    next_tokens = ' ' + next_tokens
            response = next_tokens

            history = deepcopy(old_history)
            history.append({"type":"response", "value":response})

            yield response, history
            if len(test_input.input_ids[0]) > origin_size + generation_config.max_new_tokens:
                break

        print("Response:", response)
        print("History:", history)


class TailFreeLogitsWarper(LogitsWarper):
    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class MirostatLogitsWarper(LogitsWarper):
    def __init__(self, mirostat_mode: int, mirostat_tau: float, mirostat_eta: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if mirostat_mode not in [2]:
            raise ValueError(f"`mirostat` has to be a an integer 2, but is {mirostat_mode}")
        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if (i == 0):
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0)

        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True).to('cuda')

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0))
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


def get_logits_warper_patch(self, generation_config):
    warpers = self._get_logits_warper_old(generation_config)
    warpers_to_add = LogitsProcessorList()
    min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1

    if generation_config.mirostat_mode is not None and generation_config.mirostat_mode == 2:
        warpers_to_add.append(MirostatLogitsWarper(mirostat_mode=generation_config.mirostat_mode, mirostat_eta=generation_config.mirostat_eta, mirostat_tau=generation_config.mirostat_tau, min_tokens_to_keep=min_tokens_to_keep))
        # We need to disable samplers other than temperature
        for warper in warpers:
            if not isinstance(warper, TemperatureLogitsWarper):
                warpers.remove(warper)
    else:
        if generation_config.tfs is not None and 0.0 <= generation_config.tfs <= 1.0:
            warpers_to_add.append(TailFreeLogitsWarper(tfs=generation_config.tfs, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_a is not None and 0.0 <= generation_config.top_a <= 1.0:
            warpers_to_add.append(TopALogitsWarper(top_a=generation_config.top_a, min_tokens_to_keep=min_tokens_to_keep))

    if warpers and isinstance(warpers[-1], LogitNormalization):
        warpers = warpers[:-1] + warpers_to_add + [warpers[-1]]
    else:
        warpers += warpers_to_add

    return warpers


def generation_config_init_patch(self, **kwargs):
    self.__init___old(**kwargs)
    self.tfs = kwargs.pop("tfs", 1.0)
    self.top_a = kwargs.pop("top_a", 0.0)
    self.mirostat_mode = kwargs.pop("mirostat_mode", 0)
    self.mirostat_eta = kwargs.pop("mirostat_eta", 0.1)
    self.mirostat_tau = kwargs.pop("mirostat_tau", 5)


def hijack_samplers():
    transformers.GenerationMixin._get_logits_warper_old = transformers.GenerationMixin._get_logits_warper
    transformers.GenerationMixin._get_logits_warper = get_logits_warper_patch

    transformers.GenerationConfig.__init___old = transformers.GenerationConfig.__init__
    transformers.GenerationConfig.__init__ = generation_config_init_patch



class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False



class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """

    def __init__(self, func, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()


def clear_torch_cache():
    gc.collect()
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()
