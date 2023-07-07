"""
Usage: 
python merge_llama_with_visualcla_lora.py \
    --text_model path/to/chinese/alpaca/plus \
    --vision_model path/to/clip/vit
    --lora_model path/to/visualcla/lora \
    --output_dir path/to/output/dir
"""
import torch
import argparse
import visualcla
import os
import argparse
import peft
from peft import PeftModel
from transformers import  CLIPImageProcessor
import gc
assert hasattr(peft.LoraModel,'merge_and_unload'), "'merge_and_unload' method cannot been found in PeftModel. Please update your Peft to 0.3.0."

parser = argparse.ArgumentParser()
parser.add_argument('--text_model', default=None, required=True,
                    type=str, help="Path to Chinese-Alpaca model")
parser.add_argument('--vision_model', default=None, required=True,
                    type=str, help="Path to CLIP-ViT model")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Path to VisualCLA LoRA")
parser.add_argument('--output_dir', default='./merged_model',
                    type=str, help="Path to output dir")

emb_to_model_size = {
    4096 : '7B',
    5120 : '13B',
    6656 : '33B',
    8192 : '65B',
}

if __name__=='__main__':

    args = parser.parse_args()
    text_model_path = args.text_model
    vision_model_path = args.vision_model
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip())!=0]
    output_dir = args.output_dir

    print(f"Text model: {text_model_path}")
    print(f"Vision model: {vision_model_path}")
    print(f"LoRA model(s) {lora_model_paths}:")

    base_model, tokenizer, image_processor = visualcla.get_model_and_tokenizer_and_processor(
        visualcla_model=None,
        text_model=text_model_path,
        vision_model=vision_model_path,
        lora_model=lora_model_paths[0],
        torch_dtype=torch.float16,
        default_device=torch.device('cpu'),
        device_map={"": "cpu"},
        load_in_8bit=False
    )

    embedding_size = base_model.get_input_embeddings().weight.size(1)
    model_size = emb_to_model_size[embedding_size]
    print(f"Peft version: {peft.__version__}")
    print(f"Loading LoRA for {model_size} model")

    lora_model = None
    lora_model_sd = None
    for lora_index, lora_model_path in enumerate(lora_model_paths):
        print(f"Loading LoRA {lora_model_path}...")

        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        assert len(tokenizer) >= model_vocab_size, \
        (f"The vocab size of the tokenizer {len(tokenizer)} is smaller than the vocab size of the model {model_vocab_size}\n"
        "This is not the intended use. Please check your model and tokenizer.")
        print(f"Model embedding size is {model_vocab_size}. Resize embedding size to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

        print(f"Loading LoRA weights")
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            device_map={"": "cpu"},
            torch_dtype=torch.float16,
        )
        print(f"Merging with merge_and_unload...")
        base_model = lora_model.merge_and_unload()

    tokenizer.save_pretrained(output_dir)
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_path)
    print("Saving CLIPImageProcessor...")
    image_processor.save_pretrained(output_dir)
    print("Saving to Hugging Face format...")
    base_model.text_model.save_pretrained(os.path.join(output_dir,'text_encoder'))
    base_model.vision_model.save_pretrained(os.path.join(output_dir,'vision_encoder'))
    base_model.config.save_pretrained(output_dir)
    resampler_state_dict = {k:v for k,v in base_model.state_dict().items()
                             if (not k.startswith('text_model')) and (not k.startswith('vision_model'))}
    torch.save(resampler_state_dict,os.path.join(output_dir,'pytorch_model.bin'))
    print("Done")