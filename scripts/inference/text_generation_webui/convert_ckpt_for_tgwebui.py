import torch
import os
import shutil
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Path to VisualCLA LoRA")


if __name__=='__main__':
    args = parser.parse_args()

    raw_dir = args.lora_model
    vision_dir = '../' + raw_dir.rstrip('/\\') + '_vision_lora_model_tgwebui'
    text_dir = '../' + raw_dir.rstrip('/\\') + '_text_lora_model_tgwebui'
    shutil.copytree(raw_dir, vision_dir, dirs_exist_ok=True)
    shutil.copytree(raw_dir, text_dir, dirs_exist_ok=True)

    # copy config
    vlpe_config = os.path.join(raw_dir, 'config.json')
    with open(os.path.join(vlpe_config), 'r') as f:
        visual_resampler_config = json.load(f)['visual_resampler_config']
    with open(os.path.join(vision_dir, 'visual_resampler_config.json'), 'w') as f:
        json.dump(visual_resampler_config, f, indent=2)

    lora_config = 'adapter_config.json'
    with open(os.path.join(raw_dir, lora_config), 'r') as f:
        text_lora_config = json.load(f)
    text_lora_config['modules_to_save'] = [
        "embed_tokens",
        "lm_head",
    ]
    text_lora_config['target_modules'] = ".*(self_attn|mlp).*(q_proj|k_proj|v_proj|o_proj|gate_proj|down_proj|up_proj)$"
    with open(os.path.join(text_dir, lora_config), 'w') as f:
        json.dump(text_lora_config, f, indent=2)    

    model_path = os.path.join(raw_dir, 'adapter_model.bin')
    assert os.path.exists(model_path), f"Cannot found model checkpoint in directory {raw_dir}"

    raw_ckpt = torch.load(model_path, map_location=torch.device('cpu'))


    # vision model LoRA
    vision_model_ckpt = {k:v for k,v in raw_ckpt.items() if k.startswith('base_model.model.vision_model')}
    vision_model_ckpt = {k.replace('vision_model.vision_model', 'vision_model'):v for k,v in vision_model_ckpt.items()}

    vision_model_path = os.path.join(vision_dir, 'adapter_model.bin')
    torch.save(vision_model_ckpt, vision_model_path)

    # image projection layer
    image_projection_layer_ckpt = {k:v for k,v in raw_ckpt.items() if k.startswith('base_model.model.image_projection_layer')}
    image_projection_layer_ckpt = {k.replace('base_model.model.image_projection_layer.', ''):v for k,v in image_projection_layer_ckpt.items()}

    image_projection_layer_path = os.path.join(vision_dir, 'image_projection_layer_model.bin')
    torch.save(image_projection_layer_ckpt, image_projection_layer_path)

    # visual resampler
    visual_resampler_ckpt = {k:v for k,v in raw_ckpt.items() if k.startswith('base_model.model.visual_resampler')}
    visual_resampler_ckpt = {k.replace('base_model.model.visual_resampler.', ''):v for k,v in visual_resampler_ckpt.items()}

    visual_resampler_path = os.path.join(vision_dir, 'visual_resampler_model.bin')
    torch.save(visual_resampler_ckpt, visual_resampler_path)

    # text model LoRA
    text_model_ckpt = {k:v for k,v in raw_ckpt.items() if k.startswith('base_model.model.text_model')}
    text_model_ckpt = {k.replace('text_model.', ''):v for k,v in text_model_ckpt.items()}

    text_model_path = os.path.join(text_dir, 'adapter_model.bin')
    torch.save(text_model_ckpt, text_model_path)
