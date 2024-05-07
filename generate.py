import argparse
import json
from pheye_builder import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import warnings

def get_args():
    parser = argparse.ArgumentParser(description="Generation settings")

    parser.add_argument('--hf_model', type=str, default="miguelcarv/Pheye-x4-448")
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--device', type=str, default="cpu")

    args = parser.parse_args()

    return args


def get_config(hf_model_path):
    config_path = hf_hub_download(hf_model_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def get_model_path(hf_model_path):
    return hf_hub_download(hf_model_path, "checkpoint.pt")



if __name__ == "__main__":

    with warnings.catch_warnings(action="ignore"): # ehehehe

        SYSTEM_PROMPT = "You are an AI visual assistant and you are seeing a single image. You will receive an instruction regarding that image. Your goal is to follow the instruction as faithfully as you can."

        args = get_args()
        config = get_config(args.hf_model)

        model, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=config["encoder"],
            lang_decoder_path=config["decoder"],
            tokenizer_path=config["tokenizer"],
            cross_attn_every_n_layers=config["cross_interval"],
            level=config["level"],
            reduce_factor=config["reduce"],
            from_layer=config["from_layer"],
            encoder_dtype=eval(config["encoder_dtype"]),
            decoder_dtype=eval(config["decoder_dtype"]),
            dtype=eval(config["other_params_dtype"])
        ) 

        if config["first_level"]:
            model.vision_encoder.add_first_level_adapter()

        model_path = get_model_path(args.hf_model)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(args.device)

        image = [Image.open(args.image_path).convert("RGB")]
        prompt = [f"{SYSTEM_PROMPT}\n\nInstruction: {args.prompt}\nOutput:"]
        inputs = tokenizer(prompt, padding='longest', return_tensors='pt')

        model.eval()
        with torch.no_grad():
            outputs = model.generate(vision_x=image, 
                                    lang_x=inputs.input_ids.to(args.device),
                                    device=args.device,
                                    max_new_tokens=args.max_new_tokens,
                                    num_beams = args.num_beams,
                                    eos_token_id = tokenizer.eos_token_id,
                                    pad_token_id = tokenizer.pad_token_id,
                                    attention_mask=inputs.attention_mask.to(args.device))
            answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("Output:")[-1].lstrip()

        
        print(f"Instruction: {args.prompt}")
        print(f"Response: {answer}")