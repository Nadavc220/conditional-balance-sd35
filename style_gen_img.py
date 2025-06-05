import os
import argparse
import torch
from diffusers import StableDiffusion3Pipeline

from models import create_infer_model
from utils import generate_file_name

NUM_JOINT_ATT_LAYERS = 38

def main(args):
    device = 'cuda'

    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to(device)
    infer_pipeline = create_infer_model(pipeline, args.num_style_layers, controlnet_model=False)

    # Init output path
    args.output_path = os.path.join(args.output_path, 'text', str(args.seed))
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize DDPM parameters
    torch.manual_seed(args.seed)
    target_prompts = [f"{args.reference_prompt} of {content_prompt} in the style of {args.style_prompt}" for content_prompt in args.content_prompts]
    reference_prompt = f"{args.reference_prompt} in the style of {args.style_prompt}"
    prompts = [reference_prompt] + target_prompts

    # Run Pipeline
    images = infer_pipeline(prompts,
                            num_inference_steps=args.num_inference_steps,
                            num_images_per_prompt=args.num_images_per_prompt,
                            guidance_scale=3.5)
    
    # Save Output
    f_name = generate_file_name(args.num_style_layers, controlnet_removal_amount=0, content_image_path=None, seed=args.seed)
    images[0].save(os.path.join(args.output_path, f"ref_{args.seed}.png"))
    for i in range(1, len(images)):
        images[i].save(os.path.join(args.output_path, f"{i}_" + f_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed")
    parser.add_argument('--content_prompts', nargs='+', type=str, help='List of prompts to generate')
    parser.add_argument("--style_prompt", type=str)
    parser.add_argument("--reference_prompt", type=str)
    parser.add_argument('--lambda_s', default=28, type=float)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument('--output_path', default='./outputs')
    args = parser.parse_args()

    
    assert 0 <= args.lambda_s <= NUM_JOINT_ATT_LAYERS, "Plase enter a float value in range [0, 1] (ratio) or an int value in range [0, 70] (number of layers for stylization)"
    if 0 < args.lambda_s <= 1:
        args.lambda_s = args.lambda_s * NUM_JOINT_ATT_LAYERS
    args.num_style_layers = int(args.lambda_s) 

    assert args.num_images_per_prompt > 0, f"'num_output_imgs' should be at least 1. Current value: {args.num_images_per_prompt}"

    main(args)