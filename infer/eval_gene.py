import os
import math
import json
import clip
from tqdm import tqdm
import hydra
import torch
import pyrootutils
from PIL import Image
import shortuuid
import argparse
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64

instruction_prompt = '[INST] Generate an image: {caption} [/INST]\n'


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division

    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
    

def compute_clip_score(image, text_description, model, preprocess, device):
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize([text_description]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    clip_score = (image_features @ text_features.T).item()
    return clip_score


def eval_modal(args):
    disable_torch_init()
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer_cfg = OmegaConf.load(args.tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    image_transform_cfg = OmegaConf.load(args.image_transform_cfg_path)
    image_transform = hydra.utils.instantiate(image_transform_cfg)

    visual_encoder_cfg = OmegaConf.load(args.visual_encoder_cfg_path)
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    visual_encoder.eval().to(device, dtype=dtype)
    print('Init visual encoder done')

    llm_cfg = OmegaConf.load(args.llm_cfg_path)
    llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
    print('Init llm done.')

    agent_model_cfg = OmegaConf.load(args.agent_cfg_path)
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

    agent_model.eval().to(device, dtype=dtype)
    print('Init agent mdoel Done')

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.diffusion_model_path, subfolder="scheduler")
    print('init vae')
    vae = AutoencoderKL.from_pretrained(args.diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
    print('init unet')
    unet = UNet2DConditionModel.from_pretrained(args.diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

    adapter_cfg = OmegaConf.load(args.adapter_cfg_path)
    adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()

    discrete_model_cfg = OmegaConf.load(args.discrete_model_cfg_path)
    discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
    print('Init adapter done')
    
    model, preprocess = clip.load("./pretrained/clip/ViT-B-32.pt", device=device)
    print('Init clip')

    adapter.init_pipe(vae=vae,
                      scheduler=noise_scheduler,
                      visual_encoder=visual_encoder,
                      image_transform=image_transform,
                      discrete_model=discrete_model,
                      dtype=dtype,
                      device=device)

    print('Init adapter pipe done')

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    count = 0
    for line in tqdm(questions):
        count += 1

        caption = line["text"]
        image_save_path = args.save_dir + "/" + str(caption) + '.png'

        # image generation
        with torch.no_grad():
            prompt = instruction_prompt.format_map({'caption': caption})
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokenizer.bos_token_id] + prompt_ids).to(device, dtype=torch.long).unsqueeze(0)
            output = agent_model.generate(tokenizer=tokenizer, input_ids=input_ids, num_img_gen_tokens=num_img_out_tokens)

        if output['has_img_output']:
            images = adapter.generate(image_embeds=output['img_gen_feat'].to(device), num_inference_steps=50)
            clip_score = compute_clip_score(images[0], caption, model, preprocess, device)
            if args.save_image:
                save_path = os.path.join(image_save_path)
                images[0].save(save_path)
        torch.cuda.empty_cache()
        ans_file.write(json.dumps({"caption": caption, "score": clip_score}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_cfg_path", type=str, default="configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml")
    parser.add_argument("--image_transform_cfg_path", type=str, default="configs/processer/qwen_448_transform.yaml")
    parser.add_argument("--visual_encoder_cfg_path", type=str, default="configs/visual_encoder/qwen_vitg_448.yaml")
    parser.add_argument("--llm_cfg_path", type=str, default="configs/clm_models/llm_seed_x_i.yaml")
    parser.add_argument("--agent_cfg_path", type=str, default="configs/clm_models/agent_seed_x_i.yaml")
    parser.add_argument("--adapter_cfg_path", type=str, default="configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize.yaml")
    parser.add_argument("--discrete_model_cfg_path", type=str, default="configs/discrete_model/discrete_identity.yaml")
    parser.add_argument("--diffusion_model_path", type=str, default="pretrained/stable-diffusion-xl-base-1.0")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--save_image", type=bool, default=False)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()

    eval_modal(args)