import os
import re
import math
import json
import clip
from tqdm import tqdm
import hydra
import torch
import pyrootutils
from PIL import Image
import argparse
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
BOP_TOKEN = '<patch>'
EOI_TOKEN = '</img>'
EOP_TOKEN = '</patch>'
IMG_TOKEN = '<img_{:05d}>'

resolution_grids = ['1x1']
base_resolution = 448

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64
instruction_prompt = '[INST] {instruction} [/INST]\n'


import base64
import torch
import math
import ast
from PIL import Image
from io import BytesIO


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def select_best_resolution_v2(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size and aspect ratio.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    original_aspect_ratio = original_height / original_width
    original_area = original_width * original_height
    best_fit = None
    min_aspect_ratio_diff = float('inf')
    min_area_ratio = float('inf')

    for width, height in possible_resolutions:
        aspect_ratio = height / width
        area = width * height
        aspect_ratio_diff = max(aspect_ratio, original_aspect_ratio) / min(aspect_ratio, original_aspect_ratio)
        area_ratio = max(area, original_area) / min(area, original_area)

        if aspect_ratio_diff < min_aspect_ratio_diff or (aspect_ratio_diff == min_aspect_ratio_diff and area_ratio < min_area_ratio):
            min_aspect_ratio_diff = aspect_ratio_diff
            min_area_ratio = area_ratio
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution, keep_ratio=False):
    """
    Resize and pad an image to a target resolution

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    if keep_ratio:
        # maintaining aspect ratio
        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized_image, (paste_x, paste_y))
    else:
        # not maintaining aspect ratio
        new_image = image.resize((target_width, target_height))
        
    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width1, height1 = select_best_resolution(image_size, possible_resolutions)
    width2, height2 = select_best_resolution_v2(image_size, possible_resolutions)
    if width1*height1 > width2*height2:
        width, height = width2, height2
    else:
        width, height = width1, height1
    return width // patch_size, height // patch_size


def process_anyres_image(image, image_transform, grid_pinpoints, base_image_size):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        image_transform: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # best_resolution = select_best_resolution(image.size, possible_resolutions)
    width1, height1 = select_best_resolution(image.size, possible_resolutions)
    width2, height2 = select_best_resolution_v2(image.size, possible_resolutions)
    if width1*height1 > width2*height2:
        width, height = width2, height2
    else:
        width, height = width1, height1
    best_resolution = [width, height]

    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, base_image_size)

    image_original_resize = image.resize((base_image_size, base_image_size))

    image_patches =  patches + [image_original_resize] # add the original image as the last patch
    image_patches = [image_transform(image_patch) for image_patch in image_patches]
    
    patch_grid = (best_resolution[0]//base_image_size, best_resolution[1]//base_image_size)
    x_index = (torch.arange(patch_grid[0]).repeat(patch_grid[1], 1)  + 0.5)/patch_grid[0]
    y_index = (torch.arange(patch_grid[1]).unsqueeze(1).repeat(1, patch_grid[0]) + 0.5)/patch_grid[1]
    patch_pos = torch.stack([x_index, y_index], dim=-1).flatten(0, 1) # h*w, 2

    origin_pos = torch.tensor([[0.5, 0.5]])
    patch_pos  = torch.cat([patch_pos, origin_pos], dim=0) # h*w+1, 2

    return torch.stack(image_patches, dim=0), patch_pos


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def anyres_data_collate(batch, tokenizer, dataset_name=None):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images', 'images_patch_length', 'patch_position', 'image_size']:
                results[key] = torch.cat(cur, dim=0)
            else:
                if key in ['input_ids']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=tokenizer.pad_token_id)
                elif key in ['attention_mask']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=0)
                elif key in ['labels']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=-100)
                elif key in ['ids_gen_mask', 'ids_cmp_mask']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=False)

                else:
                    results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    results['dataset_name'] = dataset_name

    return results


def anyres_data_collate_old(batch, dataset_name=None):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images', 'images_patch_length', 'patch_position', 'image_size']:
                results[key] = torch.cat(cur, dim=0)
            else:
                results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    results['dataset_name'] = dataset_name

    return results


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

    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def compute_clip_score(image, target_image_path, model, preprocess, device):
    output_image = preprocess(image).unsqueeze(0).to(device)
    target_image = preprocess(Image.open(target_image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        output_features = model.encode_image(output_image)
        target_features = model.encode_image(target_image)

    output_features = output_features / output_features.norm(dim=1, keepdim=True)
    target_features = target_features / target_features.norm(dim=1, keepdim=True)

    clip_score = (output_features @ target_features.T).item()
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
                      dtype=dtype,
                      device=device)

    print('Init adapter pipe done')

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

    bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
    eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

    grid_pinpoints = []
    for scale in resolution_grids:
        s1, s2 = scale.split('x')
        grid_pinpoints.append([int(s1) * base_resolution, int(s2) * base_resolution])
    grid_pinpoints = grid_pinpoints

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    count = 0
    for line in tqdm(questions):
        count += 1
        clip_score = 0

        instruction = line["instruction"]
        source_image_path = args.image_folder + line["source_image"]
        target_image_path = args.image_folder + line["target_image"]
        image_save_path = args.save_dir + "/" + line["source_image"].split("/")[-1]

        # image generation
        with torch.no_grad():
            image = Image.open(source_image_path).convert('RGB')
            source_image = image.resize((1024, 1024))

            image_tensor, patch_pos_tensor = process_anyres_image(image, image_transform, grid_pinpoints, base_resolution)
            embeds_cmp_mask = torch.tensor([True] * image_tensor.shape[0]).to(device, dtype=torch.bool)

            patch_pos = [patch_pos_tensor]
            patch_position = torch.cat(patch_pos, dim=0)

            image_tensor = image_tensor.to(device, dtype=dtype)

            patch_length = image_tensor.shape[0]
            image_tokens = ''
            for _ in range(patch_length - 1):
                image_tokens += BOP_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
            image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

            prompt = instruction_prompt.format_map({'instruction': image_tokens + instruction})

            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = [tokenizer.bos_token_id] + input_ids

            input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)

            ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))[0].tolist()
            eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))[0].tolist()

            for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
                ids_cmp_mask[boi_idx + 1:eoi_idx] = True

            input_ids = input_ids.unsqueeze(0)
            ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

            with torch.no_grad():
                image_embeds = visual_encoder(image_tensor)
                output = agent_model.generate(tokenizer=tokenizer,
                                              input_ids=input_ids,
                                              image_embeds=image_embeds,
                                              embeds_cmp_mask=embeds_cmp_mask,
                                              patch_positions=patch_position,
                                              ids_cmp_mask=ids_cmp_mask,
                                              max_new_tokens=512,
                                              num_img_gen_tokens=num_img_out_tokens)
            text = re.sub('<[^>]*>', '', output['text'])
            print(text)

        if output['has_img_output']:
            images = adapter.generate(image_embeds=output['img_gen_feat'], latent_image=source_image, num_inference_steps=50)
            clip_score = compute_clip_score(images[0], target_image_path, model, preprocess, device)
            if args.save_image:
                save_path = os.path.join(image_save_path)
                images[0].save(save_path)
        torch.cuda.empty_cache()
        ans_file.write(json.dumps({"souce_image": source_image_path, "instruction": instruction, "score": clip_score}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_cfg_path", type=str, default="configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml")
    parser.add_argument("--image_transform_cfg_path", type=str, default="configs/processer/qwen_448_transform.yaml")
    parser.add_argument("--visual_encoder_cfg_path", type=str, default="configs/visual_encoder/qwen_vitg_448.yaml")
    parser.add_argument("--llm_cfg_path", type=str, default="configs/clm_models/llm_seed_x_edit.yaml")
    parser.add_argument("--agent_cfg_path", type=str, default="configs/clm_models/agent_seed_x_i.yaml")
    parser.add_argument("--adapter_cfg_path", type=str, default="configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize.yaml")
    parser.add_argument("--discrete_model_cfg_path", type=str, default="configs/discrete_model/discrete_identity.yaml")
    parser.add_argument("--diffusion_model_path", type=str, default="pretrained/stable-diffusion-xl-base-1.0")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--save_image", type=bool, default=False)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()

    eval_modal(args)