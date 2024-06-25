# The main image generating functions
import os
import torch
import random
from datetime import datetime
import numpy as np
from PIL import Image

from stylegan3_fun import legacy, dnnlib
from stylegan3_fun.gen_images import make_transform
from . import text_variables, support_functions


def prepare_model(model):
    # Generate image function part 1 - modified from stylegan3_fun.gen_images.generate_images
    print(f"*** Bros ***: Preparing for image generation with model: {model}")
    device = torch.device("cuda")
    with dnnlib.util.open_url(model) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    return device, G, label


def create_image(seed, psi, device, G, label):
    # Generate image function part 2 - modified from stylegan3_fun.gen_images.generate_images
    noise_mode = "const"
    # can be 'const', 'random', 'none' - doesn't seem to make a difference

    translate = (0, 0)
    rotate = 0
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    if hasattr(G.synthesis, "input"):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    img = G(z, label, truncation_psi=psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_image = Image.fromarray(img[0].cpu().numpy(), "RGB")
    return process_borders(pil_image)


def process_borders(image: Image.Image) -> Image.Image:
    # convert to numpy
    img_array = np.array(image)
    original_height, original_width = img_array.shape[:2]
    # function to check if borders exist
    def find_border_width(arr, threshold=10):
        for i in range(len(arr)):
            if np.mean(np.abs(arr[i] - arr[0])) > threshold:
                return i
        return 0
    # run it for all sides
    top_border = find_border_width(img_array)
    bottom_border = find_border_width(img_array[::-1])
    left_border = find_border_width(img_array.transpose(1, 0, 2))
    right_border = find_border_width(img_array.transpose(1, 0, 2)[::-1])
    if top_border == bottom_border == left_border == right_border == 0:
        # no borders found
        return image
    # crop the borders
    cropped_array = img_array[top_border:original_height-bottom_border, left_border:original_width-right_border]
    # back to PIL type and resize
    cropped_image = Image.fromarray(cropped_array)
    resized_image = cropped_image.resize((original_width, original_height), Image.LANCZOS)
    return resized_image


def generate_single_image(model, seed, psi, neg_psi):
    # Generates a single image
    # Check if seed is valid, otherwise generate random
    if type(seed) == int and seed >= 0 and seed <= 4294967295:
        seed = seed
    else:
        seed = random.randint(0, 4294967295)
    # Negate psi if needed
    psi = psi if not neg_psi else -psi
    # Prepare model, write to console and generate image
    device, G, label = prepare_model(model)
    print(f"*** Bros ***: Generating single image with seed: {seed} and psi: {psi}")
    image_result = create_image(seed, psi, device, G, label)
    # Return the image and the text with used seed
    output_text = f"<pre>Seed:\t{seed}\npsi:\t{psi}</pre>"
    image_save_filename = f"{datetime.now().strftime('%Y%m%d%H%M')}_{seed}_{str(psi).replace('-', 'n').replace('.', '')}.png"
    button_text = (
        f"{text_variables.save_single_image_button_text} as {image_save_filename}"
    )
    return [image_result, str(seed), image_save_filename, output_text, button_text]


def bulk_generate_images(model, seeds, psi_values, user_amount, output_folder):
    # Bulk generate images
    if not user_amount or user_amount == "" or user_amount == 0:
        # User specified the amount as 0 or blank, use seeds and psi_values
        user_amount = 0
        seeds_list = support_functions.parse_parameter_string(seeds)
        psi_list = support_functions.parse_parameter_string(psi_values, "psi")
        if len(seeds_list) == 0:
            # No valid seeds found, so no bros
            return "<pre>No valid seeds found, so no bros :(</pre>"
        elif len(psi_list) == 0:
            # At least 1 seed but no valid psi values found, use 0.65
            psi_list = [0.65]
    else:
        # User entered a valid amount, use that
        user_amount = int(user_amount)
    # This shouldn't ever happen, but just in case check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    # Prepare model
    device, G, label = prepare_model(model)
    # Generate the images, depending on user_amount
    if user_amount == 0:
        # Generate all combinations of seeds and psi values
        counter = 0
        tasks_amount = len(seeds_list) * len(psi_list)
        for seed in seeds_list:
            for psi in psi_list:
                counter += 1
                print(
                    f"*** Bros ***: Generating image {counter}/{tasks_amount} (seed: {seed}, psi: {psi})"
                )
                image_result = create_image(seed, psi, device, G, label)
                image_filename = f"{str(counter).zfill(4)}_{datetime.now().strftime('%Y%m%d%H%M')}_{seed}_{str(psi).replace('-', 'n').replace('.', '')}.png"
                print(
                    f"*** Bros ***: Saving image {counter}/{tasks_amount} as {image_filename}"
                )
                image_result.save(os.path.join(output_folder, image_filename))
    else:
        # Generate user_amount random images, make sure the seeds are unique
        seeds_list = []
        unfilled = True
        while unfilled:
            seed = random.randint(0, 4294967295)
            if seed not in seeds_list:
                seeds_list.append(seed)
            if len(seeds_list) == user_amount:
                unfilled = False
        counter = 0
        for seed in seeds_list:
            counter += 1
            psi = 0.65
            print(
                f"*** Bros ***: Generating image {counter}/{user_amount} (seed: {seed}, psi: {psi})"
            )
            image_result = create_image(seed, psi, device, G, label)
            image_filename = f"{str(counter).zfill(4)}_{datetime.now().strftime('%Y%m%d%H%M')}_{seed}_{str(psi).replace('-', 'm').replace('.', '')}.png"
            print(
                f"*** Bros ***: Saving image {counter}/{user_amount} as {image_filename}"
            )
            image_result.save(os.path.join(output_folder, image_filename))
    return f"<pre>Successfully generated and saved {user_amount} bros in {output_folder}.</pre>"
