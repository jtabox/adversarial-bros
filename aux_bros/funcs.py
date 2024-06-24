# Functions used in the main script
import os
import re
import torch
import random
from datetime import datetime
import numpy as np
from PIL import Image

from stylegan3_fun import legacy, dnnlib
from stylegan3_fun.gen_images import make_transform
from . import text_vars

def set_gen_seed(seed_text=""):
    # Sets the seed to either random or the last generated seed
    # If called without arg or with erroneous string, return -1
    if not seed_text or seed_text == "" or not str(seed_text).strip().isdigit():
        return -1
    # Called with the last_gen_seed, return that
    return int(str(seed_text).strip())


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
    return Image.fromarray(img[0].cpu().numpy(), "RGB")


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
    button_text = f"{text_vars.save_single_image_button_text} as {image_save_filename}"
    return [image_result, str(seed), image_save_filename, output_text, button_text]


def save_single_image(
    image_result, output_folder, image_save_filename, existing_output_text
):
    # Saves the generated single image
    if not image_result.any():
        return "<pre>No bro to save :(</pre>"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    image_save_filename = (
        image_save_filename
        if image_save_filename and image_save_filename != ""
        else f"{datetime.now().strftime('%Y%m%d%H%M')}_tmp_bro.png"
    )
    # The image is in numpy array format, convert to PIL and save
    Image.fromarray(image_result, "RGB").save(
        os.path.join(output_folder, image_save_filename)
    )
    return f"{existing_output_text}<br><pre>Bro saved.\nFile:\t{image_save_filename}\nFolder:\t{output_folder}</pre>"


def bulk_generate_images(model, seeds, psi_values, user_amount, output_folder):
    # Bulk generate images
    if not user_amount or user_amount == "" or user_amount == 0:
        # User specified the amount as 0 or blank, use seeds and psi_values
        user_amount = 0
        seeds_list = parse_parameter_string(seeds)
        psi_list = parse_parameter_string(psi_values, "psi")
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


def open_output_folder(output_folder):
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        os.startfile(output_folder)


def set_output_folder(output_folder):
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        return [output_folder, f"{text_vars.open_output_folder_button_text} ({output_folder})"]
    elif not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        return [output_folder, f"{text_vars.open_output_folder_button_text} ({output_folder})"]
    else:
        return [
            os.path.join(os.path.dirname(__file__), "output"),
            f"{text_vars.open_output_folder_button_text} ({os.path.join(os.path.dirname(__file__), 'output')})",
        ]


def parse_parameter_string(parameter_string, parameter_type="seeds"):
    # Cleans up and parses the seeds and psi_values input strings into lists of valid values
    # seeds: integers and ranges of integers >= 0 are valid
    # psi_values: integers and floats, but no ranges
    # Start with regexing out anything not in (0-9 - , .), split by comma and remove empty strings
    parameter_values = re.sub(r"[^0-9\.,-]", "", parameter_string).split(",")
    parameter_values = [value for value in parameter_values if value != ""]
    result_list = []
    if len(parameter_values) == 0:
        # Nothing of interest found after the initial clean up, return an empty list
        return result_list
    # Check each value for applicable data, depending on the type
    regex_int = re.compile(r"^\d+$|^\d+\-\d+$")
    regex_float = re.compile(r"^-?\d+(?:\.\d+)?$")
    for value in parameter_values:
        if parameter_type == "seeds":
            if regex_int.match(value):
                if "-" in value:
                    # Found range
                    value_range = value.split("-")
                    result_list.extend(
                        range(
                            min(int(value_range[0]), int(value_range[1])),
                            max(int(value_range[0]), int(value_range[1])) + 1,
                        )
                    )
                else:
                    # Found integer
                    result_list.append(int(value))
        elif parameter_type == "psi":
            if regex_float.match(value):
                # Found float
                result_list.append(float(value))
    # Clean up nr.2: Remove duplicates and , , and sort
    for value in result_list:
        # Remove out of range values
        if parameter_type == "seeds":
            if value < 0 or value > 4294967295:
                result_list.remove(value)
        elif parameter_type == "psi":
            if value < -1 or value > 1:
                result_list.remove(value)
            elif "." in str(value):
                # Round floats to 2 decimal points
                value = round(value, 2)
    # Remove duplicates, sort and return
    result_list = list(set(result_list))
    result_list.sort()
    return result_list


def bulk_update_amount(user_amount, seeds, psi_values):
    # Update the amount of images to generate and put the result in the generate button's label
    result_amount = 0
    output_text = ""
    if not user_amount or user_amount == "" or user_amount == 0:
        # User specified the amount as 0 or blank, use seeds and psi_values
        seeds_list = parse_parameter_string(seeds)
        psi_list = parse_parameter_string(psi_values, "psi")
        if len(seeds_list) == 0:
            # No valid seeds found, so no bros
            result_amount = 0
            output_text = "Seed generation specified, but no valid seeds could be parsed from the relevant textbox.\n\nNo bros will be generated :("
        else:
            if len(psi_list) == 0:
                # At least 1 seed but no valid psi values found
                psi_list = [0.65]
            # Valid seeds and psi values found
            result_amount = len(seeds_list) * len(psi_list)
            warning_text = (
                ""
                if result_amount <= 100
                else "WARNING: These settings will result in more than 100 images being generated!!!\n\n"
            )
            output_text = (
                f"{warning_text}Num of seeds:\t\t{len(seeds_list)}\n"
                + f"Num of psi values:\t{len(psi_list)}\n"
                + f"-----------------------------------\n"
                + f"Total bros to gen:\t{result_amount}"
            )
    else:
        # User entered a valid amount, use that
        result_amount = int(user_amount)
        output_text = f"Valid generated image amount specified, will use it to create {result_amount} bros with random seeds and a psi value of 0.65."
    # Return the label with "no" if 0
    result_amount = "no" if result_amount == 0 else result_amount
    output_text = f"<pre>{output_text}</pre>"
    return [f"👬👬 Generate {result_amount} bros 👬👬", output_text]
