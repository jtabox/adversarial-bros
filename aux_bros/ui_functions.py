# Functions called by UI actions & elements.
import os
from datetime import datetime
from PIL import Image

from . import text_variables, support_functions


def set_gen_seed(seed_text: str = '') -> int:
    """Is called when the user presses the button to set a random seed, or use last gen's seed.
    If called without argument or with an empty string returns -1 for random image generation seed.
    Otherwise parses the string passed and returns the seed.

    Args:
        seed_text (str, optional): The simple_output_text UI element's text, which contains the last generated image's seed. Defaults to "".

    Returns:
        int: The seed requested, or -1 for random.
    """
    # Sets the seed to either random or the last generated seed
    # If called without arg or with erroneous string, return -1
    if not seed_text or seed_text == "" or not str(seed_text).strip().isdigit():
        return -1
    # Called with the last_gen_seed, return that
    return int(str(seed_text).strip())


def open_output_folder(output_folder):
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        os.startfile(output_folder)


def set_output_folder(output_folder):
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        return [
            output_folder,
            f"{text_variables.open_output_folder_button_text} ({output_folder})",
        ]
    elif not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        return [
            output_folder,
            f"{text_variables.open_output_folder_button_text} ({output_folder})",
        ]
    else:
        return [
            os.path.join(os.path.dirname(__file__), "output"),
            f"{text_variables.open_output_folder_button_text} ({os.path.join(os.path.dirname(__file__), 'output')})",
        ]


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


def bulk_update_amount(user_amount, seeds, psi_values):
    # Update the amount of images to generate and put the result in the generate button's label
    result_amount = 0
    output_text = ""
    if not user_amount or user_amount == "" or user_amount == 0:
        # User specified the amount as 0 or blank, use seeds and psi_values
        seeds_list = support_functions.parse_parameter_string(seeds)
        psi_list = support_functions.parse_parameter_string(psi_values, "psi")
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
