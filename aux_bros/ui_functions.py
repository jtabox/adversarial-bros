# Functions called by UI actions & elements.
import os
from datetime import datetime
from PIL import Image
import re

from . import text_variables

# various folders
# the root folder of the project, not the module
root_folder = os.path.dirname(os.path.dirname(__file__))
output_folder = "outputs"
model_folder = os.path.join("models", "gan")

# vars to update module paths, since I moved stuff in their own subfolders, and everything in the universe is now broken
new_stylegan_subfolder = "stylegan3_fun"
new_codeformer_subfolder = "codeformer"
prefix_mapping = {
    "dnnlib": new_stylegan_subfolder,
    "torch_utils": new_stylegan_subfolder,
    "facelib": new_codeformer_subfolder,
}


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


def update_module_path(module_name: str) -> str:
    """Update the module path to the new subfolder structure.

    Args:
        module_name (str): The module name to update.

    Returns:
        str: The updated module name.
    """
    for old_prefix, new_prefix in prefix_mapping.items():
        if module_name.startswith(old_prefix):
            return f"{new_prefix}.{module_name}"
    return module_name


def update_imports_in_code(code_string):
    for old_prefix, new_prefix in prefix_mapping.items():
        from_string_to_replace = f"from {old_prefix}"
        from_string_to_replace_with = f"from {new_prefix}.{old_prefix}"
        import_string_to_replace = f"import {old_prefix}"
        import_string_to_replace_with = f"from {new_prefix} import {old_prefix}"
        code_string = code_string.replace(
            from_string_to_replace, from_string_to_replace_with
        ).replace(import_string_to_replace, import_string_to_replace_with)
    return code_string


def scan_model_folder(model_folder: str) -> list:
    """Scans the model folder for .pkl files and returns a list of them.

    Args:
        model_folder (str): The path to the model folder.

    Returns:
        list: A list of .pkl files in the model folder.
    """
    model_files = []
    if os.path.exists(model_folder) and os.path.isdir(model_folder):
        model_files = [
            f
            for f in os.listdir(model_folder)
            if os.path.isfile(os.path.join(model_folder, f))
            and f.endswith(".pkl")
        ]
    return model_files


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


def open_output_folder(output_folder: str) -> None:
    """Opens the output folder in file explorer.

    Args:
        output_folder (str): The path to the output folder.
    """
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        os.startfile(output_folder)


def set_output_folder(output_folder: str) -> list:
    """Validates and sets the output folder for saving generated images, creating it if it doesn't exist.

    Args:
        output_folder (str): The path to the output folder.

    Returns:
        list: The output folder path and a string to display in the UI.
    """
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
