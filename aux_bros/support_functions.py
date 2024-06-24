# Support functions used in various scripts
import re


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


# functions to update module paths, since I moved stuff in their own subfolders, and everything in the universe is now broken
new_stylegan_subfolder = "stylegan3_fun"
new_codeformer_subfolder = "codeformer"
prefix_mapping = {
    "dnnlib": new_stylegan_subfolder,
    "torch_utils": new_stylegan_subfolder,
    "facelib": new_codeformer_subfolder,
}


def update_module_path(module_name):
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
