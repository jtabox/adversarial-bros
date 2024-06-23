# functions to update module paths, since I moved stuff in their own subfolders, and everything in the universe is now broken
new_stylegan_subfolder = 'stylegan3_fun'
new_codeformer_subfolder = 'CodeFormer'
prefix_mapping = {
    'dnnlib': new_stylegan_subfolder,
    'torch_utils': new_stylegan_subfolder,
    'facelib': new_codeformer_subfolder,
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
        code_string = code_string.replace(from_string_to_replace, from_string_to_replace_with).replace(import_string_to_replace, import_string_to_replace_with)
    return code_string