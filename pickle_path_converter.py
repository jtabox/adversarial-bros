# This script converts the module paths in a pickled object from the old module paths to the new module paths.
# Courtesy of GPT-4o that did all the work for me. I just had to copy and paste the code.

# To run this successfully, the following changes must be made in stylegan3_fun/torch_utils/persistence.py:

# Add this import statement:
# from aux_bros.path_utils import *

# Change the following functions:

# def _module_to_src(module):
#     ...code...
#     if src is None:
#         module = update_module_path(module) <--- Add this line ---
#         src = inspect.getsource(module)
#     ...code...

# def _src_to_module(src):
#     ...code...
#     if module is None:
#         src = update_imports_in_code(src) <--- Add this line ---
#         module_name = "_imported_module_" + uuid.uuid4().hex
#     ...code...

#----------------------------------------------------------------------------

import pickle
import re
import pickletools

# Define the replacements dictionary
replacements = {
    'torch_utils': 'stylegan3_fun.torch_utils',
    'dnnlib': 'stylegan3_fun.dnnlib'
}

current_pickle_file = 'model/BroGANv1.0.0.pkl'
new_pickle_file = 'model/BroGANv1.0.0-upd-paths2.pkl'
new_optimized_pickle_file = 'model/BroGANv1.0.0-upd-paths2-optimized.pkl'

# Main updater function
def update_module_paths(obj, replacements):
    if isinstance(obj, dict):
        print("Got a dict, recursing...")
        # Recursively update keys and values in dictionaries
        return {update_module_paths(k, replacements): update_module_paths(v, replacements) for k, v in obj.items()}
    elif isinstance(obj, list):
        print("Got a list, recursing...")
        # Recursively update elements in lists
        return [update_module_paths(i, replacements) for i in obj]
    elif isinstance(obj, tuple):
        print("Got a tuple, recursing...")
        # Recursively update elements in tuples
        return tuple(update_module_paths(i, replacements) for i in obj)
    elif isinstance(obj, str):
        print("Found str, updating...")
        # Update the module paths in strings
        for old_str, new_str in replacements.items():
            # Replace whole word occurrences of old_str with new_str
            obj = re.sub(r'\b{}\b'.format(re.escape(old_str)), new_str, obj)
        return obj
    else:
        # Return the object unchanged if it's not a dict, list, tuple, or str
        return obj

# Custom Unpickler class that updates the module paths on-the-fly
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Update the module path using the replacements dictionary
        for old_str, new_str in replacements.items():
            if module.startswith(old_str):
                module = module.replace(old_str, new_str)
        return super().find_class(module, name)

# Loads the pickle and sends it for updating
def update_pickle_file(input_path, output_path, replacements):
    # Load the pickled object
    print(f"Loading {input_path} ...")
    with open(input_path, 'rb') as f:
        obj = CustomUnpickler(f).load()

    # Update the module paths in the object
    print("Pickle loaded, updating the module paths ...")
    updated_obj = update_module_paths(obj, replacements)

    # Save the updated object back to a pickle file
    print(f"Updating complete. Saving the updated pickle to {output_path} ...")
    with open(output_path, 'wb') as f:
        pickle.dump(updated_obj, f)

# Update the paths in the pickled model
print(f"Starting the update process...")
update_pickle_file(current_pickle_file, new_pickle_file, replacements)
print(f"Update complete.")

# Verify the updated model
with open(new_pickle_file, 'rb') as f:
    updated_model = pickle.load(f)
    # save the object to a text file
    with open('updated_model.txt', 'w') as f:
        f.write(str(updated_model))
    print("Updated model loaded successfully, check updated_model.txt.")

# Do a pickletools optimization on the updated pickle file
print(f"Optimizing the updated pickle file ...")
with open(new_pickle_file, 'rb') as f:
    data = f.read()
    optimized_data = pickletools.optimize(data)
    with open(new_optimized_pickle_file, 'wb') as f:
        f.write(optimized_data)

# Verify the updated model
with open(new_optimized_pickle_file, 'rb') as f:
    updated_model = pickle.load(f)
    # save the object to a text file
    with open('optimized_updated_model.txt', 'w') as f:
        f.write(str(updated_model))
    print("Optimized updated model loaded successfully, check optimized_updated_model.txt.")

print(f"Done.")
