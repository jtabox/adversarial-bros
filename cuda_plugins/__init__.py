import os
import importlib.util

def load_plugin(plugin_name):
    file_name = f"{plugin_name}.pyd"
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    if os.path.exists(file_path):
        spec = importlib.util.spec_from_file_location(plugin_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Plugin {plugin_name} not found")
