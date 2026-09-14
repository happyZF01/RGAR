import importlib


def get_obj_from_str(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def instantiate_from_config(config):
    if not config or "target" not in config:
        raise KeyError("Expected a config mapping with a 'target' entry.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))
