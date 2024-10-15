"""
@author: akatz
@title: Depthcrafter Nodes
@nickname: Depthcrafter Nodes
@description: Custom nodes for use with DepthCrafter.
"""

from .nodes import DownloadAndLoadDepthCrafterModel, DepthCrafter

NODE_CONFIG = {
  "DownloadAndLoadDepthCrafterModel": {"class": DownloadAndLoadDepthCrafterModel, "name": "DownloadAndLoadDepthCrafterModel"},
  "DepthCrafter": {"class": DepthCrafter, "name": "Depthcrafter"},
}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]

ascii_art = """
Depthcrafter Nodes Loaded
"""
print(ascii_art)