from server import PromptServer
from aiohttp import web
import json
import warnings

from .utils import get_results

class JYSegment:
    """提示词选择器节点，用于在ComfyUI中动态选择预定义的提示词"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            # 这样可以为 FUNCTION 提供 node_id 参数
            "hidden": { "node_id": "UNIQUE_ID" }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "process"
    CATEGORY = "钧奕"
    
   
    def __init__(self):
        self._last_pairs = None
        
    def process(self, input_image, node_id) -> tuple:
        """处理选择的提示词"""
        try:
            #warnings.warn("ahaha")
            #print(f"处理提示词时出错2")
            res_str = get_results(input_image)
            return (res_str,)
        except Exception as e:
            print(f"处理提示词时出错: {str(e)}")
            return ("",)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "JYSegment": JYSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JYSegment": "钧奕图片分割"
}

