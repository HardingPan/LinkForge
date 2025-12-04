"""
分析相关工具
"""

from __future__ import annotations

from typing import Optional

from .base_tool import Tool, ToolResult
from ..utils.llm_utils import describe_image


class AnalyzeImageTool(Tool):
    """分析图像工具"""
    
    def __init__(self):
        super().__init__(
            name="analyze_image",
            description="使用LLM分析图像"
        )
    
    def execute(
        self,
        llm,
        image_path: str,
        instruction: str = "请分析这个图像中的3D模型结构",
        **kwargs
    ) -> ToolResult:
        """执行图像分析"""
        try:
            if not llm:
                return ToolResult(success=False, message="LLM未初始化")
            
            if not image_path:
                return ToolResult(success=False, message="未提供图像路径")
            
            result = describe_image(llm, image_path, instruction=instruction)
            
            return ToolResult(
                success=True,
                message="图像分析完成",
                data={"analysis": result, "image_path": image_path}
            )
        
        except Exception as e:
            return ToolResult(success=False, message=f"图像分析失败: {str(e)}")


__all__ = [
    "AnalyzeImageTool",
]
