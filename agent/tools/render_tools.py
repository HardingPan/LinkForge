"""
渲染相关工具
"""

from __future__ import annotations

from typing import List, Optional, Union
from pathlib import Path

from .base_tool import Tool, ToolResult
from ..utils.render_controller import MujocoRenderController


class RenderHighlightedTool(Tool):
    """渲染高亮图像工具"""
    
    def __init__(self):
        super().__init__(
            name="render_highlighted",
            description="渲染带高亮效果的多视角图像"
        )
    
    def execute(
        self, 
        render_controller: MujocoRenderController,
        save_path: str = "highlighted_render.png",
        **kwargs
    ) -> ToolResult:
        """执行高亮渲染"""
        try:
            if not render_controller:
                return ToolResult(success=False, message="渲染控制器未初始化")
            
            data_url = render_controller.render(
                num_views=9,
                mosaic=True,
                save=True,
                save_path=save_path
            )
            
            return ToolResult(
                success=True,
                message=f"高亮渲染完成: {save_path}",
                data={"data_url": data_url, "save_path": save_path}
            )
        
        except Exception as e:
            return ToolResult(success=False, message=f"渲染失败: {str(e)}")


class RenderOriginalTool(Tool):
    """渲染原始图像工具"""
    
    def __init__(self):
        super().__init__(
            name="render_original",
            description="渲染原始多视角图像（无高亮）"
        )
    
    def execute(
        self,
        render_controller: MujocoRenderController,
        save_path: str = "original_render.png",
        **kwargs
    ) -> ToolResult:
        """执行原始渲染"""
        try:
            if not render_controller:
                return ToolResult(success=False, message="渲染控制器未初始化")
            
            data_url = render_controller.render_original(
                num_views=9,
                mosaic=True,
                save=True,
                save_path=save_path
            )
            
            return ToolResult(
                success=True,
                message=f"原始渲染完成: {save_path}",
                data={"data_url": data_url, "save_path": save_path}
            )
        
        except Exception as e:
            return ToolResult(success=False, message=f"渲染失败: {str(e)}")


class SetHighlightsTool(Tool):
    """设置高亮部件工具"""
    
    def __init__(self):
        super().__init__(
            name="set_highlights",
            description="设置需要高亮的部件"
        )
    
    def execute(
        self,
        render_controller: MujocoRenderController,
        mesh_names: List[str],
        **kwargs
    ) -> ToolResult:
        """执行设置高亮"""
        try:
            if not render_controller:
                return ToolResult(success=False, message="渲染控制器未初始化")
            
            if not mesh_names:
                return ToolResult(success=False, message="未提供要高亮的部件名称")
            
            render_controller.set_highlights(mesh_names)
            
            return ToolResult(
                success=True,
                message=f"已设置高亮部件: {mesh_names}",
                data={"mesh_names": mesh_names}
            )
        
        except Exception as e:
            return ToolResult(success=False, message=f"设置高亮失败: {str(e)}")


__all__ = [
    "RenderHighlightedTool",
    "RenderOriginalTool", 
    "SetHighlightsTool",
]
