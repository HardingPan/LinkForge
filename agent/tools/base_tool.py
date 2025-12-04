"""
工具基类和工具注册系统
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """工具抽象基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    def validate_params(self, required_params: List[str], **kwargs) -> Optional[str]:
        """验证必需参数"""
        missing = [param for param in required_params if param not in kwargs]
        if missing:
            return f"缺少必需参数: {', '.join(missing)}"
        return None


class FunctionTool(Tool):
    """基于函数的工具实现"""
    
    def __init__(self, name: str, description: str, func: Callable):
        super().__init__(name, description)
        self.func = func
    
    def execute(self, **kwargs) -> ToolResult:
        """执行函数工具"""
        try:
            result = self.func(**kwargs)
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict) and "success" in result:
                return ToolResult(
                    success=result["success"],
                    message=result.get("message", ""),
                    data=result.get("data")
                )
            else:
                return ToolResult(
                    success=True,
                    message="工具执行成功",
                    data={"result": result}
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"工具执行失败: {str(e)}"
            )


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool
    
    def register_function(self, name: str, description: str, func: Callable) -> None:
        """注册函数工具"""
        tool = FunctionTool(name, description, func)
        self.register(tool)
    
    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """列出所有工具"""
        return list(self._tools.values())
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """执行工具"""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                message=f"工具不存在: {name}"
            )
        return tool.execute(**kwargs)
    
    def __contains__(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools
    
    def __getitem__(self, name: str) -> Tool:
        """获取工具（字典式访问）"""
        return self._tools[name]


__all__ = [
    "Tool",
    "ToolResult",
    "FunctionTool", 
    "ToolRegistry",
]
