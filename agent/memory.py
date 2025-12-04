"""
记忆系统模块
提供长记忆和短记忆功能
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str
    content: str
    timestamp: float
    memory_type: str  # "short" or "long"
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


class MemorySystem(ABC):
    """记忆系统抽象基类"""
    
    @abstractmethod
    def store(self, content: str, memory_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储记忆项"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, memory_type: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """检索记忆项"""
        pass
    
    @abstractmethod
    def clear(self, memory_type: Optional[str] = None) -> None:
        """清除记忆"""
        pass


class ShortTermMemory(MemorySystem):
    """短记忆系统 - 基于内存的临时存储"""
    
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self._memories: List[MemoryItem] = []
        self._counter = 0
    
    def store(self, content: str, memory_type: str = "short", metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储短记忆项"""
        memory_id = f"short_{self._counter}_{int(time.time())}"
        self._counter += 1
        
        item = MemoryItem(
            id=memory_id,
            content=content,
            timestamp=time.time(),
            memory_type=memory_type,
            metadata=metadata or {}
        )
        
        self._memories.append(item)
        
        # 保持最大数量限制
        if len(self._memories) > self.max_items:
            self._memories = self._memories[-self.max_items:]
        
        return memory_id
    
    def retrieve(self, query: str, memory_type: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """检索短记忆项（简单关键词匹配）"""
        results = []
        query_lower = query.lower()
        
        for item in reversed(self._memories):  # 最新的在前
            if memory_type and item.memory_type != memory_type:
                continue
            
            if query_lower in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    def clear(self, memory_type: Optional[str] = None) -> None:
        """清除短记忆"""
        if memory_type:
            self._memories = [m for m in self._memories if m.memory_type != memory_type]
        else:
            self._memories.clear()


class LongTermMemory(MemorySystem):
    """长记忆系统 - 基于文件持久化存储"""
    
    def __init__(self, storage_path: Union[str, Path]):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._memories: List[MemoryItem] = []
        self._load_memories()
    
    def _load_memories(self) -> None:
        """从文件加载记忆"""
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._memories = [MemoryItem.from_dict(item) for item in data]
            except Exception as e:
                print(f"加载记忆文件失败: {e}")
                self._memories = []
    
    def _save_memories(self) -> None:
        """保存记忆到文件"""
        memory_file = self.storage_path / "memories.json"
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                data = [item.to_dict() for item in self._memories]
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆文件失败: {e}")
    
    def store(self, content: str, memory_type: str = "long", metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储长记忆项"""
        memory_id = f"long_{int(time.time())}_{hash(content) % 10000}"
        
        item = MemoryItem(
            id=memory_id,
            content=content,
            timestamp=time.time(),
            memory_type=memory_type,
            metadata=metadata or {}
        )
        
        self._memories.append(item)
        self._save_memories()
        
        return memory_id
    
    def retrieve(self, query: str, memory_type: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """检索长记忆项（简单关键词匹配）"""
        results = []
        query_lower = query.lower()
        
        for item in reversed(self._memories):  # 最新的在前
            if memory_type and item.memory_type != memory_type:
                continue
            
            if query_lower in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    def clear(self, memory_type: Optional[str] = None) -> None:
        """清除长记忆"""
        if memory_type:
            self._memories = [m for m in self._memories if m.memory_type != memory_type]
        else:
            self._memories.clear()
        
        self._save_memories()


class MemoryManager:
    """记忆管理器 - 统一管理短记忆和长记忆"""
    
    def __init__(self, long_term_storage_path: Union[str, Path] = "./agent_memory"):
        self.short_memory = ShortTermMemory()
        self.long_memory = LongTermMemory(long_term_storage_path)
    
    def store_short(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储短记忆"""
        return self.short_memory.store(content, "short", metadata)
    
    def store_long(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储长记忆"""
        return self.long_memory.store(content, "long", metadata)
    
    def retrieve(self, query: str, memory_type: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """检索记忆（从短记忆和长记忆中）"""
        results = []
        
        if memory_type is None or memory_type == "short":
            short_results = self.short_memory.retrieve(query, "short", limit)
            results.extend(short_results)
        
        if memory_type is None or memory_type == "long":
            long_results = self.long_memory.retrieve(query, "long", limit)
            results.extend(long_results)
        
        # 按时间戳排序，最新的在前
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]
    
    def clear_short(self) -> None:
        """清除短记忆"""
        self.short_memory.clear()
    
    def clear_long(self) -> None:
        """清除长记忆"""
        self.long_memory.clear()
    
    def clear_all(self) -> None:
        """清除所有记忆"""
        self.clear_short()
        self.clear_long()


__all__ = [
    "MemoryItem",
    "MemorySystem", 
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryManager",
]
