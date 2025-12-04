"""
场景感知智能体
负责渲染全场景图像，进行场景分析，并将结果存储为长期记忆
"""

from __future__ import annotations

import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.output_parsers import PydanticOutputParser

from .utils.render_controller import MujocoRenderController
from .utils.llm_utils import build_llm, describe_image, describe_multiple_images
from .utils.data_models import PartAnalysisResult, ScenePartAnalysisResult, PartAnalysisLLMResponse, PartMotionTypeResponse
from .utils.prompt_templates import (
    build_scene_analysis_prompt,
    build_part_analysis_prompt,
    build_part_motion_type_prompt
)
from .memory import MemoryManager
from .render_orchestrator import RenderOrchestrator


class SceneAwarenessAgent:
    """场景感知智能体
    
    功能：
    1. 渲染3D模型的全场景图像
    2. 使用LLM分析场景内容
    3. 将分析结果存储为长期记忆
    4. 提供场景查询和检索功能
    """
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        memory_storage_path: str = "./scene_awareness_memory",
        user_hints: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """初始化场景感知智能体
        
        Args:
            llm_model: LLM模型名称，None则使用默认模型
            memory_storage_path: 长期记忆存储路径
            user_hints: 用户提示字典，格式：{part_name: {motion_type, ...}}
        """
        try:
            self.llm = build_llm(model=llm_model) if llm_model else build_llm()
        except Exception as e:
            raise RuntimeError(f"LLM初始化失败: {e}。场景感知智能体需要LLM支持。")
        
        # 不在这里初始化memory，而是在analyze_scene中为每个任务创建独立的记忆库
        self.render_controller: Optional[MujocoRenderController] = None
        self.memory: Optional[MemoryManager] = None
        self.memory_path: str = "./scene_memory"  # 默认记忆路径
        
        # 存储用户提示（绝对正确的信息）
        self.user_hints = user_hints or {}
        
        # 存储场景part分析结果（不保存到记忆库，保存在智能体变量中）
        self.scene_part_analysis: Optional[ScenePartAnalysisResult] = None
        
        # 渲染配置
        self.render_options = {
            "num_views": 9,  # 3x3 视角
            "mosaic": True,
            "save": True,
            "image_quality": "medium"
        }
    
    def analyze_scene(self, xml_path: str, task_instruction: str = "") -> Dict[str, Any]:
        """分析场景的主要入口方法
        
        Args:
            xml_path: XML模型文件路径
            task_instruction: 任务指令，用于指导分析重点
            
        Returns:
            包含分析结果的字典
        """
        start_time = time.time()
        
        try:
            # 验证输入
            if not Path(xml_path).exists():
                return {
                    "success": False,
                    "message": f"XML文件不存在: {xml_path}",
                    "error_details": "文件路径验证失败"
                }
            
            # 使用固定的记忆库路径
            memory_path = "./scene_memory"
            self.memory_path = memory_path  # 保存记忆路径
            self.memory = MemoryManager(memory_path)
            
            # 检查是否已经有渲染结果
            existing_overall_image = None
            render_orchestrator = RenderOrchestrator(memory_path)
            rendering_results = render_orchestrator.load_rendering_results_from_memory()
            if rendering_results.get("overall_image_path") and Path(rendering_results["overall_image_path"]).exists():
                existing_overall_image = rendering_results["overall_image_path"]
                print(f"检测到已有渲染结果，使用已有图像: {existing_overall_image}")
            else:
                print(f"未找到渲染结果，将重新渲染")
            
            # 初始化渲染控制器
            self._initialize_render_controller(xml_path)
            
            # 执行场景分析流水线（传入已有图像路径，避免重复渲染）
            analysis_result = self._run_scene_analysis_pipeline(xml_path, task_instruction, existing_overall_image)
            
            # 存储分析结果到长期记忆（不再使用task_id）
            self._store_scene_analysis_result(analysis_result)
            
            processing_time = time.time() - start_time
            analysis_result["processing_time"] = processing_time
            analysis_result["memory_path"] = memory_path
            
            # 场景描述已生成（不打印详细内容）
            # 从分析结果中提取场景描述
            scene_description = analysis_result.get('analysis_text', '')
            
            return {
                "success": True,
                "message": "场景分析完成",
                "result": analysis_result,
                "scene_description": scene_description  # 单独返回场景描述
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "message": f"场景分析失败: {str(e)}",
                "error_details": str(e),
                "processing_time": processing_time
            }
    
    def _initialize_render_controller(self, xml_path: str) -> None:
        """初始化渲染控制器"""
        self.render_controller = MujocoRenderController(xml_path)
    
    def _run_scene_analysis_pipeline(self, xml_path: str, task_instruction: str, 
                                    existing_overall_image: Optional[str] = None) -> Dict[str, Any]:
        """运行场景分析流水线
        
        Args:
            xml_path: XML文件路径
            task_instruction: 任务指令
            existing_overall_image: 已有的overall图像路径（如果提供，则不重新渲染）
        """
        # 1. 渲染全场景图像（如果有已有图像，则使用已有图像）
        if existing_overall_image:
            scene_data = {
                "image_path": existing_overall_image,
                "image_type": "overall",
                "render_success": True
            }
            print(f"使用已有overall图像: {existing_overall_image}")
        else:
            scene_data = self._render_scene_overall(xml_path, task_instruction)
        
        # 2. 生成场景分析文本
        analysis_text = self._generate_scene_analysis(scene_data, task_instruction)
        
        # 3. 提取场景关键信息
        scene_info = self._extract_scene_info(analysis_text, xml_path)
        
        # 4. 组合最终结果
        return {
            "xml_path": xml_path,
            "task_instruction": task_instruction,
            "image_path": scene_data.get("image_path", ""),
            "image_type": scene_data.get("image_type", "overall"),
            "analysis_text": analysis_text,
            "scene_info": scene_info,
            "timestamp": time.time()
        }
    
    def _render_scene_overall(self, xml_path: str, task_instruction: str) -> Dict[str, Any]:
        """渲染全场景图像"""
        if not self.render_controller or not self.llm:
            return {}
        
        scene_data = {}
        
        try:
            # 渲染原始图像（整体场景）
            # 将场景图保存到记忆库文件夹中
            memory_path = "./scene_memory"
            Path(memory_path).mkdir(parents=True, exist_ok=True)
            original_path = f"{memory_path}/scene_overall_{int(time.time())}.png"
            self.render_controller.render_original(
                num_views=self.render_options["num_views"],
                mosaic=self.render_options["mosaic"],
                save=self.render_options["save"],
                save_path=original_path
            )
            
            scene_data["image_path"] = original_path
            scene_data["image_type"] = "overall"
            scene_data["render_success"] = True
            
        except Exception as e:
            scene_data["error"] = f"场景渲染失败: {str(e)}"
            scene_data["render_success"] = False
        
        return scene_data
    
    def _generate_scene_analysis(self, scene_data: Dict[str, Any], task_instruction: str) -> str:
        """生成场景分析文本"""
        if not scene_data.get("render_success") or not self.llm:
            return "场景分析失败：渲染或LLM不可用"
        
        image_path = scene_data.get("image_path", "")
        
        start_time = time.time()
        
        # 构建场景分析提示（使用模板函数）
        scene_instruction = build_scene_analysis_prompt(task_instruction)
        
        # 重试机制：最多重试3次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                analysis_text = describe_image(self.llm, image_path, instruction=scene_instruction)
                return analysis_text
            except Exception as e:
                error_msg = str(e)
                
                # 如果是连接错误且还有重试机会，等待后重试
                if "Connection" in error_msg or "timeout" in error_msg.lower() or "网络" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 递增等待时间：2秒、4秒、6秒
                        import time as time_module
                        time_module.sleep(wait_time)
                        continue
                
                # 最后一次尝试失败，返回错误信息
                if attempt == max_retries - 1:
                    print(f"✗ 场景分析失败: {error_msg}")
                    # 返回一个默认的描述，而不是错误信息，避免影响后续流程
                    return "场景分析暂时不可用，将使用其他上下文信息进行推理"
        
        return "场景分析暂时不可用，将使用其他上下文信息进行推理"
    
    def _extract_scene_info(self, analysis_text: str, xml_path: str) -> Dict[str, Any]:
        """从分析文本中提取关键场景信息"""
        scene_info = {
            "device_type": "未知",
            "main_function": "未知",
            "total_components": 0,
            "motion_parts": [],
            "fixed_parts": [],
            "key_features": [],
            "complexity_level": "未知"
        }
        
        try:
            # 基于分析文本提取信息（简单的关键词匹配）
            text_lower = analysis_text.lower()
            
            # 设备类型识别
            device_types = {
                "柜子": ["柜", "橱", "抽屉柜", "衣柜", "鞋柜"],
                "桌子": ["桌", "台", "茶几", "书桌"],
                "椅子": ["椅", "凳", "座椅"],
                "床": ["床", "床架", "床头"],
                "冰箱": ["冰箱", "冷藏", "冷冻"],
                "洗衣机": ["洗衣机", "洗涤"],
                "微波炉": ["微波炉", "烤箱"],
                "机械装置": ["机械", "装置", "机构", "传动"]
            }
            
            for device_type, keywords in device_types.items():
                if any(keyword in text_lower for keyword in keywords):
                    scene_info["device_type"] = device_type
                    break
            
            # 运动部件识别
            motion_keywords = ["门", "盖", "抽屉", "拉手", "把手", "旋钮", "开关", "门把手", "门盖", "铰链"]
            fixed_keywords = ["主体", "框架", "底座", "外壳", "箱体", "柜体", "本体", "支架"]
            
            # 简单的关键词计数
            motion_count = sum(1 for keyword in motion_keywords if keyword in text_lower)
            fixed_count = sum(1 for keyword in fixed_keywords if keyword in text_lower)
            
            scene_info["motion_parts"] = [f"运动部件_{i+1}" for i in range(motion_count)]
            scene_info["fixed_parts"] = [f"固定部件_{i+1}" for i in range(fixed_count)]
            scene_info["total_components"] = motion_count + fixed_count
            
            # 复杂度评估
            if scene_info["total_components"] <= 3:
                scene_info["complexity_level"] = "简单"
            elif scene_info["total_components"] <= 6:
                scene_info["complexity_level"] = "中等"
            else:
                scene_info["complexity_level"] = "复杂"
            
            # 关键特征提取
            if "旋转" in text_lower or "转动" in text_lower:
                scene_info["key_features"].append("旋转运动")
            if "滑动" in text_lower or "推拉" in text_lower:
                scene_info["key_features"].append("滑动运动")
            if "铰链" in text_lower or "铰接" in text_lower:
                scene_info["key_features"].append("铰链连接")
            
        except Exception as e:
            scene_info["extraction_error"] = str(e)
        
        return scene_info
    
    def _store_scene_analysis_result(self, analysis_result: Dict[str, Any]) -> None:
        """存储场景分析结果到长期记忆（不再使用task_id）"""
        try:
            # 1. 存储场景描述文本（专门的标记，方便其他智能体调用）
            scene_description = analysis_result.get('analysis_text', '')
            
            # 检查场景描述是否有效（不是错误信息）
            is_valid = self._is_valid_scene_description(scene_description)
            
            # 只有有效的场景描述才存储
            if is_valid:
                # 获取场景信息
                scene_info = analysis_result.get('scene_info', {})
                device_type = scene_info.get("device_type", "未知")
                main_function = scene_info.get("main_function", "未知")
                total_components = scene_info.get("total_components", 0)
                complexity_level = scene_info.get("complexity_level", "未知")
                motion_parts = scene_info.get("motion_parts", [])
                fixed_parts = scene_info.get("fixed_parts", [])
                key_features = scene_info.get("key_features", [])
                task_instruction = analysis_result.get("task_instruction", "")
                
                # 构建完整的场景描述上下文（包含总结和详细分析）
                scene_summary = f"""场景总结：
- 设备类型：{device_type}
- 主要功能：{main_function}
- 组件数量：{total_components}
- 复杂度：{complexity_level}"""
                
                if motion_parts:
                    scene_summary += f"\n- 运动部件：{', '.join(motion_parts[:5])}"  # 最多显示5个
                    if len(motion_parts) > 5:
                        scene_summary += f" 等共{len(motion_parts)}个"
                
                if fixed_parts:
                    scene_summary += f"\n- 固定部件：{', '.join(fixed_parts[:5])}"  # 最多显示5个
                    if len(fixed_parts) > 5:
                        scene_summary += f" 等共{len(fixed_parts)}个"
                
                if key_features:
                    scene_summary += f"\n- 关键特征：{', '.join(key_features[:5])}"  # 最多显示5个
                
                if task_instruction:
                    scene_summary += f"\n- 任务指令：{task_instruction}"
                
                scene_description_content = f"""{scene_summary}

详细场景描述：

{scene_description}
"""
                scene_description_metadata = {
                    "memory_category": "scene_description",  # 专门标记
                    "memory_type": "scene_description",
                    "is_valid": True,  # 标记为有效
                    "xml_path": analysis_result.get("xml_path", ""),
                    "image_path": analysis_result.get("image_path", ""),
                    "device_type": device_type,
                    "main_function": main_function,
                    "total_components": total_components,
                    "complexity_level": complexity_level,
                    "motion_parts": motion_parts,
                    "fixed_parts": fixed_parts,
                    "key_features": key_features,
                    "task_instruction": task_instruction,
                    "timestamp": analysis_result.get("timestamp", time.time())
                }
                scene_description_id = self.memory.store_long(scene_description_content, scene_description_metadata)
                # 场景描述已存储（不打印）
            # 无效的场景描述不存储（不打印）
            
            # 2. 存储场景分析结果（包含结构化信息）
            memory_content = f"""
场景分析结果：
- 设备类型：{analysis_result.get('scene_info', {}).get('device_type', '未知')}
- 主要功能：{analysis_result.get('scene_info', {}).get('main_function', '未知')}
- 组件数量：{analysis_result.get('scene_info', {}).get('total_components', 0)}
- 复杂度：{analysis_result.get('scene_info', {}).get('complexity_level', '未知')}
- 任务指令：{analysis_result.get('task_instruction', '无')}

详细分析：
{analysis_result.get('analysis_text', '')}
"""
            
            # 构建元数据
            metadata = {
                "memory_category": "scene_analysis",  # 标记为场景分析
                "xml_path": analysis_result.get("xml_path", ""),
                "image_path": analysis_result.get("image_path", ""),
                "device_type": analysis_result.get("scene_info", {}).get("device_type", "未知"),
                "total_components": analysis_result.get("scene_info", {}).get("total_components", 0),
                "complexity_level": analysis_result.get("scene_info", {}).get("complexity_level", "未知"),
                "task_instruction": analysis_result.get("task_instruction", ""),
                "timestamp": analysis_result.get("timestamp", time.time())
            }
            
            # 存储到长期记忆
            memory_id = self.memory.store_long(memory_content, metadata)
            # 场景分析结果已存储（不打印）
            
        except Exception as e:
            print(f"存储场景分析结果到记忆失败: {e}")
    
    def _is_valid_scene_description(self, description: str) -> bool:
        """检查场景描述是否有效
        
        Args:
            description: 场景描述文本
            
        Returns:
            如果描述有效返回True，否则返回False
        """
        if not description or not description.strip():
            return False
        
        # 检查是否是错误信息
        invalid_keywords = [
            "场景分析失败",
            "场景分析暂时不可用",
            "渲染或LLM不可用",
            "Connection error",
            "连接错误",
            "网络错误",
            "timeout",
            "超时"
        ]
        
        description_lower = description.lower()
        for keyword in invalid_keywords:
            if keyword.lower() in description_lower:
                return False
        
        # 检查描述是否太短（可能是错误信息）
        if len(description.strip()) < 20:
            return False
        
        return True
    
    def query_scene_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """查询场景记忆
        
        Args:
            query: 查询关键词
            limit: 返回结果数量限制
            
        Returns:
            匹配的记忆列表
        """
        if not self.memory:
            print("记忆库未初始化，请先运行analyze_scene")
            return []
            
        try:
            memories = self.memory.retrieve(query, memory_type="long", limit=limit)
            return [
                {
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "timestamp": memory.timestamp,
                    "memory_id": memory.id
                }
                for memory in memories
            ]
        except Exception as e:
            print(f"查询场景记忆失败: {e}")
            return []
    
    def query_task_memory(self, task_id: str, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """查询特定任务的记忆
        
        Args:
            task_id: 任务ID
            query: 查询关键词，为空则返回所有记录
            limit: 返回结果数量限制
            
        Returns:
            匹配的记忆列表
        """
        try:
            # 创建临时记忆管理器来访问特定任务的记忆
            memory_path = "./scene_memory"
            temp_memory = MemoryManager(memory_path)
            
            if query:
                memories = temp_memory.retrieve(query, memory_type="long", limit=limit)
            else:
                # 如果没有查询词，返回所有记录
                memories = temp_memory.retrieve("场景分析结果", memory_type="long", limit=limit)
            
            return [
                {
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "timestamp": memory.timestamp,
                    "memory_id": memory.id
                }
                for memory in memories
            ]
        except Exception as e:
            print(f"查询任务 {task_id} 的记忆失败: {e}")
            return []
    
    def get_scene_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取场景分析历史"""
        return self.query_scene_memory("场景分析结果", limit=limit)
    
    def clear_scene_memory(self) -> None:
        """清除场景记忆"""
        if self.memory:
            self.memory.clear_long()
            print("场景记忆已清除")
        else:
            print("记忆库未初始化")
    
    def _clear_scene_analysis_memories_only(self) -> None:
        """只清空场景分析相关的记忆，保留渲染结果
        
        渲染结果的metadata中包含rendering_type字段（"overall"或"part_highlighted"）
        场景分析结果的metadata中包含memory_category字段（"scene_description"或"scene_analysis"）
        """
        if not self.memory:
            return
        
        try:
            # 获取所有长记忆
            all_memories = self.memory.long_memory._memories.copy()
            
            # 过滤出需要保留的记忆（渲染结果）
            memories_to_keep = []
            memories_to_remove = []
            
            for memory in all_memories:
                metadata = memory.metadata or {}
                rendering_type = metadata.get("rendering_type")
                memory_category = metadata.get("memory_category")
                
                # 保留渲染结果（rendering_type为"overall"或"part_highlighted"）
                if rendering_type in ["overall", "part_highlighted"]:
                    memories_to_keep.append(memory)
                # 保留可视化结果（visualization_type为"axis_analysis"）
                elif metadata.get("visualization_type") == "axis_analysis":
                    memories_to_keep.append(memory)
                # 清空场景分析相关的记忆
                elif memory_category in ["scene_description", "scene_analysis"]:
                    memories_to_remove.append(memory)
                # 清空part分析结果（analysis_type为"part_analysis"）
                elif metadata.get("analysis_type") == "part_analysis":
                    memories_to_remove.append(memory)
                # 其他记忆保留（可能是其他类型的记忆）
                else:
                    memories_to_keep.append(memory)
            
            # 更新记忆库
            self.memory.long_memory._memories = memories_to_keep
            self.memory.long_memory._save_memories()
            
            print(f"已清空 {len(memories_to_remove)} 条场景分析相关的记忆，保留 {len(memories_to_keep)} 条其他记忆（包括渲染结果）")
            
        except Exception as e:
            print(f"清空场景分析记忆失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load_scene_from_memory(self) -> Optional[Dict[str, Any]]:
        """从记忆库加载场景信息（选择最新的）
        
        Returns:
            场景信息字典，如果未找到则返回None
        """
        try:
            # 创建临时记忆管理器
            memory_path = "./scene_memory"
            temp_memory = MemoryManager(memory_path)
            
            # 查询场景分析结果（选择最新的）
            memories = temp_memory.retrieve("场景分析结果", memory_type="long", limit=100)
            
            if not memories:
                print(f"未找到场景分析结果")
                return None
            
            # 选择最新的记忆
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            latest_memory = memories[0]
            
            # 解析场景信息
            scene_info = {
                "device_type": latest_memory.metadata.get("device_type", "未知"),
                "total_components": latest_memory.metadata.get("total_components", 0),
                "complexity_level": latest_memory.metadata.get("complexity_level", "未知"),
                "xml_path": latest_memory.metadata.get("xml_path", ""),
                "image_path": latest_memory.metadata.get("image_path", ""),
                "task_instruction": latest_memory.metadata.get("task_instruction", ""),
                "analysis_text": latest_memory.content,
                "timestamp": latest_memory.timestamp
            }
            
            print(f"成功加载场景信息")
            return scene_info
            
        except Exception as e:
            print(f"加载场景信息失败: {e}")
            return None
    
    def analyze_part_with_memory(self, task_id: str, part_name: str) -> Dict[str, Any]:
        """基于记忆库分析特定part（从记忆中读取渲染好的图像）
        
        Args:
            task_id: 任务ID
            part_name: 要分析的part名称
            
        Returns:
            包含part分析结果的字典
        """
        start_time = time.time()
        
        try:
            # 1. 从记忆库加载场景信息
            scene_info = self.load_scene_from_memory(task_id)
            if not scene_info:
                return {
                    "success": False,
                    "message": f"无法加载任务 {task_id} 的场景信息",
                    "error_details": "记忆库中未找到相关记录"
                }
            
            # 2. 从记忆中读取渲染结果
            render_orchestrator = RenderOrchestrator(self.memory_path if self.memory else "./scene_memory")
            rendering_results = render_orchestrator.load_rendering_results_from_memory(task_id)
            
            overall_image_path = rendering_results.get("overall_image_path")
            part_image_path = rendering_results.get("part_images", {}).get(part_name)
            
            if not overall_image_path:
                return {
                    "success": False,
                    "message": f"未找到任务 {task_id} 的 overall 渲染图像",
                    "error_details": "记忆中未找到 overall 渲染结果"
                }
            
            if not part_image_path:
                return {
                    "success": False,
                    "message": f"未找到 part {part_name} 的高亮渲染图像",
                    "error_details": f"记忆中未找到 part {part_name} 的渲染结果"
                }
            
            # 3. 从记忆中读取 part 的颜色映射
            part_memories = self.memory.retrieve(f"Part高亮渲染结果 - {part_name}", memory_type="long", limit=1)
            color_mapping = {}
            if part_memories:
                color_mapping = part_memories[0].metadata.get("color_mapping", {})
            
            highlight_data = {
                "image_path": part_image_path,
                "image_type": "highlighted",
                "color_mapping": color_mapping,
                "render_success": True
            }
            
            # 4. 获取场景描述（作为上下文）
            scene_description = self.get_scene_description(task_id)
            
            # 5. 基于场景记忆、场景描述、overall图像和part高亮图像进行分析
            part_analysis = self._analyze_part_with_context(
                part_name, scene_info, highlight_data, overall_image_path, scene_description
            )
            
            # 5. 存储part分析结果到记忆库（不再使用task_id）
            self._store_part_analysis_result(part_analysis, part_name)
            
            processing_time = time.time() - start_time
            part_analysis["processing_time"] = processing_time
            
            return {
                "success": True,
                "message": f"part {part_name} 分析完成",
                "result": part_analysis
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "message": f"part {part_name} 分析失败: {str(e)}",
                "error_details": str(e),
                "processing_time": processing_time
            }
    
    def _render_part_highlighted(self, part_name: str, task_id: str) -> Dict[str, Any]:
        """渲染特定part的高亮图像"""
        if not self.render_controller:
            return {"render_success": False, "error": "渲染控制器未初始化"}
        
        highlight_data = {}
        
        try:
            # 设置高亮
            self.render_controller.set_highlights([part_name])
            
            # 渲染高亮图像
            # 将图片保存到记忆库文件夹中
            memory_path = "./scene_memory"
            Path(memory_path).mkdir(parents=True, exist_ok=True)
            highlighted_path = f"{memory_path}/part_highlighted_{part_name}_{int(time.time())}.png"
            self.render_controller.render(
                num_views=self.render_options["num_views"],
                mosaic=self.render_options["mosaic"],
                save=self.render_options["save"],
                save_path=highlighted_path
            )
            
            # 获取颜色映射
            color_mapping = self.render_controller.get_highlight_color_mapping(fmt="hex")
            
            highlight_data["image_path"] = highlighted_path
            highlight_data["image_type"] = "highlighted"
            highlight_data["color_mapping"] = color_mapping
            highlight_data["render_success"] = True
            
        except Exception as e:
            highlight_data["error"] = f"part {part_name} 高亮渲染失败: {str(e)}"
            highlight_data["render_success"] = False
        
        return highlight_data
    
    def _render_part_highlighted_with_controller(self, part_name: str, task_id: str, 
                                               render_controller: MujocoRenderController) -> Dict[str, Any]:
        """使用指定的渲染控制器渲染特定part的高亮图像"""
        highlight_data = {}
        
        try:
            # 设置高亮
            render_controller.set_highlights([part_name])
            
            # 渲染高亮图像
            # 将图片保存到记忆库文件夹中
            memory_path = "./scene_memory"
            Path(memory_path).mkdir(parents=True, exist_ok=True)
            highlighted_path = f"{memory_path}/part_highlighted_{part_name}_{int(time.time())}.png"
            render_controller.render(
                num_views=self.render_options["num_views"],
                mosaic=self.render_options["mosaic"],
                save=self.render_options["save"],
                save_path=highlighted_path
            )
            
            # 获取颜色映射
            color_mapping = render_controller.get_highlight_color_mapping(fmt="hex")
            
            highlight_data["image_path"] = highlighted_path
            highlight_data["image_type"] = "highlighted"
            highlight_data["color_mapping"] = color_mapping
            highlight_data["render_success"] = True
            
        except Exception as e:
            highlight_data["error"] = f"part {part_name} 高亮渲染失败: {str(e)}"
            highlight_data["render_success"] = False
        
        return highlight_data
    
    def _analyze_part_with_context(self, part_name: str, scene_info: Dict[str, Any], 
                                 highlight_data: Dict[str, Any],
                                 overall_image_path: Optional[str] = None,
                                 scene_description: Optional[str] = None) -> Dict[str, Any]:
        """基于场景上下文分析特定part（支持同时输入 overall 和 part 高亮图像）
        
        Args:
            part_name: part名称
            scene_info: 场景信息
            highlight_data: part高亮渲染数据
            overall_image_path: overall图像路径（可选，如果提供则一起分析）
            scene_description: 场景描述文本（作为上下文）
        """
        # 检查是否有用户提示（优先级最高）
        user_hint = None
        matched_key = None
        
        if part_name in self.user_hints:
            user_hint = self.user_hints[part_name]
            matched_key = part_name
        else:
            # 模糊匹配
            for hint_key, hint_value in self.user_hints.items():
                if hint_key.lower() in part_name.lower() or part_name.lower() in hint_key.lower():
                    user_hint = hint_value
                    matched_key = hint_key
                    break
        
        if user_hint and "motion_type" in user_hint:
            # 使用用户提示的运动类型，其他信息仍需要LLM分析
            motion_type = user_hint.get("motion_type", "unknown")
            print(f"💡 使用用户提示的运动类型: {part_name} (匹配自: {matched_key}) -> {motion_type}")
            # 继续使用LLM分析其他信息，但强制使用用户提示的运动类型
            # 这里先标记，后续会覆盖LLM返回的motion_type
        
        if not highlight_data.get("render_success") or not self.llm:
            return {"analysis_text": "part分析失败：渲染或LLM不可用"}
        
        part_image_path = highlight_data.get("image_path", "")
        color_mapping = highlight_data.get("color_mapping", {})
        
        # 构建 part 分析提示（使用模板函数）
        part_instruction = build_part_analysis_prompt(
            part_name=part_name,
            scene_info=scene_info,
            scene_description=scene_description,
            color_mapping=color_mapping
        )
        
        try:
            # 如果提供了 overall 图像路径，则一起分析两张图像
            if overall_image_path and Path(overall_image_path).exists():
                # 使用多图像分析：overall图像 + part高亮图像
                analysis_text = describe_multiple_images(
                    self.llm, 
                    [overall_image_path, part_image_path],
                    instruction=part_instruction
                )
            else:
                # 仅分析part高亮图像
                analysis_text = describe_image(self.llm, part_image_path, instruction=part_instruction)
            
            # 使用PydanticOutputParser解析LLM输出（首选方案）
            parser = PydanticOutputParser(pydantic_object=PartAnalysisLLMResponse)
            llm_response = parser.parse(analysis_text)
            
            # 统一运动类型为英文（转换中文到英文）
            motion_type = llm_response.motion_type.lower()
            motion_type_mapping = {
                "固定": "fixed",
                "滑动": "sliding",
                "旋转": "rotating",
                "旋转的": "rotating",
                "滑动的": "sliding",
                "固定的": "fixed"
            }
            if motion_type in motion_type_mapping:
                motion_type = motion_type_mapping[motion_type]
            elif motion_type not in ["fixed", "sliding", "rotating"]:
                # 如果既不是中文也不是标准英文，尝试模糊匹配
                if "固定" in motion_type or "stationary" in motion_type or "不动" in motion_type:
                    motion_type = "fixed"
                elif "滑动" in motion_type or "slide" in motion_type:
                    motion_type = "sliding"
                elif "旋转" in motion_type or "rotate" in motion_type or "转动" in motion_type:
                    motion_type = "rotating"
                else:
                    motion_type = "unknown"
            
            # 如果用户提示了运动类型，优先使用用户提示（绝对正确）
            if user_hint and "motion_type" in user_hint:
                motion_type = user_hint.get("motion_type", motion_type)
            
            # 转换为part_info格式
            part_info = {
                "function": llm_response.function,
                "motion_type": motion_type,
                "position": llm_response.position,
                "material": llm_response.material,
                "confidence": llm_response.confidence,
                "detailed_position": llm_response.detailed_position,
                "specific_function": llm_response.specific_function,
                "motion_description": llm_response.motion_description,
                "motion_axis": llm_response.motion_axis,
                "motion_range": llm_response.motion_range,
                "interaction_method": llm_response.interaction_method,
                "relative_to_ground": llm_response.relative_to_ground,
                "connection_type": llm_response.connection_type,
                "importance_level": llm_response.importance_level
            }
            
            return {
                "part_name": part_name,
                "analysis_text": analysis_text,
                "part_info": part_info,
                "scene_context": {
                    "device_type": scene_info["device_type"],
                    "total_components": scene_info["total_components"],
                    "complexity_level": scene_info["complexity_level"]
                },
                "image_path": part_image_path,
                "overall_image_path": overall_image_path,
                "color_mapping": color_mapping,
                "timestamp": time.time()
            }
            
        except Exception as e:
            raise RuntimeError(f"Part {part_name} 分析失败: {str(e)}")
    
    
    
    def _store_part_analysis_result(self, part_analysis: Dict[str, Any], part_name: str) -> None:
        """存储part分析结果到记忆库（不再使用task_id）"""
        try:
            # 创建临时记忆管理器
            memory_path = "./scene_memory"
            temp_memory = MemoryManager(memory_path)
            
            # 构建记忆内容
            memory_content = f"""
Part分析结果 - {part_name}：
- 功能：{part_analysis.get('part_info', {}).get('function', '未知')}
- 运动类型：{part_analysis.get('part_info', {}).get('motion_type', '未知')}
- 位置：{part_analysis.get('part_info', {}).get('position', '未知')}
- 材质：{part_analysis.get('part_info', {}).get('material', '未知')}
- 置信度：{part_analysis.get('part_info', {}).get('confidence', 0.5):.2f}

详细语义信息：
- 详细位置：{part_analysis.get('part_info', {}).get('detailed_position', '未知')}
- 具体功能：{part_analysis.get('part_info', {}).get('specific_function', '未知')}
- 运动描述：{part_analysis.get('part_info', {}).get('motion_description', '未知')}
- 交互方式：{part_analysis.get('part_info', {}).get('interaction_method', '未知')}
- 相对地面：{part_analysis.get('part_info', {}).get('relative_to_ground', '未知')}
- 连接方式：{part_analysis.get('part_info', {}).get('connection_type', '未知')}
- 重要性：{part_analysis.get('part_info', {}).get('importance_level', '未知')}

场景上下文：
- 设备类型：{part_analysis.get('scene_context', {}).get('device_type', '未知')}
- 组件数量：{part_analysis.get('scene_context', {}).get('total_components', 0)}

高亮渲染图片：{part_analysis.get('image_path', '无')}

详细分析：
{part_analysis.get('analysis_text', '')}
"""
            
            # 构建元数据
            metadata = {
                "part_name": part_name,
                "analysis_type": "part_analysis",
                "function": part_analysis.get('part_info', {}).get('function', '未知'),
                "motion_type": part_analysis.get('part_info', {}).get('motion_type', '未知'),
                "position": part_analysis.get('part_info', {}).get('position', '未知'),
                "material": part_analysis.get('part_info', {}).get('material', '未知'),
                "confidence": part_analysis.get('part_info', {}).get('confidence', 0.5),
                "image_path": part_analysis.get("image_path", ""),
                "timestamp": part_analysis.get("timestamp", time.time())
            }
            
            # 存储到记忆库
            memory_id = temp_memory.store_long(memory_content, metadata)
            
            print(f"Part {part_name} 分析结果已存储到记忆库，ID: {memory_id}")
            
        except Exception as e:
            print(f"存储Part {part_name} 分析结果到记忆失败: {e}")
    
    def analyze_all_parts_with_memory(self, max_workers: int = 4) -> Dict[str, Any]:
        """多线程分析场景中的所有part（从记忆中读取渲染好的图像）
        
        Args:
            max_workers: 最大线程数
            
        Returns:
            包含所有part分析结果的字典
        """
        start_time = time.time()
        
        try:
            # 1. 从记忆库加载场景信息（选择最新的）
            # 加载场景信息（静默加载）
            scene_info = self.load_scene_from_memory()
            if not scene_info:
                return {
                    "success": False,
                    "message": f"无法加载场景信息",
                    "error_details": "记忆库中未找到相关记录"
                }
            
            # 2. 从记忆中读取渲染结果（选择最新的）
            # 从记忆中读取渲染结果（静默加载）
            render_orchestrator = RenderOrchestrator(self.memory_path if self.memory else "./scene_memory")
            rendering_results = render_orchestrator.load_rendering_results_from_memory(verbose=True)
            
            overall_image_path = rendering_results.get("overall_image_path")
            part_images = rendering_results.get("part_images", {})
            
            # 渲染结果已加载（不打印）
            
            if not overall_image_path:
                return {
                    "success": False,
                    "message": f"未找到 overall 渲染图像",
                    "error_details": "记忆中未找到 overall 渲染结果"
                }
            
            if not part_images:
                return {
                    "success": False,
                    "message": f"未找到任何 part 渲染图像",
                    "error_details": "记忆中未找到 part 渲染结果"
                }
            
            # 3. 获取所有part名称（从渲染结果中获取）
            all_parts = list(part_images.keys())
            
            print(f"📊 分析 {len(all_parts)} 个part的运动类型（{max_workers} 线程）...")
            
            # 4. 多线程分析所有part（从记忆中读取图像）
            parts_analysis_results = self._analyze_parts_parallel_from_memory(
                all_parts, scene_info, max_workers, overall_image_path, part_images
            )
            
            # 5. 分类part（运动部件、固定部件、未知部件）
            motion_parts, fixed_parts, unknown_parts = self._classify_parts(parts_analysis_results)
            
            # 6. 创建场景part分析结果（不再使用task_id）
            total_processing_time = time.time() - start_time
            self.scene_part_analysis = ScenePartAnalysisResult(
                task_id="",  # 不再使用task_id
                scene_info=scene_info,
                parts_analysis=parts_analysis_results,
                motion_parts=motion_parts,
                fixed_parts=fixed_parts,
                unknown_parts=unknown_parts,
                total_processing_time=total_processing_time,
                analysis_timestamp=time.time()
            )
            
            print(f"✓ Part分析完成: {len(motion_parts)}个运动, {len(fixed_parts)}个固定 ({total_processing_time:.1f}秒)")
            
            return {
                "success": True,
                "message": f"成功分析 {len(all_parts)} 个part",
                "result": self.scene_part_analysis
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "message": f"多线程part分析失败: {str(e)}",
                "error_details": str(e),
                "processing_time": processing_time
            }
    
    def _get_all_parts_from_xml(self, xml_path: str) -> List[str]:
        """从XML中获取所有part名称"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            parts = []
            for geom in root.iter("geom"):
                if geom.get("type") == "mesh":
                    mesh_name = geom.get("mesh")
                    if mesh_name:
                        parts.append(mesh_name)
            
            return parts
        except Exception as e:
            print(f"解析XML文件失败: {e}")
            return []
    
    def _analyze_parts_parallel_from_memory(self, all_parts: List[str], scene_info: Dict[str, Any], 
                                          max_workers: int,
                                          overall_image_path: str, part_images: Dict[str, str]) -> List[PartAnalysisResult]:
        """并行分析所有part（从记忆中读取图像）"""
        parts_analysis_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_part = {
                executor.submit(
                    self._analyze_single_part_from_memory, 
                    part_name, scene_info, overall_image_path, part_images.get(part_name)
                ): part_name
                for part_name in all_parts if part_images.get(part_name)
            }
            
            # 收集结果
            for future in as_completed(future_to_part):
                part_name = future_to_part[future]
                try:
                    part_result = future.result()
                    if part_result:
                        parts_analysis_results.append(part_result)
                        print(f"✓ 完成part分析: {part_name} ({part_result.motion_type}, 置信度: {part_result.confidence:.2f})")
                    else:
                        print(f"✗ part分析失败: {part_name}")
                except Exception as e:
                    print(f"✗ part {part_name} 分析异常: {e}")
        
        return parts_analysis_results
    
    def _analyze_single_part_from_memory(self, part_name: str, scene_info: Dict[str, Any], 
                                       overall_image_path: str, part_image_path: str) -> Optional[PartAnalysisResult]:
        """从记忆中读取图像并分析单个part（用于并行分析）"""
        part_start_time = time.time()
        
        try:
            if not part_image_path or not Path(part_image_path).exists():
                print(f"part {part_name} 图像不存在: {part_image_path}")
                return None
            
            # 从记忆中读取 part 的颜色映射
            memory_path = self.memory_path if self.memory else "./scene_memory"
            temp_memory = MemoryManager(memory_path)
            part_memories = temp_memory.retrieve(f"Part高亮渲染结果 - {part_name}", memory_type="long", limit=1)
            color_mapping = {}
            if part_memories:
                color_mapping = part_memories[0].metadata.get("color_mapping", {})
            
            highlight_data = {
                "image_path": part_image_path,
                "image_type": "highlighted",
                "color_mapping": color_mapping,
                "render_success": True
            }
            
            # 获取场景描述（作为上下文，选择最新的）
            scene_description = self.get_scene_description()
            
            # 完整分析（使用 overall 图像和 part 高亮图像一起分析）
            part_analysis = self._analyze_part_with_context(
                part_name, scene_info, highlight_data, overall_image_path, scene_description
            )
            
            # 再次检查用户提示，确保运动类型正确（用户提示优先级最高）
            user_hint_final = None
            if part_name in self.user_hints:
                user_hint_final = self.user_hints[part_name]
            else:
                for hint_key, hint_value in self.user_hints.items():
                    if hint_key.lower() in part_name.lower() or part_name.lower() in hint_key.lower():
                        user_hint_final = hint_value
                        break
            
            final_motion_type = part_analysis.get('part_info', {}).get('motion_type', 'unknown')
            if user_hint_final and "motion_type" in user_hint_final:
                final_motion_type = user_hint_final.get("motion_type", final_motion_type)
            
            # 创建PartAnalysisResult
            part_result = PartAnalysisResult(
                part_name=part_name,
                function=part_analysis.get('part_info', {}).get('function', 'unknown'),
                motion_type=final_motion_type,
                position=part_analysis.get('part_info', {}).get('position', 'unknown'),
                material=part_analysis.get('part_info', {}).get('material', 'unknown'),
                confidence=part_analysis.get('part_info', {}).get('confidence', 0.5),
                analysis_text=part_analysis.get('analysis_text', ''),
                image_path=part_analysis.get('image_path', ''),
                processing_time=time.time() - part_start_time,
                timestamp=part_analysis.get('timestamp', time.time()),
                # 详细语义信息
                detailed_position=part_analysis.get('part_info', {}).get('detailed_position', 'unknown'),
                specific_function=part_analysis.get('part_info', {}).get('specific_function', 'unknown'),
                motion_description=part_analysis.get('part_info', {}).get('motion_description', 'unknown'),
                motion_axis=part_analysis.get('part_info', {}).get('motion_axis'),
                motion_range=part_analysis.get('part_info', {}).get('motion_range'),
                interaction_method=part_analysis.get('part_info', {}).get('interaction_method', 'unknown'),
                relative_to_ground=part_analysis.get('part_info', {}).get('relative_to_ground', 'unknown'),
                connection_type=part_analysis.get('part_info', {}).get('connection_type', 'unknown'),
                importance_level=part_analysis.get('part_info', {}).get('importance_level', 'unknown')
            )
            
            return part_result
            
        except Exception as e:
            print(f"part {part_name} 分析失败: {e}")
            return None
    
    def _analyze_part_motion_type_only(self, part_name: str, scene_info: Dict[str, Any], 
                                       highlight_data: Dict[str, Any], 
                                       overall_image_path: Optional[str] = None,
                                       scene_description: Optional[str] = None) -> Dict[str, Any]:
        """快速分析part的运动类型（仅返回motion_type）"""
        # 检查是否有用户提示（优先级最高）
        # 支持精确匹配和模糊匹配（如果精确匹配失败，尝试模糊匹配）
        user_hint = None
        matched_key = None
        
        if part_name in self.user_hints:
            user_hint = self.user_hints[part_name]
            matched_key = part_name
        else:
            # 模糊匹配：检查part_name是否包含在用户提示的key中，或用户提示的key是否包含在part_name中
            for hint_key, hint_value in self.user_hints.items():
                if hint_key.lower() in part_name.lower() or part_name.lower() in hint_key.lower():
                    user_hint = hint_value
                    matched_key = hint_key
                    print(f"  💡 模糊匹配用户提示: {hint_key} -> {part_name}")
                    break
        
        if user_hint:
            motion_type = user_hint.get("motion_type", "unknown")
            print(f"💡 使用用户提示的运动类型: {part_name} (匹配自: {matched_key}) -> {motion_type}")
            return {
                "motion_type": motion_type,
                "confidence": 1.0,  # 用户提示的置信度为1.0（绝对正确）
                "brief_reasoning": f"使用用户提示：{user_hint}"
            }
        
        if not highlight_data.get("render_success") or not self.llm:
            return {"motion_type": "unknown", "confidence": 0.0, "brief_reasoning": "Analysis failed"}
        
        part_image_path = highlight_data.get("image_path", "")
        
        # 构建 part 运动类型分析提示（使用模板函数）
        part_instruction = build_part_motion_type_prompt(
            part_name=part_name,
            scene_info=scene_info,
            scene_description=scene_description
        )
        
        try:
            # 如果提供了 overall 图像路径，则一起分析两张图像
            if overall_image_path and Path(overall_image_path).exists():
                # 使用多图像分析：overall图像 + part高亮图像
                analysis_text = describe_multiple_images(
                    self.llm, 
                    [overall_image_path, part_image_path],
                    instruction=part_instruction
                )
            else:
                # 仅分析part高亮图像
                analysis_text = describe_image(self.llm, part_image_path, instruction=part_instruction)
            
            # 使用PydanticOutputParser解析LLM输出
            parser = PydanticOutputParser(pydantic_object=PartMotionTypeResponse)
            motion_response = parser.parse(analysis_text)
            
            # 统一运动类型为英文（转换中文到英文）
            motion_type = motion_response.motion_type.lower()
            motion_type_mapping = {
                "固定": "fixed",
                "滑动": "sliding",
                "旋转": "rotating",
                "旋转的": "rotating",
                "滑动的": "sliding",
                "固定的": "fixed"
            }
            if motion_type in motion_type_mapping:
                motion_type = motion_type_mapping[motion_type]
            elif motion_type not in ["fixed", "sliding", "rotating"]:
                # 如果既不是中文也不是标准英文，尝试模糊匹配
                if "固定" in motion_type or "stationary" in motion_type or "不动" in motion_type:
                    motion_type = "fixed"
                elif "滑动" in motion_type or "slide" in motion_type:
                    motion_type = "sliding"
                elif "旋转" in motion_type or "rotate" in motion_type or "转动" in motion_type:
                    motion_type = "rotating"
                else:
                    motion_type = "unknown"
            
            return {
                "motion_type": motion_type,
                "confidence": motion_response.confidence,
                "brief_reasoning": motion_response.brief_reasoning
            }
            
        except Exception as e:
            raise RuntimeError(f"Part {part_name} motion type analysis failed: {str(e)}")
    
    def _classify_parts(self, parts_analysis_results: List[PartAnalysisResult]) -> Tuple[List[str], List[str], List[str]]:
        """分类part为运动部件、固定部件、未知部件"""
        motion_parts = []
        fixed_parts = []
        unknown_parts = []
        
        for part_result in parts_analysis_results:
            if part_result.motion_type in ["sliding", "rotating"]:
                motion_parts.append(part_result.part_name)
            elif part_result.motion_type == "fixed":
                fixed_parts.append(part_result.part_name)
            else:
                unknown_parts.append(part_result.part_name)
        
        return motion_parts, fixed_parts, unknown_parts
    
    def get_scene_part_analysis(self) -> Optional[ScenePartAnalysisResult]:
        """获取场景part分析结果"""
        return self.scene_part_analysis
    
    def get_motion_parts(self) -> List[str]:
        """获取运动部件列表"""
        if self.scene_part_analysis:
            return self.scene_part_analysis.motion_parts
        return []
    
    def get_fixed_parts(self) -> List[str]:
        """获取固定部件列表"""
        if self.scene_part_analysis:
            return self.scene_part_analysis.fixed_parts
        return []
    
    def get_part_analysis_by_name(self, part_name: str) -> Optional[PartAnalysisResult]:
        """根据名称获取part分析结果"""
        if self.scene_part_analysis:
            return self.scene_part_analysis.get_part_by_name(part_name)
        return None
    
    def get_scene_description(self, summary_only: bool = False) -> Optional[str]:
        """获取场景描述文本（从记忆中读取，选择最新的有效描述）
        
        Args:
            summary_only: 如果为True，只返回总结性信息；如果为False，返回完整描述（包含总结和详细描述）
        
        Returns:
            场景描述文本，如果未找到则返回None
        """
        try:
            memory_path = self.memory_path if self.memory else "./scene_memory"
            temp_memory = MemoryManager(memory_path)
            
            # 查询场景描述（使用专门的标记）
            memories = temp_memory.retrieve("场景描述", memory_type="long", limit=100)
            
            # 查找场景描述记录，优先选择有效的
            scene_descriptions = [m for m in memories if m.metadata.get("memory_category") == "scene_description"]
            if scene_descriptions:
                # 按时间排序
                scene_descriptions.sort(key=lambda x: x.timestamp, reverse=True)
                
                # 优先选择有效的场景描述
                for memory in scene_descriptions:
                    # 检查metadata中的is_valid标记
                    is_valid = memory.metadata.get("is_valid", False)
                    
                    # 提取场景描述文本
                    content = memory.content
                    
                    # 检查描述是否有效（先检查完整内容）
                    if "场景总结：" in content:
                        # 新格式：包含总结和详细描述
                        if summary_only:
                            # 只返回总结部分
                            parts = content.split("详细场景描述：", 1)
                            summary = parts[0].replace("场景总结：", "").strip()
                            if self._is_valid_scene_description(summary) or is_valid:
                                return summary
                        else:
                            # 返回完整描述（包含总结和详细描述）
                            if self._is_valid_scene_description(content) or is_valid:
                                return content
                    else:
                        # 旧格式：只有详细描述，尝试从metadata构建总结
                        if summary_only:
                            # 从metadata构建总结
                            summary = self._build_summary_from_metadata(memory.metadata)
                            if summary:
                                return summary
                        else:
                            # 返回原始内容
                            if self._is_valid_scene_description(content) or is_valid:
                                return content
                
                # 如果没有找到有效的，尝试从metadata构建总结作为fallback
                if summary_only:
                    for memory in scene_descriptions:
                        summary = self._build_summary_from_metadata(memory.metadata)
                        if summary:
                            return summary
                
                return None
            
            return None
            
        except Exception as e:
            print(f"从记忆中获取场景描述失败: {e}")
            return None
    
    def _build_summary_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        """从metadata构建场景总结
        
        Args:
            metadata: 记忆项的metadata
            
        Returns:
            场景总结文本，如果无法构建则返回None
        """
        try:
            device_type = metadata.get("device_type", "未知")
            main_function = metadata.get("main_function", "未知")
            total_components = metadata.get("total_components", 0)
            complexity_level = metadata.get("complexity_level", "未知")
            motion_parts = metadata.get("motion_parts", [])
            fixed_parts = metadata.get("fixed_parts", [])
            key_features = metadata.get("key_features", [])
            task_instruction = metadata.get("task_instruction", "")
            
            summary = f"""场景总结：
- 设备类型：{device_type}
- 主要功能：{main_function}
- 组件数量：{total_components}
- 复杂度：{complexity_level}"""
            
            if motion_parts:
                summary += f"\n- 运动部件：{', '.join(motion_parts[:5])}"
                if len(motion_parts) > 5:
                    summary += f" 等共{len(motion_parts)}个"
            
            if fixed_parts:
                summary += f"\n- 固定部件：{', '.join(fixed_parts[:5])}"
                if len(fixed_parts) > 5:
                    summary += f" 等共{len(fixed_parts)}个"
            
            if key_features:
                summary += f"\n- 关键特征：{', '.join(key_features[:5])}"
            
            if task_instruction:
                summary += f"\n- 任务指令：{task_instruction}"
            
            return summary
            
        except Exception as e:
            print(f"从metadata构建场景总结失败: {e}")
            return None


__all__ = [
    "SceneAwarenessAgent",
]
