"""
约束推理智能体
负责根据part的分析结果推理其具体的运动约束（滑动方向、旋转类型等）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from langchain.output_parsers import PydanticOutputParser

from .utils.llm_utils import build_llm, describe_multiple_images, describe_image
from .utils.data_models import (
    PartAnalysisResult, 
    MotionConstraintResult, 
    MotionConstraintLLMResponse,
    AxisSelectionLLMResponse
)
from .utils.prompt_templates import (
    build_sliding_constraint_prompt,
    build_rotating_constraint_prompt,
    build_axis_selection_prompt
)
from .memory import MemoryManager
from .render_orchestrator import RenderOrchestrator
from .tools.constraint_analysis_tools import AnalyzeMotionTypeTool


class ConstraintReasoningAgent:
    """约束推理智能体
    
    功能：
    1. 接收选择的part及其分析结果
    2. 根据运动类型推理具体的运动约束：
       - sliding: 推理滑动方向
       - rotating: 判断旋转类型（centerline/edge/custom_axis）
    3. 使用LLM分析图像和已有的分析结果
    """
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        memory_storage_path: str = "./scene_memory",
        user_hints: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """初始化约束推理智能体
        
        Args:
            llm_model: LLM模型名称，None则使用默认模型
            memory_storage_path: 记忆存储路径
            user_hints: 用户提示字典，格式：{part_name: {motion_type, sliding_direction, rotation_type, motion_range, ...}}
        """
        try:
            self.llm = build_llm(model=llm_model) if llm_model else build_llm()
        except Exception as e:
            raise RuntimeError(f"LLM初始化失败: {e}。约束推理智能体需要LLM支持。")
        
        self.memory_path = memory_storage_path
        self.memory = MemoryManager(memory_storage_path)
        
        # 存储用户提示（绝对正确的信息）
        self.user_hints = user_hints or {}
        
        # 初始化约束分析工具
        self.analyze_motion_tool = AnalyzeMotionTypeTool()
        
        # 缓存渲染结果，避免重复加载
        self._cached_rendering_results: Optional[Dict[str, Any]] = None
        self._rendering_results_loaded = False
    
    def reason_motion_constraint(
        self, 
        part_name: str,
        part_analysis: Optional[PartAnalysisResult] = None
    ) -> Dict[str, Any]:
        """推理part的运动约束
        
        Args:
            part_name: 要推理的part名称
            part_analysis: part的分析结果（如果为None，则从记忆中加载）
            
        Returns:
            包含运动约束推理结果的字典
        """
        start_time = time.time()
        
        try:
            # 0. 检查是否有用户提示（优先级最高）
            if part_name in self.user_hints:
                user_hint = self.user_hints[part_name]
                print(f"💡 使用用户提示: {part_name} -> {user_hint}")
                return self._create_result_from_user_hint(part_name, user_hint, part_analysis)
            
            # 1. 获取part分析结果
            if part_analysis is None:
                part_analysis = self._load_part_analysis_from_memory(part_name)
                if not part_analysis:
                    return {
                        "success": False,
                        "message": f"未找到part {part_name} 的分析结果",
                        "error_details": "记忆中未找到相关记录"
                    }
            
            motion_type = part_analysis.motion_type
            
            # 2. 验证运动类型
            if motion_type == "fixed":
                return {
                    "success": False,
                    "message": f"Part {part_name} 是固定部件，无需推理运动约束",
                    "error_details": "固定部件没有运动约束"
                }
            
            if motion_type not in ["sliding", "rotating"]:
                return {
                    "success": False,
                    "message": f"Part {part_name} 的运动类型未知: {motion_type}",
                    "error_details": "无法推理未知运动类型的约束"
                }
            
            # 3. 从记忆中读取渲染图像（使用缓存避免重复加载）
            if not self._rendering_results_loaded:
                # 加载渲染结果（静默加载）
                render_orchestrator = RenderOrchestrator(self.memory_path)
                self._cached_rendering_results = render_orchestrator.load_rendering_results_from_memory(verbose=False)
                self._rendering_results_loaded = True
            
            rendering_results = self._cached_rendering_results or {}
            overall_image_path = rendering_results.get("overall_image_path")
            part_images = rendering_results.get("part_images", {})
            part_image_path = part_images.get(part_name)
            
            # 如果找不到渲染结果，返回错误
            if not overall_image_path or not part_image_path:
                error_details = f"记忆中未找到part {part_name} 的渲染图像"
                error_details += f"\nOverall图像: {overall_image_path is not None}"
                error_details += f"\nPart图像: {part_image_path is not None}"
                error_details += f"\n可用的part图像: {list(part_images.keys())}"
                
                return {
                    "success": False,
                    "message": f"未找到part {part_name} 的渲染图像",
                    "error_details": error_details
                }
            
            # 4. 根据运动类型进行推理（集成工具分析）
            if motion_type == "sliding":
                constraint_result = self._reason_sliding_constraint_with_tool(
                    part_name, part_analysis, overall_image_path, part_image_path
                )
            elif motion_type == "rotating":
                constraint_result = self._reason_rotating_constraint_with_tool(
                    part_name, part_analysis, overall_image_path, part_image_path
                )
            else:
                return {
                    "success": False,
                    "message": f"不支持的运动类型: {motion_type}",
                    "error_details": "只支持sliding和rotating类型"
                }
            
            # 5. 创建MotionConstraintResult
            motion_constraint = MotionConstraintResult(
                part_name=part_name,
                motion_type=motion_type,
                sliding_direction=constraint_result.get("sliding_direction"),
                sliding_orientation=constraint_result.get("sliding_orientation"),
                rotation_type=constraint_result.get("rotation_type"),
                axis_description=constraint_result.get("axis_description"),
                axis_location=constraint_result.get("axis_location"),
                selected_axis=constraint_result.get("selected_axis"),
                selected_axis_id=constraint_result.get("selected_axis_id"),
                selected_axis_info=constraint_result.get("selected_axis_info"),  # 新增
                all_candidate_axes=constraint_result.get("all_candidate_axes"),
                axis_selection_confidence=constraint_result.get("axis_selection_confidence"),
                axis_selection_reasoning=constraint_result.get("axis_selection_reasoning"),
                visualization_path=constraint_result.get("visualization_path"),  # 新增
                motion_range=constraint_result.get("motion_range"),  # 新增
                motion_range_description=constraint_result.get("motion_range_description"),  # 新增
                confidence=constraint_result.get("confidence", 0.5),
                reasoning=constraint_result.get("reasoning", ""),
                timestamp=time.time()
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": f"Part {part_name} 运动约束推理完成",
                "result": motion_constraint,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "message": f"Part {part_name} 运动约束推理失败: {str(e)}",
                "error_details": str(e),
                "processing_time": processing_time
            }
    
    def _load_part_analysis_from_memory(self, part_name: str) -> Optional[PartAnalysisResult]:
        """从记忆中加载part分析结果（选择最新的）"""
        try:
            # 查询part分析结果
            memories = self.memory.retrieve(f"Part分析结果 - {part_name}", memory_type="long", limit=100)
            
            # 选择最新的记录
            if memories:
                memories.sort(key=lambda x: x.timestamp, reverse=True)
                memory = memories[0]
                # 从metadata中重建PartAnalysisResult
                metadata = memory.metadata
                return PartAnalysisResult(
                    part_name=part_name,
                    function=metadata.get("function", "unknown"),
                    motion_type=metadata.get("motion_type", "unknown"),
                    position=metadata.get("position", "unknown"),
                    material=metadata.get("material", "unknown"),
                    confidence=metadata.get("confidence", 0.5),
                    analysis_text=memory.content,
                    image_path=metadata.get("image_path", ""),
                    processing_time=0.0,
                    timestamp=metadata.get("timestamp", time.time()),
                    detailed_position="unknown",
                    specific_function="unknown",
                    motion_description="unknown",
                    motion_axis=None,
                    motion_range=None,
                    interaction_method="unknown",
                    relative_to_ground="unknown",
                    connection_type="unknown",
                    importance_level="unknown"
                )
            
            return None
            
        except Exception as e:
            print(f"从记忆中加载part分析结果失败: {e}")
            return None
    
    def _reason_sliding_constraint_with_tool(
        self,
        part_name: str,
        part_analysis: PartAnalysisResult,
        overall_image_path: str,
        part_image_path: str,
        scene_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """推理滑动部件的滑动方向约束（集成工具分析）
        
        流程：
        1. 直接使用已有的motion_type="sliding"（不再重新推理）
        2. 调用工具分析滑动方向候选
        3. 使用LLM从候选方向中选择最合适的
        """
        try:
            # 1. 获取XML路径（使用缓存）
            if self._cached_rendering_results:
                xml_path = self._cached_rendering_results.get("xml_path")
            else:
                render_orchestrator = RenderOrchestrator(self.memory_path)
                rendering_results = render_orchestrator.load_rendering_results_from_memory(verbose=False)
                xml_path = rendering_results.get("xml_path")
            
            if not xml_path:
                # 回退到原来的方法
                return self._reason_sliding_constraint(
                    part_name, part_analysis, overall_image_path, part_image_path, scene_description
                )
            
            # 2. 调用工具分析滑动方向（将可视化保存到记忆目录）
            memory_output_dir = str(Path(self.memory_path).absolute())
            tool_result = self.analyze_motion_tool.execute(
                xml_path=xml_path,
                part_name=part_name,
                motion_type="sliding",
                visualize=True,
                output_dir=memory_output_dir,
                part_function=part_analysis.function  # 传递part功能信息
            )
            
            if not tool_result.success:
                # 工具失败，回退到原来的方法
                return self._reason_sliding_constraint(
                    part_name, part_analysis, overall_image_path, part_image_path, scene_description
                )
            
            # 3. 提取候选方向
            directions_data = tool_result.data.get("directions", [])
            color_mapping = tool_result.data.get("color_mapping", {})
            index_mapping = tool_result.data.get("index_mapping", {})  # 新增：序号映射
            visualization_path = tool_result.data.get("visualization_path")
            
            # 打印序号映射信息（滑动轴：只显示3个轴）
            if index_mapping:
                print(f"\n📋 序号映射信息 (Index Mapping):")
                for seq_num in sorted(index_mapping.keys()):
                    info = index_mapping[seq_num]
                    print(f"   序号 {seq_num}:")
                    # 对于滑动轴，优先显示轴信息
                    if "axis" in info:
                        print(f"     - 轴: {info['axis']}轴（双向滑动）")
                    if "axis_id" in info:
                        print(f"     - Axis ID: {info['axis_id']}")
                    if "semantic_info" in info:
                        print(f"     - 语义信息: {info['semantic_info']}")
                    if "description" in info:
                        print(f"     - 描述: {info['description']}")
                    if "edge_id" in info:
                        print(f"     - Edge ID: {info['edge_id']}")
                    if "direction_id" in info:
                        print(f"     - Direction ID: {info['direction_id']}")
                    if "reference_direction_id" in info:
                        print(f"     - 参考方向ID: {info['reference_direction_id']}")
                    if "alignment_axis" in info:
                        print(f"     - 对齐轴: {info['alignment_axis']}")
                    if "alignment_score" in info:
                        print(f"     - 对齐分数: {info['alignment_score']:.4f}")
                    if "direction" in info:
                        dir_vec = info["direction"]
                        print(f"     - 正方向向量: [{dir_vec[0]:.4f}, {dir_vec[1]:.4f}, {dir_vec[2]:.4f}]")
                print()
            
            # 将可视化图像路径保存到记忆中（不再使用task_id）
            if visualization_path and Path(visualization_path).exists():
                self._store_visualization_to_memory(
                    part_name, visualization_path, "sliding"
                )
            
            if not directions_data:
                # 没有候选方向，回退到原来的方法
                return self._reason_sliding_constraint(
                    part_name, part_analysis, overall_image_path, part_image_path, scene_description
                )
            
            # 4. 从记忆中加载场景描述和相关信息（如果未提供）
            if scene_description is None:
                # 先尝试加载完整描述
                scene_description = self._load_scene_description_from_memory(summary_only=False)
                if scene_description and not scene_description.startswith("场景分析失败"):
                    print(f"✓ 从记忆中加载场景描述（长度: {len(scene_description)} 字符）")
                else:
                    # 如果完整描述不可用，尝试加载总结性信息
                    scene_summary = self._load_scene_description_from_memory(summary_only=True)
                    if scene_summary:
                        print(f"✓ 从记忆中加载场景总结（长度: {len(scene_summary)} 字符）")
                        scene_description = scene_summary
                    else:
                        print("⚠ 未找到有效的场景描述记忆，继续使用其他上下文信息")
                        scene_description = None  # 设置为None，避免使用错误信息
            
            # 简化的进度信息
            print(f"🔍 分析 {part_name} 的滑动约束（候选方向: {len(directions_data)}个）...")
            
            # 5. 获取AABB信息（从工具结果中，提前获取以便用于空间上下文分析）
            aabb_info = None
            if tool_result.data.get("mesh_info_dict"):
                mesh_info_dict = tool_result.data.get("mesh_info_dict")
                if part_name in mesh_info_dict:
                    mesh_info = mesh_info_dict[part_name]
                    aabb_info = {
                        "size": mesh_info.aabb.size,
                        "center": mesh_info.aabb.center
                    }
            
            # 6. 分析空间上下文（相邻部件、开口方向等）
            spatial_context = self._analyze_spatial_context(
                part_name=part_name,
                part_analysis=part_analysis,
                aabb_info=aabb_info,
                mesh_info_dict=tool_result.data.get("mesh_info_dict"),
                scene_description=scene_description
            )
            
            # 7. 使用LLM从候选方向中选择最合适的（传递AABB和空间上下文信息）
            selection_prompt = build_axis_selection_prompt(
                part_name=part_name,
                part_analysis=part_analysis,
                candidate_axes=directions_data,
                motion_type="sliding",
                visualization_path=visualization_path,
                scene_description=scene_description,
                aabb_info=aabb_info,
                spatial_context=spatial_context,
                index_mapping=index_mapping  # 新增：序号映射
            )
            
            # 准备图像列表（包含可视化图像如果存在）
            images_for_selection = [overall_image_path, part_image_path]
            if visualization_path and Path(visualization_path).exists():
                images_for_selection.append(visualization_path)
            
            selection_text = describe_multiple_images(
                self.llm,
                images_for_selection,
                instruction=selection_prompt
            )
            
            parser = PydanticOutputParser(pydantic_object=AxisSelectionLLMResponse)
            selection_response = parser.parse(selection_text)
            
            # 8. 找到选中的方向
            selected_direction = None
            selected_index = selection_response.selected_index
            if 0 <= selected_index < len(directions_data):
                selected_direction = directions_data[selected_index]
            
            # 9. 同时进行传统的滑动方向推理（用于兼容性，现在包含AABB信息）
            traditional_result = self._reason_sliding_constraint(
                part_name, part_analysis, overall_image_path, part_image_path, scene_description, aabb_info
            )
            
            # 7. 构建详细的选中轴信息用于输出
            selected_axis_info = None
            if selected_direction:
                # 从color_mapping中找到对应的颜色信息
                # 优先通过新的direction_id（序号）匹配，如果没有则通过reference_direction_id匹配
                color_info = None
                selected_direction_id = selection_response.selected_axis_id
                for hex_color, info in color_mapping.items():
                    if info.get("direction_id") == selected_direction_id:
                        color_info = info
                        break
                    # 如果没有匹配，尝试通过reference_direction_id匹配
                    if info.get("reference_direction_id") == selected_direction_id:
                        color_info = info
                        break
                    # 最后尝试通过原始direction_id匹配（向后兼容）
                    if info.get("direction_id") == selected_direction.get("direction_id"):
                        color_info = info
                        break
                
                selected_axis_info = {
                    "axis_id": selection_response.selected_axis_id,  # 使用序号ID（如sliding_direction_1）
                    "reference_direction_id": selected_direction.get("reference_direction_id"),  # 保留原始ID（如positive_y）
                    "index": selected_index,
                    "direction": selected_direction.get("direction"),
                    "direction_id": selected_direction.get("direction_id"),  # 使用序号ID（如sliding_direction_3）
                    "axis": selected_direction.get("axis"),
                    "magnitude": selected_direction.get("magnitude"),
                    "description": selected_direction.get("description"),
                    "confidence": selection_response.confidence,
                    "reasoning": selection_response.reasoning
                }
                
                # 添加颜色信息
                if color_info:
                    selected_axis_info.update({
                        "color_hex": color_info.get("hex"),
                        "color_rgb": color_info.get("rgb"),
                        "color_index": color_info.get("index")
                    })
            
            return {
                "sliding_direction": traditional_result.get("sliding_direction"),
                "sliding_orientation": traditional_result.get("sliding_orientation"),
                "rotation_type": None,
                "axis_description": None,
                "axis_location": None,
                "selected_axis": selected_direction,
                "selected_axis_id": selection_response.selected_axis_id if selected_direction else None,
                "selected_axis_info": selected_axis_info,  # 新增：详细的选中轴信息
                "all_candidate_axes": directions_data,
                "axis_selection_confidence": selection_response.confidence,
                "axis_selection_reasoning": selection_response.reasoning,
                "visualization_path": visualization_path,  # 新增：可视化路径
                "motion_range": traditional_result.get("motion_range"),  # 新增
                "motion_range_description": traditional_result.get("motion_range_description"),  # 新增
                "confidence": max(traditional_result.get("confidence", 0.5), selection_response.confidence),
                "reasoning": f"{traditional_result.get('reasoning', '')}\n\n轴选择推理: {selection_response.reasoning}"
            }
            
        except Exception as e:
            print(f"使用工具分析滑动约束失败，回退到传统方法: {e}")
            # 回退到原来的方法
            return self._reason_sliding_constraint(
                part_name, part_analysis, overall_image_path, part_image_path, scene_description
            )
    
    def _reason_sliding_constraint(
        self,
        part_name: str,
        part_analysis: PartAnalysisResult,
        overall_image_path: str,
        part_image_path: str,
        scene_description: Optional[str] = None,
        aabb_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """推理滑动部件的滑动方向约束（传统方法，不使用工具）"""
        # 构建滑动约束推理提示（使用模板函数，包含AABB信息）
        instruction = build_sliding_constraint_prompt(
            part_name=part_name,
            part_analysis=part_analysis,
            scene_description=scene_description,
            aabb_info=aabb_info
        )
        
        try:
            # 使用多图像分析
            analysis_text = describe_multiple_images(
                self.llm,
                [overall_image_path, part_image_path],
                instruction=instruction
            )
            
            # 解析LLM输出
            parser = PydanticOutputParser(pydantic_object=MotionConstraintLLMResponse)
            llm_response = parser.parse(analysis_text)
            
            return {
                "sliding_direction": llm_response.sliding_direction,
                "sliding_orientation": llm_response.sliding_orientation,
                "rotation_type": None,
                "axis_description": None,
                "axis_location": None,
                "motion_range": self._convert_motion_range_to_symmetric(llm_response.motion_range),  # 转换为对称范围
                "motion_range_description": llm_response.motion_range_description,  # 新增
                "confidence": llm_response.confidence,
                "reasoning": llm_response.reasoning
            }
            
        except Exception as e:
            raise RuntimeError(f"Part {part_name} sliding constraint reasoning failed: {str(e)}")
    
    def _reason_rotating_constraint_with_tool(
        self,
        part_name: str,
        part_analysis: PartAnalysisResult,
        overall_image_path: str,
        part_image_path: str,
        scene_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """推理旋转部件的旋转类型约束（集成工具分析）
        
        流程：
        1. 先快速判断旋转类型（edge还是centerline）- 使用简化的推理，避免重复
        2. 调用工具分析旋转轴候选
        3. 使用LLM从候选轴中选择最合适的
        """
        try:
            # 1. 快速判断旋转类型（edge还是centerline）
            # 优先从part_analysis中获取信息，如果没有则进行简单推理
            rotation_type = None
            axis_description = None
            axis_location = None
            
            # 尝试从part_analysis的motion_description中推断旋转类型
            motion_desc = part_analysis.motion_description.lower() if part_analysis.motion_description else ""
            if "edge" in motion_desc or "边缘" in motion_desc or "边" in motion_desc:
                rotation_type = "edge"
            elif "center" in motion_desc or "中心" in motion_desc or "centerline" in motion_desc:
                rotation_type = "centerline"
            else:
                # 如果无法从描述推断，进行快速推理（不调用LLM，只做简单判断）
                # 默认使用edge（大多数门都是edge旋转）
                rotation_type = "edge"
                axis_description = "vertical edge"
                axis_location = "along the edge"
            
            # 如果没有推断出，使用默认值
            if not rotation_type:
                rotation_type = "edge"
            
            # 2. 获取XML路径（使用缓存）
            if self._cached_rendering_results:
                xml_path = self._cached_rendering_results.get("xml_path")
            else:
                render_orchestrator = RenderOrchestrator(self.memory_path)
                rendering_results = render_orchestrator.load_rendering_results_from_memory(verbose=False)
                xml_path = rendering_results.get("xml_path")
            
            if not xml_path:
                # 没有XML路径，返回基本结果
                return {
                    "sliding_direction": None,
                    "sliding_orientation": None,
                    "rotation_type": rotation_type,
                    "axis_description": axis_description or "rotation axis",
                    "axis_location": axis_location or "to be determined",
                    "confidence": 0.7,
                    "reasoning": f"旋转类型: {rotation_type}（从part分析结果推断），但无法获取XML路径进行工具分析"
                }
            
            # 3. 根据旋转类型调用相应的工具
            # 将rotation_type映射到工具需要的motion_type
            if rotation_type == "edge":
                tool_motion_type = "edge_rotation"
            elif rotation_type == "centerline":
                tool_motion_type = "centerline_rotation"
            else:
                # 对于custom_axis，尝试centerline作为默认
                tool_motion_type = "centerline_rotation"
            
            # 将可视化保存到记忆目录
            memory_output_dir = str(Path(self.memory_path).absolute())
            tool_result = self.analyze_motion_tool.execute(
                xml_path=xml_path,
                part_name=part_name,
                motion_type=tool_motion_type,
                visualize=True,
                output_dir=memory_output_dir,
                part_function=part_analysis.function  # 传递part功能信息
            )
            
            if not tool_result.success:
                # 工具失败，返回基本结果
                return {
                    "sliding_direction": None,
                    "sliding_orientation": None,
                    "rotation_type": rotation_type,
                    "axis_description": axis_description or "rotation axis",
                    "axis_location": axis_location or "to be determined",
                    "confidence": 0.7,
                    "reasoning": f"旋转类型: {rotation_type}（从part分析结果推断），但工具分析失败"
                }
            
            # 4. 提取候选轴
            axes_data = tool_result.data.get("axes", [])
            color_mapping = tool_result.data.get("color_mapping", {})
            index_mapping = tool_result.data.get("index_mapping", {})  # 新增：序号映射
            visualization_path = tool_result.data.get("visualization_path")
            
            # 打印序号映射信息
            if index_mapping:
                print(f"\n📋 序号映射信息 (Index Mapping):")
                for seq_num in sorted(index_mapping.keys()):
                    info = index_mapping[seq_num]
                    print(f"   序号 {seq_num}:")
                    if "semantic_info" in info:
                        print(f"     - 语义信息: {info['semantic_info']}")
                    if "edge_id" in info:
                        print(f"     - Edge ID: {info['edge_id']}")
                    if "direction_id" in info:
                        print(f"     - Direction ID: {info['direction_id']}")
                    if "axis_id" in info:
                        print(f"     - Axis ID: {info['axis_id']}")
                    if "alignment_axis" in info:
                        print(f"     - 对齐轴: {info['alignment_axis']}")
                    if "alignment_score" in info:
                        print(f"     - 对齐分数: {info['alignment_score']:.4f}")
                    if "direction" in info:
                        dir_vec = info["direction"]
                        print(f"     - 方向向量: [{dir_vec[0]:.4f}, {dir_vec[1]:.4f}, {dir_vec[2]:.4f}]")
                print()
            
            # 将可视化图像路径保存到记忆中（不再使用task_id）
            if visualization_path and Path(visualization_path).exists():
                self._store_visualization_to_memory(
                    part_name, visualization_path, tool_motion_type
                )
            
            if not axes_data:
                # 没有候选轴，返回基本结果
                return {
                    "sliding_direction": None,
                    "sliding_orientation": None,
                    "rotation_type": rotation_type,
                    "axis_description": axis_description or "rotation axis",
                    "axis_location": axis_location or "to be determined",
                    "confidence": 0.7,
                    "reasoning": f"旋转类型: {rotation_type}（从part分析结果推断），但工具未找到候选轴"
                }
            
            # 5. 从记忆中加载场景描述和相关信息（如果未提供）
            if scene_description is None:
                # 先尝试加载完整描述
                scene_description = self._load_scene_description_from_memory(summary_only=False)
                if scene_description and not scene_description.startswith("场景分析失败"):
                    print(f"✓ 从记忆中加载场景描述（长度: {len(scene_description)} 字符）")
                else:
                    # 如果完整描述不可用，尝试加载总结性信息
                    scene_summary = self._load_scene_description_from_memory(summary_only=True)
                    if scene_summary:
                        print(f"✓ 从记忆中加载场景总结（长度: {len(scene_summary)} 字符）")
                        scene_description = scene_summary
                    else:
                        print("⚠ 未找到有效的场景描述记忆，继续使用其他上下文信息")
                        scene_description = None  # 设置为None，避免使用错误信息
            
            # 简化的进度信息
            print(f"🔍 分析 {part_name} 的旋转约束（类型: {rotation_type}, 候选轴: {len(axes_data)}个）...")
            
            # 6. 获取AABB信息（从工具结果中，提前获取以便用于空间上下文分析）
            aabb_info = None
            if tool_result.data.get("mesh_info_dict"):
                mesh_info_dict = tool_result.data.get("mesh_info_dict")
                if part_name in mesh_info_dict:
                    mesh_info = mesh_info_dict[part_name]
                    aabb_info = {
                        "size": mesh_info.aabb.size,
                        "center": mesh_info.aabb.center
                    }
            
            # 7. 分析空间上下文（相邻部件、开口方向等）
            spatial_context = self._analyze_spatial_context(
                part_name=part_name,
                part_analysis=part_analysis,
                aabb_info=aabb_info,
                mesh_info_dict=tool_result.data.get("mesh_info_dict"),
                scene_description=scene_description
            )
            
            # 8. 使用LLM从候选轴中选择最合适的（传递AABB和空间上下文信息）
            selection_prompt = build_axis_selection_prompt(
                part_name=part_name,
                part_analysis=part_analysis,
                candidate_axes=axes_data,
                motion_type=tool_motion_type,
                visualization_path=visualization_path,
                scene_description=scene_description,
                aabb_info=aabb_info,
                spatial_context=spatial_context,
                index_mapping=index_mapping  # 新增：序号映射
            )
            
            # 准备图像列表（包含可视化图像如果存在）
            images_for_selection = [overall_image_path, part_image_path]
            if visualization_path and Path(visualization_path).exists():
                images_for_selection.append(visualization_path)
            
            selection_text = describe_multiple_images(
                self.llm,
                images_for_selection,
                instruction=selection_prompt
            )
            
            parser = PydanticOutputParser(pydantic_object=AxisSelectionLLMResponse)
            selection_response = parser.parse(selection_text)
            
            # 9. 找到选中的轴
            selected_axis = None
            selected_index = selection_response.selected_index
            selected_axis_id = selection_response.selected_axis_id
            
            # 打印选中的轴信息
            # 选中的轴信息（不再详细打印）
            
            # 通过index_mapping查找对应的序号
            matched_sequence_number = None
            if index_mapping:
                for seq_num, info in index_mapping.items():
                    if tool_motion_type == "edge_rotation":
                        if info.get("edge_id") == selected_axis_id:
                            matched_sequence_number = seq_num
                            print(f"   - 对应的序号: {seq_num}")
                            if "semantic_info" in info:
                                print(f"   - 语义信息: {info['semantic_info']}")
                            break
                    elif tool_motion_type == "centerline_rotation":
                        if info.get("axis_id") == selected_axis_id:
                            matched_sequence_number = seq_num
                            break
            
            if 0 <= selected_index < len(axes_data):
                selected_axis = axes_data[selected_index]
                # 验证选中的轴ID是否匹配（静默处理，不打印）
                if tool_motion_type == "edge_rotation":
                    actual_edge_id = selected_axis.get("edge_id")
                    if actual_edge_id != selected_axis_id:
                        # 尝试通过edge_id查找正确的axis
                        for idx, axis in enumerate(axes_data):
                            if axis.get("edge_id") == selected_axis_id:
                                selected_axis = axis
                                selected_index = idx
                                break
                elif tool_motion_type == "centerline_rotation":
                    actual_axis_id = selected_axis.get("axis_id")
                    if actual_axis_id != selected_axis_id:
                        # 尝试通过axis_id查找正确的axis
                        for idx, axis in enumerate(axes_data):
                            if axis.get("axis_id") == selected_axis_id:
                                selected_axis = axis
                                selected_index = idx
                                break
            else:
                # 尝试通过ID查找（静默处理）
                if tool_motion_type == "edge_rotation":
                    for idx, axis in enumerate(axes_data):
                        if axis.get("edge_id") == selected_axis_id:
                            selected_axis = axis
                            selected_index = idx
                            break
                elif tool_motion_type == "centerline_rotation":
                    for idx, axis in enumerate(axes_data):
                        if axis.get("axis_id") == selected_axis_id:
                            selected_axis = axis
                            selected_index = idx
                            break
            
            # 10. 进行旋转范围推理（使用传统方法，但包含AABB信息）
            rotation_range_result = self._reason_rotating_constraint(
                part_name, part_analysis, overall_image_path, part_image_path, scene_description, aabb_info
            )
            
            # 9. 构建详细的选中轴信息用于输出
            selected_axis_info = None
            if selected_axis:
                # 从color_mapping中找到对应的颜色信息
                color_info = None
                selected_id = selection_response.selected_axis_id
                for hex_color, info in color_mapping.items():
                    # 根据轴类型匹配不同的ID字段
                    if tool_motion_type == "edge_rotation":
                        if info.get("edge_id") == selected_id:
                            color_info = info
                            break
                    elif tool_motion_type == "centerline_rotation":
                        if info.get("axis_id") == selected_id:
                            color_info = info
                            break
                
                selected_axis_info = {
                    "axis_id": selection_response.selected_axis_id,
                    "index": selected_index,
                    "rotation_type": rotation_type,
                    "motion_type": tool_motion_type,
                }
                # 根据轴类型添加不同信息
                if "midpoint" in selected_axis:
                    # Edge旋转
                    selected_axis_info.update({
                        "midpoint": selected_axis.get("midpoint"),
                        "direction": selected_axis.get("direction"),
                        "length": selected_axis.get("length"),
                        "alignment_axis": selected_axis.get("alignment_axis"),
                        "alignment_score": selected_axis.get("alignment_score"),
                        "edge_id": selected_axis.get("edge_id"),
                    })
                elif "point" in selected_axis:
                    # 中心线旋转
                    selected_axis_info.update({
                        "point": selected_axis.get("point"),
                        "direction": selected_axis.get("direction"),
                        "axis_type": selected_axis.get("axis_type"),
                        "axis_id": selected_axis.get("axis_id"),
                    })
                
                # 添加颜色信息
                if color_info:
                    selected_axis_info.update({
                        "color_hex": color_info.get("hex"),
                        "color_rgb": color_info.get("rgb"),
                        "color_index": color_info.get("index")
                    })
                
                selected_axis_info.update({
                    "confidence": selection_response.confidence,
                    "reasoning": selection_response.reasoning
                })
            
            return {
                "sliding_direction": None,
                "sliding_orientation": None,
                "rotation_type": rotation_type,
                "axis_description": rotation_range_result.get("axis_description") or axis_description or "rotation axis",
                "axis_location": rotation_range_result.get("axis_location") or axis_location or "to be determined",
                "selected_axis": selected_axis,
                "selected_axis_id": selection_response.selected_axis_id if selected_axis else None,
                "selected_axis_info": selected_axis_info,  # 新增：详细的选中轴信息
                "all_candidate_axes": axes_data,
                "axis_selection_confidence": selection_response.confidence,
                "axis_selection_reasoning": selection_response.reasoning,
                "visualization_path": visualization_path,  # 新增：可视化路径
                "motion_range": rotation_range_result.get("motion_range"),  # 新增
                "motion_range_description": rotation_range_result.get("motion_range_description"),  # 新增
                "confidence": selection_response.confidence,  # 使用轴选择的置信度
                "reasoning": f"旋转类型: {rotation_type}（从part分析结果推断）。轴选择推理: {selection_response.reasoning}"
            }
            
        except Exception as e:
            print(f"使用工具分析旋转约束失败，回退到传统方法: {e}")
            # 回退到原来的方法
            return self._reason_rotating_constraint(
                part_name, part_analysis, overall_image_path, part_image_path, scene_description
            )
    
    def _reason_rotating_constraint(
        self,
        part_name: str,
        part_analysis: PartAnalysisResult,
        overall_image_path: str,
        part_image_path: str,
        scene_description: Optional[str] = None,
        aabb_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """推理旋转部件的旋转类型约束（传统方法，不使用工具）"""
        # 构建旋转约束推理提示（使用模板函数，包含AABB信息）
        instruction = build_rotating_constraint_prompt(
            part_name=part_name,
            part_analysis=part_analysis,
            scene_description=scene_description,
            aabb_info=aabb_info
        )
        
        try:
            # 使用多图像分析
            analysis_text = describe_multiple_images(
                self.llm,
                [overall_image_path, part_image_path],
                instruction=instruction
            )
            
            # 解析LLM输出
            parser = PydanticOutputParser(pydantic_object=MotionConstraintLLMResponse)
            llm_response = parser.parse(analysis_text)
            
            return {
                "sliding_direction": None,
                "sliding_orientation": None,
                "rotation_type": llm_response.rotation_type,
                "axis_description": llm_response.axis_description,
                "axis_location": llm_response.axis_location,
                "motion_range": self._convert_motion_range_to_symmetric(llm_response.motion_range),  # 转换为对称范围
                "motion_range_description": llm_response.motion_range_description,  # 新增
                "confidence": llm_response.confidence,
                "reasoning": llm_response.reasoning
            }
            
        except Exception as e:
            raise RuntimeError(f"Part {part_name} rotation constraint reasoning failed: {str(e)}")
    
    def _load_scene_description_from_memory(self, summary_only: bool = False) -> Optional[str]:
        """从记忆中加载场景描述（选择最新的有效描述）
        
        Args:
            summary_only: 如果为True，只返回总结性信息；如果为False，返回完整描述（包含总结和详细描述）
        
        Returns:
            场景描述文本，如果未找到则返回None
        """
        try:
            # 查询场景描述（使用专门的标记）
            memories = self.memory.retrieve("场景描述", memory_type="long", limit=100)
            
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
    
    def _create_result_from_user_hint(
        self, 
        part_name: str, 
        user_hint: Dict[str, Any],
        part_analysis: Optional[PartAnalysisResult] = None
    ) -> Dict[str, Any]:
        """从用户提示创建约束结果（用户提示是绝对正确的）
        
        Args:
            part_name: 部件名称
            user_hint: 用户提示字典
            part_analysis: 部件分析结果（可选，用于补充信息）
            
        Returns:
            包含运动约束推理结果的字典
        """
        motion_type = user_hint.get("motion_type", "unknown")
        
        # 如果是固定部件，直接返回
        if motion_type == "fixed":
            return {
                "success": False,
                "message": f"Part {part_name} 是固定部件（用户提示），无需推理运动约束",
                "error_details": "固定部件没有运动约束"
            }
        
        # 构建基础结果
        result_data = {
            "part_name": part_name,
            "motion_type": motion_type,
            "confidence": 1.0,  # 用户提示的置信度为1.0（绝对正确）
            "reasoning": f"使用用户提示：{user_hint}",
            "timestamp": time.time()
        }
        
        # 处理滑动约束
        if motion_type == "sliding":
            sliding_direction = user_hint.get("sliding_direction")
            if sliding_direction:
                result_data["sliding_direction"] = sliding_direction
                result_data["sliding_orientation"] = f"沿{sliding_direction}轴滑动（用户提示）"
            else:
                result_data["sliding_direction"] = "x"  # 默认值
                result_data["sliding_orientation"] = "滑动方向（用户提示，默认x轴）"
        
        # 处理旋转约束
        elif motion_type == "rotating":
            rotation_type = user_hint.get("rotation_type")
            if rotation_type:
                result_data["rotation_type"] = rotation_type
                result_data["axis_description"] = f"{rotation_type}旋转（用户提示）"
                result_data["axis_location"] = "根据用户提示确定"
            else:
                result_data["rotation_type"] = "centerline"  # 默认值
                result_data["axis_description"] = "中心线旋转（用户提示，默认centerline）"
                result_data["axis_location"] = "中心"
        
        # 处理运动范围
        if "motion_range" in user_hint:
            motion_range_value = user_hint["motion_range"]
            result_data["motion_range"] = self._convert_motion_range_to_symmetric(motion_range_value)
            result_data["motion_range_description"] = f"运动范围：{motion_range_value}（用户提示）"
        
        # 创建MotionConstraintResult对象
        from .utils.data_models import MotionConstraintResult
        result = MotionConstraintResult(**result_data)
        
        return {
            "success": True,
            "message": f"使用用户提示成功推理 {part_name} 的运动约束",
            "result": result,
            "processing_time": 0.0  # 用户提示不需要处理时间
        }
    
    def _convert_motion_range_to_symmetric(self, motion_range_value: Optional[float]) -> Optional[Tuple[float, float]]:
        """将单个motion_range值转换为对称范围
        
        Args:
            motion_range_value: 单个值（如90表示±90度，0.4表示±0.4米）
            
        Returns:
            对称范围元组 (min, max)，如果输入为None则返回None
        """
        if motion_range_value is None:
            return None
        
        # 确保值为正数
        abs_value = abs(motion_range_value)
        
        # 转换为对称范围
        return (-abs_value, abs_value)
    
    def _build_summary_from_metadata(self, metadata: Dict[str, Any]) -> str:
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
    
    def _analyze_spatial_context(
        self,
        part_name: str,
        part_analysis: PartAnalysisResult,
        aabb_info: Optional[Dict[str, Any]],
        mesh_info_dict: Optional[Dict[str, Any]],
        scene_description: Optional[str]
    ) -> Dict[str, Any]:
        """分析空间上下文信息（相邻部件、开口方向等）
        
        Args:
            part_name: 部件名称
            part_analysis: 部件分析结果
            aabb_info: AABB信息
            mesh_info_dict: 所有部件的mesh信息字典
            scene_description: 场景描述
            
        Returns:
            空间上下文信息字典
        """
        spatial_context = {}
        
        # 1. 分析开口方向（基于motion_description和function）
        opening_direction = None
        motion_desc = part_analysis.motion_description.lower() if part_analysis.motion_description else ""
        function_desc = part_analysis.function.lower() if part_analysis.function else ""
        
        # 根据运动描述推断开口方向
        if "往外" in motion_desc or "outward" in motion_desc or "open" in motion_desc:
            # 门往外开或抽屉往外拉，通常是+Y方向（前面）
            opening_direction = {
                "description": "往外开/往外拉 (outward)",
                "direction_vector": [0.0, 1.0, 0.0],  # +Y方向
                "axis": "y",
                "direction": "positive_y"
            }
        elif "往里" in motion_desc or "inward" in motion_desc or "push" in motion_desc:
            # 往里推，通常是-Y方向（后面）
            opening_direction = {
                "description": "往里推 (inward)",
                "direction_vector": [0.0, -1.0, 0.0],  # -Y方向
                "axis": "y",
                "direction": "negative_y"
            }
        elif "左" in motion_desc or "left" in motion_desc:
            opening_direction = {
                "description": "向左 (left)",
                "direction_vector": [-1.0, 0.0, 0.0],  # -X方向
                "axis": "x",
                "direction": "negative_x"
            }
        elif "右" in motion_desc or "right" in motion_desc:
            opening_direction = {
                "description": "向右 (right)",
                "direction_vector": [1.0, 0.0, 0.0],  # +X方向
                "axis": "x",
                "direction": "positive_x"
            }
        
        # 如果没有从描述中推断出，根据功能推断
        if not opening_direction:
            if "门" in function_desc or "door" in function_desc:
                # 门通常往外开（+Y方向）
                opening_direction = {
                    "description": "门往外开 (door opens outward)",
                    "direction_vector": [0.0, 1.0, 0.0],
                    "axis": "y",
                    "direction": "positive_y"
                }
            elif "抽屉" in function_desc or "drawer" in function_desc:
                # 抽屉通常往外拉（+Y方向）
                opening_direction = {
                    "description": "抽屉往外拉 (drawer pulls outward)",
                    "direction_vector": [0.0, 1.0, 0.0],
                    "axis": "y",
                    "direction": "positive_y"
                }
        
        if opening_direction:
            spatial_context["opening_direction"] = opening_direction
        
        # 2. 分析相邻部件（从场景描述中提取，如果可能）
        # 这里简化处理，主要依赖场景描述
        if scene_description:
            # 尝试从场景描述中提取相邻部件信息
            # 这是一个简化的实现，实际可以更复杂
            spatial_context["part_position_relative_to_scene"] = part_analysis.detailed_position or part_analysis.position
        
        # 3. 如果有AABB信息，可以添加部件位置信息
        if aabb_info:
            spatial_context["part_aabb_center"] = aabb_info.get("center")
            spatial_context["part_aabb_size"] = aabb_info.get("size")
        
        return spatial_context
    
    def _store_visualization_to_memory(
        self,
        part_name: str,
        visualization_path: str,
        motion_type: str
    ) -> None:
        """将可视化图像路径保存到记忆中（不再使用task_id）"""
        try:
            content = f"""
运动轴可视化结果 - {part_name}：
- Part名称：{part_name}
- 运动类型：{motion_type}
- 可视化图像路径：{visualization_path}
"""
            metadata = {
                "part_name": part_name,
                "motion_type": motion_type,
                "visualization_path": visualization_path,
                "visualization_type": "axis_analysis",
                "image_name": f"visualization_{part_name}_{motion_type}",  # 用于检索
                "timestamp": time.time()
            }
            self.memory.store_long(content, metadata)
            print(f"✓ 可视化图像路径已保存到记忆: {visualization_path}")
        except Exception as e:
            print(f"保存可视化图像路径到记忆失败: {e}")
    
    def get_constraint_result(self, task_id: str, part_name: str) -> Optional[MotionConstraintResult]:
        """从记忆中获取已有的约束推理结果"""
        try:
            memories = self.memory.retrieve(f"运动约束推理结果 - {part_name}", memory_type="long", limit=100)
            
            for memory in memories:
                if memory.metadata.get("task_id") == task_id:
                    metadata = memory.metadata
                    return MotionConstraintResult(
                        part_name=part_name,
                        motion_type=metadata.get("motion_type", "unknown"),
                        sliding_direction=metadata.get("sliding_direction"),
                        sliding_orientation=metadata.get("sliding_orientation"),
                        rotation_type=metadata.get("rotation_type"),
                        axis_description=metadata.get("axis_description"),
                        axis_location=metadata.get("axis_location"),
                        confidence=metadata.get("confidence", 0.5),
                        reasoning=metadata.get("reasoning", ""),
                        timestamp=metadata.get("timestamp", time.time())
                    )
            
            return None
            
        except Exception as e:
            print(f"从记忆中获取约束推理结果失败: {e}")
            return None


__all__ = [
    "ConstraintReasoningAgent",
]

