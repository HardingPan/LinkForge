"""
MJCF约束生成智能体
接收前面智能体得到的运动约束信息，将其转换为MJCF格式并写入XML文件
"""

from __future__ import annotations

import time
import shutil
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .utils.data_models import MotionConstraintResult
from .utils.mjcf_constraint_models import (
    JointType,
    MJCFJointSpec,
    MJCFSiteSpec,
    MJCFEqualityConstraintSpec,
    MJCFBodySpec,
    MJCFConstraintPlan,
    MJCFGenerationResult
)
from .utils.mesh_analyzer import MeshAnalyzer


class MJCFConstraintAgent:
    """MJCF约束生成智能体
    
    功能：
    1. 接收MotionConstraintResult（来自ConstraintReasoningAgent）
    2. 将运动约束信息转换为MJCF标准格式
    3. 修改MJCF XML文件，添加joint、site、equality等约束
    4. 支持三种运动类型：滑动、转动（edge/centerline）、固定
    """
    
    def __init__(self):
        """初始化MJCF约束生成智能体"""
        pass
    
    def generate_constraints(
        self,
        xml_path: str,
        constraint_results: List[MotionConstraintResult],
        output_path: Optional[str] = None,
        create_backup: bool = True
    ) -> MJCFGenerationResult:
        """生成MJCF约束并写入XML文件
        
        Args:
            xml_path: 原始XML文件路径
            constraint_results: 运动约束结果列表（来自ConstraintReasoningAgent）
            output_path: 输出XML文件路径（如果为None，则覆盖原文件）
            create_backup: 是否创建备份
            
        Returns:
            MJCFGenerationResult包含生成结果和修改信息
        """
        start_time = time.time()
        
        try:
            # 验证输入
            if not Path(xml_path).exists():
                return MJCFGenerationResult(
                    success=False,
                    message=f"XML文件不存在: {xml_path}",
                    constraint_plans=[],
                    modifications=[]
                )
            
            if not constraint_results:
                return MJCFGenerationResult(
                    success=False,
                    message="没有提供约束结果",
                    constraint_plans=[],
                    modifications=[]
                )
            
            print(f"📝 开始生成MJCF约束，处理 {len(constraint_results)} 个部件")
            
            # 1. 将MotionConstraintResult转换为MJCFConstraintPlan
            constraint_plans = []
            for constraint in constraint_results:
                plan = self._convert_to_constraint_plan(constraint)
                if plan:
                    constraint_plans.append(plan)
                    print(f"  ✓ {constraint.part_name}: {constraint.motion_type} -> {plan.joint.type if plan.joint else 'fixed'}")
                else:
                    print(f"  ✗ {constraint.part_name}: 转换失败")
            
            if not constraint_plans:
                return MJCFGenerationResult(
                    success=False,
                    message="没有成功转换的约束方案",
                    constraint_plans=[],
                    modifications=[]
                )
            
            # 2. 加载并解析XML（确保从原始文件读取）
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 确保compiler元素存在并设置为使用度数
            compiler = root.find("compiler")
            if compiler is None:
                compiler = ET.SubElement(root, "compiler")
                print(f"  ✓ 创建compiler元素")
            compiler.set("angle", "degree")
            print(f"  ✓ 设置compiler angle='degree'（MuJoCo将自动处理角度转换）")
            
            print(f"  📄 已加载XML文件: {xml_path}")
            print(f"  📄 XML根元素: {root.tag}")
            
            # 验证XML加载：打印所有geom
            print(f"  🔍 验证XML加载，查找所有geom:")
            worldbody = root.find("worldbody")
            if worldbody:
                for body in worldbody.findall("body"):
                    body_name = body.get("name", "unnamed")
                    geoms = body.findall("geom")
                    print(f"    Body '{body_name}': 找到 {len(geoms)} 个geom")
                    for geom in geoms:
                        mesh_name = geom.get("mesh")
                        print(f"      - geom mesh='{mesh_name}'")
            
            print(f"  📊 开始处理 {len(constraint_plans)} 个约束方案")
            
            # 3. 为每个约束方案生成MJCF元素
            modifications = []
            for plan in constraint_plans:
                mods = self._apply_constraint_plan(root, plan, xml_path)
                modifications.extend(mods)
            
            # 4. 保存XML文件
            if output_path is None:
                output_path = xml_path
                if create_backup:
                    backup_path = str(Path(xml_path).with_suffix('.backup.xml'))
                    shutil.copy2(xml_path, backup_path)
            
            # 美化XML输出
            xml_str = self._prettify_xml(root)
            Path(output_path).write_text(xml_str, encoding='utf-8')
            
            print(f"✓ MJCF约束生成完成: {output_path} ({len(constraint_plans)}个约束方案)")
            
            return MJCFGenerationResult(
                success=True,
                message=f"成功生成 {len(constraint_plans)} 个约束方案",
                xml_path=output_path,
                constraint_plans=constraint_plans,
                modifications=modifications
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return MJCFGenerationResult(
                success=False,
                message=f"生成MJCF约束失败: {str(e)}",
                constraint_plans=[],
                modifications=[]
            )
    
    def _convert_to_constraint_plan(
        self,
        constraint: MotionConstraintResult
    ) -> Optional[MJCFConstraintPlan]:
        """将MotionConstraintResult转换为MJCFConstraintPlan
        
        Args:
            constraint: 运动约束结果
            
        Returns:
            MJCF约束方案，如果转换失败则返回None
        """
        try:
            part_name = constraint.part_name
            motion_type = constraint.motion_type
            
            # 固定部件：不需要joint
            if motion_type == "fixed":
                return MJCFConstraintPlan(
                    part_name=part_name,
                    motion_type="fixed",
                    rotation_type=None,
                    joint=None,
                    sites=[],
                    equality_constraints=[],
                    feature_frame=None,
                    confidence=constraint.confidence,
                    reasoning=f"固定部件，无需添加约束。{constraint.reasoning}"
                )
            
            # 滑动部件
            if motion_type == "sliding":
                return self._create_sliding_constraint_plan(constraint)
            
            # 旋转部件
            if motion_type == "rotating":
                return self._create_rotating_constraint_plan(constraint)
            
            return None
            
        except Exception as e:
            print(f"转换约束方案失败 ({constraint.part_name}): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _infer_sliding_direction_from_string(
        self,
        sliding_direction: Optional[str]
    ) -> Optional[List[float]]:
        """从滑动方向字符串推断方向向量
        
        Args:
            sliding_direction: 滑动方向字符串，如 "front_back", "left_right", "horizontal", "vertical" 等
                也支持中文描述，如 "往外拉", "往外开", "前后", "左右" 等
            
        Returns:
            方向向量 [x, y, z] 或 None
        """
        if not sliding_direction:
            return None
        
        sliding_direction_lower = sliding_direction.lower()
        
        # 映射滑动方向到方向向量（简化为三个轴）
        # MuJoCo坐标系统：X轴=左右，Y轴=前后，Z轴=上下
        direction_map = {
            # 三个轴方向
            "x": [1.0, 0.0, 0.0],  # X轴（左右）
            "y": [0.0, 1.0, 0.0],  # Y轴（前后）
            "z": [0.0, 0.0, 1.0],  # Z轴（上下）
            # 兼容旧格式（向后兼容）
            "front_back": [0.0, 1.0, 0.0],  # Y轴（前后）
            "left_right": [1.0, 0.0, 0.0],  # X轴（左右）
            "horizontal": [1.0, 0.0, 0.0],  # 默认水平方向为X轴
            "vertical": [0.0, 0.0, 1.0],    # Z轴（上下）
            # 中文方向（常见描述）
            "往外": [0.0, 1.0, 0.0],  # 往外拉/往外开，通常是+Y方向（前面）
            "往外拉": [0.0, 1.0, 0.0],  # 抽屉往外拉
            "往外开": [0.0, 1.0, 0.0],  # 门往外开
            "往里": [0.0, -1.0, 0.0],  # 往里推，通常是-Y方向（后面）
            "往里推": [0.0, -1.0, 0.0],
            "前后": [0.0, 1.0, 0.0],  # 前后方向，默认向前
            "后前": [0.0, -1.0, 0.0],  # 后前方向
            "左右": [1.0, 0.0, 0.0],  # 左右方向，默认向右
            "右左": [-1.0, 0.0, 0.0],  # 右左方向
            "上下": [0.0, 0.0, 1.0],  # 上下方向，默认向上
            "下上": [0.0, 0.0, -1.0],  # 下上方向
            "向前": [0.0, 1.0, 0.0],  # 向前
            "向后": [0.0, -1.0, 0.0],  # 向后
            "向左": [-1.0, 0.0, 0.0],  # 向左
            "向右": [1.0, 0.0, 0.0],  # 向右
            "向上": [0.0, 0.0, 1.0],  # 向上
            "向下": [0.0, 0.0, -1.0],  # 向下
        }
        
        # 检查是否匹配已知方向（优先匹配更具体的描述）
        # 先检查完整匹配，再检查部分匹配
        for key, vec in direction_map.items():
            if key in sliding_direction_lower:
                return vec
        
        # 如果没有匹配，尝试根据关键词推断
        # 对于"往外"、"往外拉"、"往外开"等，默认是+Y方向
        if "外" in sliding_direction and ("拉" in sliding_direction or "开" in sliding_direction):
            return [0.0, 1.0, 0.0]  # +Y方向（往外）
        elif "内" in sliding_direction and "推" in sliding_direction:
            return [0.0, -1.0, 0.0]  # -Y方向（往里）
        elif "前" in sliding_direction:
            return [0.0, 1.0, 0.0]  # +Y方向（前）
        elif "后" in sliding_direction:
            return [0.0, -1.0, 0.0]  # -Y方向（后）
        elif "左" in sliding_direction:
            return [-1.0, 0.0, 0.0]  # -X方向（左）
        elif "右" in sliding_direction:
            return [1.0, 0.0, 0.0]  # +X方向（右）
        elif "上" in sliding_direction:
            return [0.0, 0.0, 1.0]  # +Z方向（上）
        elif "下" in sliding_direction:
            return [0.0, 0.0, -1.0]  # -Z方向（下）
        
        # 如果没有匹配，返回None（需要其他信息）
        return None
    
    def _create_sliding_constraint_plan(
        self,
        constraint: MotionConstraintResult
    ) -> Optional[MJCFConstraintPlan]:
        """创建滑动约束方案"""
        try:
            part_name = constraint.part_name
            
            # 从selected_axis_info获取方向信息
            selected_axis_info = constraint.selected_axis_info
            direction = None
            
            if selected_axis_info:
                direction = selected_axis_info.get("direction")
            elif constraint.selected_axis:
                direction = constraint.selected_axis.get("direction")
            
            # 如果仍然没有方向，尝试从sliding_direction推断
            if not direction or len(direction) != 3:
                direction = self._infer_sliding_direction_from_string(constraint.sliding_direction)
                if not direction:
                    print(f"  ⚠ {part_name}: 无法从selected_axis_info、selected_axis或sliding_direction获取方向信息")
                    print(f"     - selected_axis_info: {constraint.selected_axis_info is not None}")
                    print(f"     - selected_axis: {constraint.selected_axis is not None}")
                    print(f"     - sliding_direction: {constraint.sliding_direction}")
                    return None
            
            # 归一化方向向量
            import numpy as np
            direction_array = np.array(direction, dtype=float)
            norm = np.linalg.norm(direction_array)
            if norm < 1e-6:
                print(f"  ⚠ {part_name}: 方向向量为零向量")
                return None
            direction_normalized = (direction_array / norm).tolist()
            
            # 打印滑动方向信息（用于调试）
            axis_names = ["X", "Y", "Z"]
            dominant_axis_idx = np.argmax(np.abs(direction_normalized))
            dominant_axis_name = axis_names[dominant_axis_idx]
            direction_sign = "正" if direction_normalized[dominant_axis_idx] > 0 else "负"
            print(f"  ✓ {part_name}: 滑动方向 = {direction_normalized}, 主导轴: {direction_sign}{dominant_axis_name}轴")
            if selected_axis_info:
                axis_id = selected_axis_info.get("axis_id", "unknown")
                reference_direction_id = selected_axis_info.get("reference_direction_id")
                direction_vec = selected_axis_info.get("direction", [])
                print(f"    选择的轴ID: {axis_id}")
                if reference_direction_id:
                    # 显示原始方向ID，让用户更清楚
                    direction_name_map = {
                        "positive_x": "+X方向（向右）",
                        "negative_x": "-X方向（向左）",
                        "positive_y": "+Y方向（向前/向上）",
                        "negative_y": "-Y方向（向后/向下）",
                        "positive_z": "+Z方向（向前/向上）",
                        "negative_z": "-Z方向（向后/向下）"
                    }
                    direction_name = direction_name_map.get(reference_direction_id, reference_direction_id)
                    print(f"    原始方向ID: {reference_direction_id} ({direction_name})")
                if direction_vec:
                    print(f"    方向向量: [{direction_vec[0]:.3f}, {direction_vec[1]:.3f}, {direction_vec[2]:.3f}]")
            
            # 确定滑动范围
            # 优先使用LLM推理的范围，否则使用默认值（对称范围）
            slide_range = (-0.4, 0.4)  # 默认范围：±0.4米
            if constraint.motion_range:
                slide_range = tuple(constraint.motion_range)
            # 使用默认范围或LLM推理的范围（不打印）
            
            # 创建滑动关节
            joint = MJCFJointSpec(
                name=f"{part_name}_slide",
                type=JointType.SLIDE,
                body_name=part_name,
                axis=tuple(direction_normalized),
                pos=None,  # 滑动关节通常不需要指定位置
                limited=True,
                range=slide_range,
                damping=1.0,  # 增加阻尼以提高稳定性
                stiffness=0.0
            )
            
            # 不创建可视化站点（site仅用于调试，不应出现在最终结果中）
            sites = []
            
            # 构建reasoning，避免重复
            if constraint.axis_selection_reasoning:
                # 如果axis_selection_reasoning存在，使用它（它已经包含了完整的推理过程）
                reasoning = f"滑动约束。{constraint.axis_selection_reasoning}"
            else:
                # 否则使用原始的reasoning
                reasoning = f"滑动约束。{constraint.reasoning}"
            
            return MJCFConstraintPlan(
                part_name=part_name,
                motion_type="sliding",
                rotation_type=None,
                joint=joint,
                sites=sites,
                equality_constraints=[],
                feature_frame=None,
                confidence=constraint.axis_selection_confidence or constraint.confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"创建滑动约束方案失败 ({constraint.part_name}): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _infer_rotation_axis_from_description(
        self,
        rotation_type: Optional[str],
        axis_description: Optional[str],
        axis_location: Optional[str]
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """从轴描述推断旋转轴信息
        
        Args:
            rotation_type: 旋转类型 "edge" 或 "centerline"
            axis_description: 轴描述，如 "vertical centerline", "horizontal edge at top"
            axis_location: 轴位置描述
            
        Returns:
            (direction, position) 方向向量和位置，如果无法推断则返回 (None, None)
        """
        if not axis_description:
            return None, None
        
        axis_desc_lower = axis_description.lower()
        direction = None
        
        # 推断方向向量
        if "vertical" in axis_desc_lower or "z" in axis_desc_lower:
            direction = [0.0, 0.0, 1.0]  # Z轴（垂直）
        elif "horizontal" in axis_desc_lower:
            if "x" in axis_desc_lower or "left" in axis_desc_lower or "right" in axis_desc_lower:
                direction = [1.0, 0.0, 0.0]  # X轴（水平左右）
            elif "y" in axis_desc_lower or "front" in axis_desc_lower or "back" in axis_desc_lower:
                direction = [0.0, 1.0, 0.0]  # Y轴（水平前后）
            else:
                direction = [1.0, 0.0, 0.0]  # 默认X轴
        elif "x" in axis_desc_lower:
            direction = [1.0, 0.0, 0.0]
        elif "y" in axis_desc_lower:
            direction = [0.0, 1.0, 0.0]
        
        # 对于edge旋转，位置通常在边缘；对于centerline旋转，位置在中心
        # 这里我们无法从描述中准确推断位置，返回None让调用者使用默认值
        position = None
        
        return direction, position
    
    def _create_rotating_constraint_plan(
        self,
        constraint: MotionConstraintResult
    ) -> Optional[MJCFConstraintPlan]:
        """创建旋转约束方案"""
        try:
            part_name = constraint.part_name
            rotation_type = constraint.rotation_type  # edge 或 centerline
            
            # 从selected_axis_info获取轴信息
            selected_axis_info = constraint.selected_axis_info
            midpoint = None
            point = None
            direction = None
            length = 1.0
            
            if selected_axis_info:
                # 从selected_axis_info提取信息
                print(f"  📋 {part_name}: selected_axis_info包含的键: {list(selected_axis_info.keys())}")
                if "midpoint" in selected_axis_info:
                    # Edge旋转
                    midpoint = selected_axis_info.get("midpoint")
                    direction = selected_axis_info.get("direction")
                    length = selected_axis_info.get("length", 1.0)
                    print(f"  ✓ {part_name}: 从selected_axis_info获取edge信息: midpoint={midpoint}, direction={direction}, length={length}")
                elif "point" in selected_axis_info:
                    # 中心线旋转
                    point = selected_axis_info.get("point")
                    direction = selected_axis_info.get("direction")
                    length = 1.0
                    print(f"  ✓ {part_name}: 从selected_axis_info获取中心线信息: point={point}, direction={direction}")
            elif constraint.selected_axis:
                # 尝试从selected_axis获取
                selected_axis = constraint.selected_axis
                print(f"  📋 {part_name}: selected_axis包含的键: {list(selected_axis.keys())}")
                if "midpoint" in selected_axis:
                    # Edge旋转
                    midpoint = selected_axis.get("midpoint")
                    direction = selected_axis.get("direction")
                    length = selected_axis.get("length", 1.0)
                    print(f"  ✓ {part_name}: 从selected_axis获取edge信息: midpoint={midpoint}, direction={direction}, length={length}")
                elif "point" in selected_axis:
                    # 中心线旋转
                    point = selected_axis.get("point")
                    direction = selected_axis.get("direction")
                    length = 1.0
                    print(f"  ✓ {part_name}: 从selected_axis获取中心线信息: point={point}, direction={direction}")
            
            # 如果仍然没有方向，尝试从axis_description推断
            if not direction or len(direction) != 3:
                inferred_direction, inferred_position = self._infer_rotation_axis_from_description(
                    rotation_type,
                    constraint.axis_description,
                    constraint.axis_location
                )
                if inferred_direction:
                    direction = inferred_direction
                    # 如果推断出了位置，使用它
                    if inferred_position:
                        if rotation_type == "edge":
                            midpoint = inferred_position
                        else:
                            point = inferred_position
                else:
                    print(f"  ⚠ {part_name}: 无法从selected_axis_info、selected_axis或axis_description获取方向信息")
                    print(f"     - selected_axis_info: {constraint.selected_axis_info is not None}")
                    print(f"     - selected_axis: {constraint.selected_axis is not None}")
                    print(f"     - rotation_type: {rotation_type}")
                    print(f"     - axis_description: {constraint.axis_description}")
                    print(f"     - axis_location: {constraint.axis_location}")
                    return None
            
            if not direction or len(direction) != 3:
                return None
            
            # 归一化方向向量
            import numpy as np
            direction_array = np.array(direction, dtype=float)
            norm = np.linalg.norm(direction_array)
            if norm < 1e-6:
                return None
            direction_normalized = (direction_array / norm).tolist()
            
            # 确定关节位置
            # 如果没有明确的位置信息，使用默认值（0, 0, 0）或从geom的包围盒推断
            joint_pos = None
            if midpoint:
                joint_pos = tuple(midpoint)
                print(f"  ✓ {part_name}: 使用edge中点作为关节位置: {joint_pos}")
            elif point:
                joint_pos = tuple(point)
                print(f"  ✓ {part_name}: 使用中心点作为关节位置: {joint_pos}")
            else:
                print(f"  ⚠ {part_name}: 未找到位置信息，joint_pos=None")
                # 如果没有位置信息，joint_pos保持为None，MuJoCo会使用body的默认位置
            
            # 确定旋转范围
            # 由于MuJoCo的compiler已设置为angle="degree"，我们可以直接使用度数
            # LLM输出的是度数，直接使用（MuJoCo会自动转换为弧度）
            # 默认范围：±90度（对称范围）
            default_range_deg = (-90.0, 90.0)
            rotation_range_deg = default_range_deg
            
            if constraint.motion_range:
                raw_range = tuple(constraint.motion_range)
                max_angle_raw = raw_range[1]
                
                # 智能检测：判断LLM输出的是度数还是弧度
                # 如果值 < π (3.14)，可能是弧度，需要转换为度数
                # 如果值 >= π，很可能是度数
                if max_angle_raw < math.pi:  # < 3.14，可能是弧度
                    # 转换为度数
                    rotation_range_deg = (math.degrees(raw_range[0]), math.degrees(raw_range[1]))
                else:
                    # 直接使用度数
                    rotation_range_deg = raw_range
                
                # 验证：对于对称范围，计算总范围（max - min）
                total_range = abs(rotation_range_deg[1] - rotation_range_deg[0])
                
                # 验证：如果范围异常小，给出警告
                if total_range < 60.0:
                    print(f"  ⚠ {part_name}: 旋转范围过小 ({total_range:.1f}度)")
            # 使用默认范围或LLM推理的范围（不打印）
            
            # 直接使用度数范围（MuJoCo会自动转换）
            rotation_range = rotation_range_deg
            
            # 创建旋转关节
            joint = MJCFJointSpec(
                name=f"{part_name}_hinge",
                type=JointType.HINGE,
                body_name=part_name,
                axis=tuple(direction_normalized),
                pos=joint_pos,
                limited=True,
                range=rotation_range,
                damping=2.0,  # 增加阻尼以提高稳定性
                stiffness=0.0
            )
            
            # 不创建可视化站点（site仅用于调试，不应出现在最终结果中）
            sites = []
            
            # 构建reasoning，避免重复
            if constraint.axis_selection_reasoning:
                # 如果axis_selection_reasoning存在，使用它（它已经包含了完整的推理过程）
                reasoning = f"旋转约束（{rotation_type}）。{constraint.axis_selection_reasoning}"
            else:
                # 否则使用原始的reasoning
                reasoning = f"旋转约束（{rotation_type}）。{constraint.reasoning}"
            
            return MJCFConstraintPlan(
                part_name=part_name,
                motion_type="rotating",
                rotation_type=rotation_type,
                joint=joint,
                sites=sites,
                equality_constraints=[],
                feature_frame=None,
                confidence=constraint.axis_selection_confidence or constraint.confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"创建旋转约束方案失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_constraint_plan(
        self,
        root: ET.Element,
        plan: MJCFConstraintPlan,
        xml_path: str
    ) -> List[str]:
        """将约束方案应用到XML树
        
        Args:
            root: XML根元素
            plan: 约束方案
            xml_path: XML文件路径（用于查找mesh信息）
            
        Returns:
            修改说明列表
        """
        modifications = []
        
        try:
            # 1. 找到包含该part geom的body和geom元素
            print(f"    🔍 开始查找part '{plan.part_name}' 的geom...")
            part_geom, parent_body = self._find_part_geom(root, plan.part_name)
            print(f"    📊 查找结果: part_geom={part_geom is not None}, parent_body={parent_body is not None}")
            if part_geom is not None:
                print(f"    📊 part_geom.tag={part_geom.tag}, mesh={part_geom.get('mesh')}")
            if parent_body is not None:
                print(f"    📊 parent_body.tag={parent_body.tag}, name={parent_body.get('name', 'unnamed')}")
            
            # 使用is None检查，而不是not检查（Element对象可能有特殊的布尔值）
            if part_geom is None or parent_body is None:
                print(f"    ❌ 检查失败: part_geom is None={part_geom is None}, parent_body is None={parent_body is None}")
                modifications.append(f"⚠ 未找到part {plan.part_name} 对应的geom，跳过")
                return modifications
            
            print(f"    ✓ 成功找到geom和body，继续处理...")
            
            # 2. 如果是固定部件，不需要添加joint，保持原样
            if plan.motion_type == "fixed":
                modifications.append(f"✓ {plan.part_name}: 固定部件，无需添加约束")
                return modifications
            
            # 3. 为part创建独立的body（如果还没有）
            part_body = self._ensure_part_body(root, plan.part_name, part_geom, parent_body)
            if part_body != parent_body:
                modifications.append(f"✓ 为part {plan.part_name} 创建独立body")
            
            # 4. 创建feature_frame（如果需要）
            if plan.feature_frame:
                feature_frame_body = self._create_feature_frame(root, part_body, plan.feature_frame)
                modifications.append(f"✓ 创建feature_frame: {plan.feature_frame.name}")
            else:
                feature_frame_body = part_body
            
            # 5. 添加joint（joint定义在body中，控制该body相对于父body的运动）
            if plan.joint:
                # 如果joint有pos，且body有mesh_center，需要调整joint的pos
                # joint的pos是相对于body的，如果body的pos是mesh_center，joint的pos需要减去mesh_center
                joint_spec = plan.joint
                if joint_spec.pos and part_body.get("_mesh_center"):
                    try:
                        mesh_center_str = part_body.get("_mesh_center")
                        mesh_center = [float(x) for x in mesh_center_str.split()]
                        if len(mesh_center) == 3:
                            # 调整joint的pos：从world坐标转换为body相对坐标
                            adjusted_pos = (
                                joint_spec.pos[0] - mesh_center[0],
                                joint_spec.pos[1] - mesh_center[1],
                                joint_spec.pos[2] - mesh_center[2]
                            )
                            # 创建调整后的joint spec
                            adjusted_joint_spec = MJCFJointSpec(
                                name=joint_spec.name,
                                type=joint_spec.type,
                                body_name=joint_spec.body_name,
                                axis=joint_spec.axis,
                                pos=adjusted_pos,
                                limited=joint_spec.limited,
                                range=joint_spec.range,
                                damping=joint_spec.damping,
                                stiffness=joint_spec.stiffness,
                                armature=joint_spec.armature,
                                parent_body=joint_spec.parent_body
                            )
                            joint_elem = self._create_joint_element(adjusted_joint_spec)
                        else:
                            joint_elem = self._create_joint_element(joint_spec)
                    except Exception as e:
                        print(f"    ⚠ 调整joint位置失败: {e}，使用原始位置")
                        joint_elem = self._create_joint_element(joint_spec)
                else:
                    joint_elem = self._create_joint_element(joint_spec)
                
                # joint必须添加到body中，不能添加到worldbody
                part_body.append(joint_elem)
                modifications.append(f"✓ 添加joint: {plan.joint.name} (type={plan.joint.type.value})")
                print(f"    ✓ Joint已添加到body: {part_body.get('name')}")
                
                # 清理临时属性
                if part_body.get("_mesh_center"):
                    part_body.attrib.pop("_mesh_center", None)
            
            # 6. 添加sites
            for site in plan.sites:
                site_elem = self._create_site_element(site)
                feature_frame_body.append(site_elem)
                modifications.append(f"✓ 添加site: {site.name}")
            
            # 7. 添加equality约束（如果需要）
            if plan.equality_constraints:
                equality_elem = self._find_or_create_equality(root)
                for eq_constraint in plan.equality_constraints:
                    eq_elem = self._create_equality_element(eq_constraint)
                    equality_elem.append(eq_elem)
                    modifications.append(f"✓ 添加equality约束: {eq_constraint.name}")
            
            # 8. 添加actuator以便控制joint（如果需要）
            if plan.joint:
                actuator_elem = self._find_or_create_actuator(root)
                position_actuator = ET.Element("position")
                position_actuator.set("name", f"{plan.joint.name}_actuator")
                position_actuator.set("joint", plan.joint.name)
                position_actuator.set("kp", "100")  # 位置增益
                position_actuator.set("kv", "10")   # 速度增益
                actuator_elem.append(position_actuator)
                modifications.append(f"✓ 添加actuator: {plan.joint.name}_actuator")
            
        except Exception as e:
            modifications.append(f"✗ 应用约束方案失败 ({plan.part_name}): {str(e)}")
            import traceback
            traceback.print_exc()
        
        return modifications
    
    def _find_part_geom(self, root: ET.Element, part_name: str) -> Tuple[Optional[ET.Element], Optional[ET.Element]]:
        """查找part对应的geom元素和其父body
        
        通过匹配geom的mesh名称来找到对应的geom和body
        Returns:
            (geom元素, 父body元素) 或 (None, None)
        """
        print(f"    🔍 查找part '{part_name}' 对应的geom...")
        
        # 先查找worldbody下的直接body（最常见的情况）
        worldbody = root.find("worldbody")
        if worldbody is not None:
            print(f"    ✓ 找到worldbody，开始查找...")
            for body in worldbody.findall("body"):
                body_name = body.get("name", "unnamed")
                print(f"      检查body: {body_name}")
                for geom in body.findall("geom"):
                    mesh_name = geom.get("mesh")
                    print(f"        找到geom，mesh='{mesh_name}'")
                    if mesh_name == part_name:
                        print(f"    ✓ 找到匹配的geom: mesh='{mesh_name}' == part_name='{part_name}'")
                        return geom, body
        
        # 如果没找到，查找所有body（包括嵌套的）
        print(f"    ⚠ 在worldbody下未找到，查找所有body...")
        for body in root.findall(".//body"):
            for geom in body.findall("geom"):
                mesh_name = geom.get("mesh")
                if mesh_name == part_name:
                    print(f"    ✓ 找到匹配的geom: mesh='{mesh_name}' == part_name='{part_name}'")
                    return geom, body
        
        # 调试：打印所有geom的mesh名称
        print(f"    ❌ 未找到part '{part_name}'，当前XML中的所有geom:")
        all_geoms_found = []
        for body in root.findall(".//body"):
            body_name = body.get("name", "unnamed")
            for geom in body.findall("geom"):
                mesh_name = geom.get("mesh")
                all_geoms_found.append((body_name, mesh_name))
                print(f"      在body '{body_name}'中找到geom，mesh='{mesh_name}'")
        
        if not all_geoms_found:
            print(f"    ⚠ 警告：XML中没有任何geom元素！")
        
        return None, None
    
    def _find_part_body(self, root: ET.Element, part_name: str) -> Optional[ET.Element]:
        """查找part对应的body元素（向后兼容）
        
        通过匹配geom的mesh名称来找到对应的body
        """
        _, body = self._find_part_geom(root, part_name)
        return body
    
    def _ensure_part_body(
        self,
        root: ET.Element,
        part_name: str,
        part_geom: ET.Element,
        parent_body: ET.Element
    ) -> ET.Element:
        """确保part有独立的body
        
        如果geom已经在独立的body中（body只包含这个geom），则返回该body
        否则，创建一个新的body，将geom移动到新body，并返回新body
        
        Args:
            root: XML根元素
            part_name: part名称
            part_geom: part的geom元素
            parent_body: 当前包含geom的body
            
        Returns:
            part的独立body元素
        """
        # 检查当前body是否只包含这个geom（即已经是独立的body）
        geoms_in_body = parent_body.findall("geom")
        if len(geoms_in_body) == 1 and geoms_in_body[0] == part_geom:
            # 已经是独立的body，直接返回
            # 但需要确保body名称正确
            if not parent_body.get("name") or parent_body.get("name") != f"{part_name}_body":
                parent_body.set("name", f"{part_name}_body")
            return parent_body
        
        # 需要创建新的独立body
        # 找到worldbody
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        
        # 创建新的body（作为worldbody的直接子元素，这样joint才能控制它相对于world的运动）
        new_body = ET.SubElement(worldbody, "body")
        new_body.set("name", f"{part_name}_body")
        
        # 重要：在MuJoCo中，mesh的顶点位置是相对于mesh的局部坐标系的
        # 当我们创建新body时，需要计算mesh的中心位置，设置body的pos
        # 这样geom可以保持在正确的位置
        # joint的pos是相对于body的，所以joint的位置也需要相应调整
        
        mesh_name = part_geom.get("mesh")
        mesh_center = None
        
        if mesh_name:
            try:
                # 获取XML路径
                xml_path = self._get_xml_path_from_root(root)
                if xml_path and Path(xml_path).exists():
                    # 分析mesh获取中心位置
                    mesh_analyzer = MeshAnalyzer(xml_path)
                    mesh_info_dict = mesh_analyzer.analyze()
                    mesh_info = mesh_info_dict.get(mesh_name)
                    
                    if mesh_info:
                        # 使用mesh的AABB中心作为body的位置
                        mesh_center = mesh_info.aabb.center
                        new_body.set("pos", f"{mesh_center[0]} {mesh_center[1]} {mesh_center[2]}")
                        print(f"    📍 设置body位置为mesh '{mesh_name}' 中心: ({mesh_center[0]:.4f}, {mesh_center[1]:.4f}, {mesh_center[2]:.4f})")
            except Exception as e:
                print(f"    ⚠ 无法计算mesh中心位置: {e}，body位置保持默认(0,0,0)")
        
        # 复制geom到新body（保留所有属性）
        new_geom = ET.SubElement(new_body, "geom")
        for key, value in part_geom.attrib.items():
            new_geom.set(key, value)
        
        # 保存mesh_center供后续joint使用
        new_body.set("_mesh_center", f"{mesh_center[0]} {mesh_center[1]} {mesh_center[2]}" if mesh_center else "0 0 0")
        
        # 从原body中移除geom
        parent_body.remove(part_geom)
        
        print(f"    ✓ 创建了独立body: {part_name}_body，geom已移动")
        
        return new_body
    
    def _get_xml_path_from_root(self, root: ET.Element) -> Optional[str]:
        """从XML根元素获取文件路径（如果可能）"""
        # 使用保存的xml_path
        return getattr(self, '_current_xml_path', None)
    
    def _create_feature_frame(
        self,
        root: ET.Element,
        parent_body: ET.Element,
        feature_frame_spec: MJCFBodySpec
    ) -> ET.Element:
        """创建feature_frame body"""
        feature_frame = ET.SubElement(parent_body, "body")
        feature_frame.set("name", feature_frame_spec.name)
        feature_frame.set("pos", f"{feature_frame_spec.pos[0]} {feature_frame_spec.pos[1]} {feature_frame_spec.pos[2]}")
        
        if feature_frame_spec.quat:
            feature_frame.set("quat", f"{feature_frame_spec.quat[0]} {feature_frame_spec.quat[1]} {feature_frame_spec.quat[2]} {feature_frame_spec.quat[3]}")
        
        return feature_frame
    
    def _create_joint_element(self, joint_spec: MJCFJointSpec) -> ET.Element:
        """创建joint XML元素"""
        joint = ET.Element("joint")
        joint.set("name", joint_spec.name)
        joint.set("type", joint_spec.type.value)
        
        if joint_spec.pos:
            joint.set("pos", f"{joint_spec.pos[0]} {joint_spec.pos[1]} {joint_spec.pos[2]}")
        
        joint.set("axis", f"{joint_spec.axis[0]} {joint_spec.axis[1]} {joint_spec.axis[2]}")
        
        if joint_spec.limited:
            joint.set("limited", "true")
            if joint_spec.range:
                joint.set("range", f"{joint_spec.range[0]} {joint_spec.range[1]}")
        
        if joint_spec.damping is not None:
            joint.set("damping", str(joint_spec.damping))
        
        if joint_spec.stiffness is not None:
            joint.set("stiffness", str(joint_spec.stiffness))
        
        if joint_spec.armature is not None:
            joint.set("armature", str(joint_spec.armature))
        
        return joint
    
    def _create_site_element(self, site_spec: MJCFSiteSpec) -> ET.Element:
        """创建site XML元素"""
        site = ET.Element("site")
        site.set("name", site_spec.name)
        site.set("size", str(site_spec.size))
        site.set("type", site_spec.type)
        
        # MuJoCo不允许同时设置pos和fromto，如果设置了fromto就不设置pos
        if site_spec.fromto:
            site.set("fromto", f"{site_spec.fromto[0]} {site_spec.fromto[1]} {site_spec.fromto[2]} {site_spec.fromto[3]} {site_spec.fromto[4]} {site_spec.fromto[5]}")
        else:
            site.set("pos", f"{site_spec.pos[0]} {site_spec.pos[1]} {site_spec.pos[2]}")
        
        return site
    
    def _find_or_create_equality(self, root: ET.Element) -> ET.Element:
        """查找或创建equality元素"""
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        return equality
    
    def _find_or_create_actuator(self, root: ET.Element) -> ET.Element:
        """查找或创建actuator元素"""
        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        return actuator
    
    def _create_equality_element(self, eq_spec: MJCFEqualityConstraintSpec) -> ET.Element:
        """创建equality约束XML元素"""
        if eq_spec.type.value == "connect":
            eq_elem = ET.Element("connect")
            eq_elem.set("name", eq_spec.name)
            eq_elem.set("site1", eq_spec.site1)
            eq_elem.set("site2", eq_spec.site2)
            
            if eq_spec.solref:
                eq_elem.set("solref", f"{eq_spec.solref[0]} {eq_spec.solref[1]}")
            
            if eq_spec.solimp:
                eq_elem.set("solimp", f"{eq_spec.solimp[0]} {eq_spec.solimp[1]} {eq_spec.solimp[2]} {eq_spec.solimp[3]} {eq_spec.solimp[4]}")
            
            return eq_elem
        else:
            # 其他类型的equality约束
            raise NotImplementedError(f"不支持的equality约束类型: {eq_spec.type}")
    
    def _prettify_xml(self, root: ET.Element) -> str:
        """美化XML输出"""
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')


__all__ = [
    "MJCFConstraintAgent",
]

