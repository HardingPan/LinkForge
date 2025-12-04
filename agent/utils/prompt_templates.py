"""
Prompt 模板集合
将所有 LLM 分析相关的 prompt 封装成函数，便于统一管理和维护
"""

from typing import Dict, Any, Optional, List
from langchain.output_parsers import PydanticOutputParser

from .data_models import (
    PartAnalysisLLMResponse,
    PartMotionTypeResponse,
    MotionConstraintLLMResponse,
    AxisSelectionLLMResponse,
    UserHintParsedResult
)


def build_scene_analysis_prompt(task_instruction: str = "") -> str:
    """构建场景分析的 prompt
    
    Args:
        task_instruction: 任务指令，用于指导分析重点
        
    Returns:
        场景分析的 prompt 文本
    """
    prompt = f"""
请对这个3D场景进行全面分析：

1. 场景识别：
   - 这是什么类型的设备、产品或环境？
   - 整体外观特征和风格是什么？

2. 主要组件分析：
   - 识别场景中的主要部件和组件
   - 描述各部件的外观特征（形状、大小、颜色、材质等）
   - 分析部件的功能和作用

3. 空间关系分析：
   - 分析部件间的空间位置关系
   - 识别可能的连接方式和装配结构
   - 分析整体布局的合理性

4. 运动部件识别：
   - 识别可能的运动部件（门、盖、抽屉、旋钮、开关等）
   - 识别固定部件（主体、框架、底座、外壳等）
   - 分析运动部件的运动方式和范围

5. 功能分析：
   - 推测这个设备或产品的主要功能
   - 分析各部件如何协同工作
   - 识别可能的操作方式和用户交互点

6. 任务相关分析：
   {f"根据任务指令'{task_instruction}'重点关注相关内容" if task_instruction else "进行通用场景分析"}

请以结构化的方式详细描述你的观察结果，特别关注：
- 设备的整体类型和用途
- 主要部件的功能和特征
- 部件间的空间和功能关系
- 可能的运动机制和操作方式
"""
    return prompt


def build_part_analysis_prompt(
    part_name: str,
    scene_info: Dict[str, Any],
    scene_description: Optional[str] = None,
    color_mapping: Dict[str, str] = None
) -> str:
    """构建 part 详细分析的 prompt
    
    Args:
        part_name: part 名称
        scene_info: 场景信息字典，包含 device_type, total_components, complexity_level 等
        scene_description: 场景描述文本（作为上下文）
        color_mapping: 颜色-部件映射字典，格式为 {part_name: hex_color}
        
    Returns:
        part 分析的 prompt 文本（包含格式说明）
    """
    if color_mapping is None:
        color_mapping = {}
    
    # 构建场景描述上下文
    scene_context_text = ""
    if scene_description:
        scene_context_text = f"""
Scene Description (Context):
{scene_description}

"""
    
    # 构建基础 prompt
    base_prompt = [
        f"Analyze the part '{part_name}' based on the overall scene image and highlighted render:",
        "",
    ]
    
    # 添加场景描述上下文
    if scene_context_text:
        base_prompt.extend([
            scene_context_text.strip(),
            "",
        ])
    
    base_prompt.extend([
        "Scene Context:",
        f"- Device Type: {scene_info.get('device_type', 'unknown')}",
        f"- Component Count: {scene_info.get('total_components', 0)}",
        f"- Complexity: {scene_info.get('complexity_level', 'unknown')}",
        "",
        "Image Description:",
        "- The first image is the overall scene image showing the complete device structure;",
        f"- The second image is the part highlighted render, where the highlighted part (bright colors) corresponds to part '{part_name}';",
        "",
        "Color-Part Mapping (highlighted parts in the render):",
    ])
    
    # 添加颜色映射
    for name, hex_color in color_mapping.items():
        base_prompt.append(f"- {name}: {hex_color}")
    
    base_prompt.extend([
        "",
        "Please provide the following detailed information in JSON format:",
        "1) **Specific Function**: The specific role of this part in the overall device;",
        "2) **Detailed Position**: The precise position of the part in the device (e.g., top-left, center, bottom, etc.);",
        "3) **Motion Method**: If movable, how it moves (rotation axis, sliding direction, motion range);",
        "4) **Interaction Method**: How users interact with this part (push-pull, rotate, press, touch, etc.);",
        "5) **Relative to Ground**: The position relationship of this part relative to the ground (direct contact, suspended, supported, etc.);",
        "6) **Connection Type**: How it connects to other parts (fixed, hinged, sliding, etc.);",
        "7) **Importance**: Importance in the overall structure (core, auxiliary, decorative, etc.);",
        "8) **Shape Features**: The specific shape and size characteristics of the part;",
        "9) **Material Inference**: Possible materials and manufacturing processes;",
        "",
        "IMPORTANT: You MUST respond with valid JSON format only. Do not include any markdown formatting, explanations, or additional text. Only return the JSON object that matches the required schema.",
    ])
    
    # 创建 PydanticOutputParser 并添加格式说明
    parser = PydanticOutputParser(pydantic_object=PartAnalysisLLMResponse)
    base_prompt.append("")
    base_prompt.append(parser.get_format_instructions())
    
    return "\n".join(base_prompt)


def build_part_motion_type_prompt(
    part_name: str,
    scene_info: Dict[str, Any],
    scene_description: Optional[str] = None
) -> str:
    """构建 part 运动类型分析的 prompt（仅分析运动类型）
    
    Args:
        part_name: part 名称
        scene_info: 场景信息字典，包含 device_type, total_components, complexity_level 等
        scene_description: 场景描述文本（作为上下文）
        
    Returns:
        part 运动类型分析的 prompt 文本（包含格式说明）
    """
    # 构建场景描述上下文
    scene_context_text = ""
    if scene_description:
        scene_context_text = f"""
Scene Description (Context):
{scene_description}

"""
    
    # 创建 PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=PartMotionTypeResponse)
    
    # 构建 prompt
    prompt = f"""
Analyze the motion type of the highlighted part "{part_name}" based on the scene context:

{scene_context_text}Scene Context:
- Device Type: {scene_info.get('device_type', 'unknown')}
- Component Count: {scene_info.get('total_components', 0)}
- Complexity: {scene_info.get('complexity_level', 'unknown')}

The first image is the overall scene image, the second image is the highlighted render of part '{part_name}' (highlighted parts are marked with bright colors).

Please determine the motion type of this part:
- "fixed": The part is stationary and does not move
- "sliding": The part moves in a linear sliding motion
- "rotating": The part rotates around an axis

Provide only the motion type and brief reasoning.

{parser.get_format_instructions()}
"""
    return prompt


def build_sliding_constraint_prompt(
    part_name: str,
    part_analysis: Any,  # PartAnalysisResult
    scene_description: Optional[str] = None,
    aabb_info: Optional[Dict[str, Any]] = None  # AABB信息：size, center等
) -> str:
    """构建滑动部件约束推理的 prompt
    
    Args:
        part_name: part 名称
        part_analysis: part 分析结果对象（PartAnalysisResult）
        scene_description: 场景描述文本（作为上下文）
        aabb_info: AABB信息字典，包含size, center等
        
    Returns:
        滑动约束推理的 prompt 文本（包含格式说明）
    """
    # 构建场景描述上下文
    scene_context = ""
    if scene_description:
        scene_context = f"""
Scene Description (Context):
{scene_description}

"""
    
    # 构建AABB信息上下文
    aabb_context = ""
    if aabb_info:
        size = aabb_info.get("size", [0, 0, 0])
        center = aabb_info.get("center", [0, 0, 0])
        aabb_context = f"""
Part Geometry (AABB - Axis-Aligned Bounding Box):
- Size (width, depth, height): [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}] meters
- Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]
- Largest dimension: {max(size):.4f} meters
- Smallest dimension: {min(size):.4f} meters

Use this geometry information to infer the sliding range. Typically, drawers slide a distance approximately equal to their depth or width (usually 0.3-0.5m for standard drawers).

"""
    
    # 创建 PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=MotionConstraintLLMResponse)
    
    # 构建 prompt
    prompt = f"""
Analyze the sliding constraint of part "{part_name}" based on the scene context and previous analysis:

{scene_context}{aabb_context}Part Analysis Summary:
- Function: {part_analysis.function}
- Motion Type: {part_analysis.motion_type}
- Position: {part_analysis.position}
- Motion Description: {part_analysis.motion_description}
- Motion Axis: {part_analysis.motion_axis or 'unknown'}
- Connection Type: {part_analysis.connection_type}
- Detailed Position: {part_analysis.detailed_position}

The first image is the overall scene image, the second image is the highlighted render of part '{part_name}' (highlighted parts are marked with bright colors).

Please determine:
1. **Sliding Direction**: 
   - "x": The part slides along X-axis (left-right direction)
   - "y": The part slides along Y-axis (front-back direction)
   - "z": The part slides along Z-axis (up-down direction)

2. **Motion Range** (motion_range): 
   - Based on the AABB size and part function, estimate the maximum sliding distance in meters
   - For drawers: typically 0.3-0.5m (based on depth/width)
   - Consider physical constraints: The part cannot slide beyond its own size
   - **IMPORTANT: Provide a single positive value (e.g., 0.4). The system will automatically convert it to symmetric range [-0.4, 0.4]**
   - The value represents the maximum distance in one direction, allowing sliding in both positive and negative directions
   - Example: 0.4 means slides ±0.4 meters (total range 0.8m)
   - Example: 0.3 means slides ±0.3 meters (total range 0.6m)

Provide:
1. sliding_direction: The sliding direction type (x, y, or z)
2. sliding_orientation: Detailed description of the sliding orientation (e.g., "slides along X-axis (left-right)", "slides along Y-axis (front-back)")
3. motion_range: A single positive value representing maximum sliding distance in meters (e.g., 0.4 for ±0.4m range)
4. motion_range_description: Description of the range reasoning (e.g., "drawer slides ±0.4m based on AABB depth, total range 0.8m")
5. confidence: Your confidence in this determination (0.0-1.0)
6. reasoning: Detailed reasoning process

{parser.get_format_instructions()}
"""
    return prompt


def build_rotating_constraint_prompt(
    part_name: str,
    part_analysis: Any,  # PartAnalysisResult
    scene_description: Optional[str] = None,
    aabb_info: Optional[Dict[str, Any]] = None  # AABB信息：size, center等
) -> str:
    """构建旋转部件约束推理的 prompt
    
    Args:
        part_name: part 名称
        part_analysis: part 分析结果对象（PartAnalysisResult）
        scene_description: 场景描述文本（作为上下文）
        aabb_info: AABB信息字典，包含size, center等
        
    Returns:
        旋转约束推理的 prompt 文本（包含格式说明）
    """
    # 构建场景描述上下文
    scene_context = ""
    if scene_description:
        scene_context = f"""
Scene Description (Context):
{scene_description}

"""
    
    # 构建AABB信息上下文
    aabb_context = ""
    if aabb_info:
        size = aabb_info.get("size", [0, 0, 0])
        center = aabb_info.get("center", [0, 0, 0])
        aabb_context = f"""
Part Geometry (AABB - Axis-Aligned Bounding Box):
- Size (width, depth, height): [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}] meters
- Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]
- Largest dimension: {max(size):.4f} meters
- Smallest dimension: {min(size):.4f} meters

Use this geometry information to infer the rotation range. **IMPORTANT: Reason about the realistic opening angle based on**:
- **Scene context**: Look at the images - are there adjacent parts, walls, or structures that would limit how far the door can open?
- **Door size and position**: Larger doors or doors in corners may have different constraints than small center doors
- **Practical use**: Consider how a user would interact with this door - what angle provides reasonable access?
- **Common ranges (in degrees)**:
  * Small cabinet doors: often 90-120 degrees
  * Standard doors: typically 90-180 degrees depending on clearance
  * Large doors or standalone doors: often 180 degrees for full access
- **Output angles in DEGREES** (not radians) - use intuitive values like 90, 120, 150, 180
- **DO NOT choose very small angles** (like 30-45 degrees) unless there's clear visual evidence of severe space constraints

"""
    
    # 创建 PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=MotionConstraintLLMResponse)
    
    # 构建 prompt
    prompt = f"""
Analyze the rotation constraint of part "{part_name}" based on the scene context and previous analysis:

{scene_context}{aabb_context}Part Analysis Summary:
- Function: {part_analysis.function}
- Motion Type: {part_analysis.motion_type}
- Position: {part_analysis.position}
- Motion Description: {part_analysis.motion_description}
- Motion Axis: {part_analysis.motion_axis or 'unknown'}
- Connection Type: {part_analysis.connection_type}
- Detailed Position: {part_analysis.detailed_position}

The first image is the overall scene image, the second image is the highlighted render of part '{part_name}' (highlighted parts are marked with bright colors).

Please determine:
1. **Rotation Type**: 
   - "centerline": The part rotates around a centerline (e.g., a door rotating around a vertical hinge at the center)
   - "edge": The part rotates around an edge (e.g., a door rotating around a hinge at one edge)
   - "custom_axis": The part rotates around a custom axis not aligned with centerline or edge

2. **Motion Range** (motion_range): 
   - **CRITICAL: You must reason about the rotation angle based on the actual use case, geometry, and physical constraints**
   - **IMPORTANT: Output angles in DEGREES (not radians)** - use intuitive values like 90, 120, 150, 180 degrees
   - Analyze the images carefully to determine how far the door/door-like part can realistically open
   - Consider common opening angles:
     * **90 degrees**: Standard for doors that open to a right angle, common for cabinet doors that need to clear adjacent structures
     * **120-150 degrees**: For doors that need to open wider but may have space constraints
     * **180 degrees**: Full opening, typically for standalone doors or doors with ample clearance
   - **Reasoning factors to consider**:
     * Look at the scene images: Are there adjacent parts, walls, or structures that would limit opening?
     * Consider the door's position: Corner doors, center doors, and edge doors have different constraints
     * Think about practical use: How would a user interact with this door? What angle provides reasonable access?
     * Geometry constraints: Larger doors may need more space to fully open
   - **DO NOT default to small angles (like 45 degrees)** - most doors should open at least 90 degrees unless there's clear evidence of limitation
   - **IMPORTANT: Provide a single positive value (e.g., 90). The system will automatically convert it to symmetric range [-90, 90]**
   - The value represents the maximum angle in one direction, allowing rotation in both positive and negative directions
   - Examples: 
     * 90 = ±90 degrees (standard cabinet door, can open both ways, total range 180 degrees)
     * 120 = ±120 degrees (wide opening, can open both ways, total range 240 degrees)
     * 180 = ±180 degrees (full opening, can rotate both ways, total range 360 degrees)

Provide:
1. rotation_type: The rotation type (centerline/edge/custom_axis)
2. axis_description: Detailed description of the rotation axis (e.g., "rotates around a vertical hinge at the left edge", "rotates around a horizontal axis at the center")
3. motion_range: A single positive value representing maximum rotation angle IN DEGREES (e.g., 90 for ±90 degrees, 180 for ±180 degrees)
4. motion_range_description: Description of the range reasoning (e.g., "cabinet door opens ±90 degrees based on function and geometry, can rotate both ways")
5. confidence: Your confidence in this determination (0.0-1.0)
6. reasoning: Detailed reasoning process

{parser.get_format_instructions()}
"""
    return prompt


def _analyze_aabb_boundaries(
    aabb_info: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """分析AABB边界信息，计算各边界面的位置、法向量和语义标签
    
    Args:
        aabb_info: AABB信息字典，包含size和center
        
    Returns:
        边界信息字典，包含各边界面的信息，如果aabb_info为None则返回None
    """
    if not aabb_info:
        return None
    
    size = aabb_info.get("size", [0, 0, 0])
    center = aabb_info.get("center", [0, 0, 0])
    
    if len(size) != 3 or len(center) != 3:
        return None
    
    # 计算AABB的最小和最大坐标
    half_size = [s / 2.0 for s in size]
    min_coords = [center[i] - half_size[i] for i in range(3)]
    max_coords = [center[i] + half_size[i] for i in range(3)]
    
    # MuJoCo坐标系：X轴=左右，Y轴=前后，Z轴=上下
    # 定义各边界面的信息
    boundaries = {
        "x_min": {
            "label": "左面",
            "english_label": "left face",
            "position": [min_coords[0], center[1], center[2]],
            "normal": [-1.0, 0.0, 0.0],  # 指向-X方向
            "axis": "x",
            "direction": "negative_x"
        },
        "x_max": {
            "label": "右面",
            "english_label": "right face",
            "position": [max_coords[0], center[1], center[2]],
            "normal": [1.0, 0.0, 0.0],  # 指向+X方向
            "axis": "x",
            "direction": "positive_x"
        },
        "y_min": {
            "label": "后面",
            "english_label": "back face",
            "position": [center[0], min_coords[1], center[2]],
            "normal": [0.0, -1.0, 0.0],  # 指向-Y方向
            "axis": "y",
            "direction": "negative_y"
        },
        "y_max": {
            "label": "前面",
            "english_label": "front face",
            "position": [center[0], max_coords[1], center[2]],
            "normal": [0.0, 1.0, 0.0],  # 指向+Y方向
            "axis": "y",
            "direction": "positive_y"
        },
        "z_min": {
            "label": "下面",
            "english_label": "bottom face",
            "position": [center[0], center[1], min_coords[2]],
            "normal": [0.0, 0.0, -1.0],  # 指向-Z方向
            "axis": "z",
            "direction": "negative_z"
        },
        "z_max": {
            "label": "上面",
            "english_label": "top face",
            "position": [center[0], center[1], max_coords[2]],
            "normal": [0.0, 0.0, 1.0],  # 指向+Z方向
            "axis": "z",
            "direction": "positive_z"
        }
    }
    
    return {
        "size": size,
        "center": center,
        "min_coords": min_coords,
        "max_coords": max_coords,
        "boundaries": boundaries
    }


def build_axis_selection_prompt(
    part_name: str,
    part_analysis: Any,  # PartAnalysisResult
    candidate_axes: List[Dict[str, Any]],
    motion_type: str,  # "edge_rotation", "centerline_rotation", "sliding"
    visualization_path: Optional[str] = None,
    scene_description: Optional[str] = None,
    aabb_info: Optional[Dict[str, Any]] = None,  # 新增：AABB信息
    spatial_context: Optional[Dict[str, Any]] = None,  # 新增：空间上下文信息
    index_mapping: Optional[Dict[int, Dict[str, Any]]] = None  # 新增：序号映射（序号从1开始）
) -> str:
    """构建从候选轴中选择最合适轴的 prompt
    
    Args:
        part_name: part 名称
        part_analysis: part 分析结果对象（PartAnalysisResult）
        candidate_axes: 候选轴列表，每个轴包含：
            - 对于edge_rotation: edge_id, midpoint, direction, length, alignment_axis等
            - 对于centerline_rotation: axis_id, point, direction等
            - 对于sliding: direction_id, direction, magnitude等
        motion_type: 运动类型（"edge_rotation", "centerline_rotation", "sliding"）
        visualization_path: 可视化图像路径（如果存在）
        scene_description: 场景描述文本（作为上下文）
        aabb_info: AABB信息字典，包含size和center（新增）
        spatial_context: 空间上下文信息字典，包含相邻部件、开口方向等（新增）
        index_mapping: 序号映射字典，key为序号（从1开始），value为轴信息（包含semantic_info等）（新增）
        
    Returns:
        轴选择的 prompt 文本（包含格式说明）
    """
    # 构建场景描述上下文
    scene_context = ""
    if scene_description:
        scene_context = f"""
Scene Description (Context):
{scene_description}

"""
    
    # 构建坐标系定义
    coordinate_system_info = """
**MuJoCo Coordinate System Definition:**
- X-axis: Left-Right direction (左右方向)
  - Positive X (+X): Right (右)
  - Negative X (-X): Left (左)
- Y-axis: Front-Back direction (前后方向)
  - Positive Y (+Y): Front (前)
  - Negative Y (-Y): Back (后)
- Z-axis: Up-Down direction (上下方向)
  - Positive Z (+Z): Up (上)
  - Negative Z (-Z): Down (下)

**Important Direction Mapping:**
- "往外开" (open outward) for doors: typically means opening in the +Y direction (forward/front)
- "往外拉" (pull outward) for drawers: typically means pulling in the +Y direction (forward/front)
- "往里推" (push inward): typically means pushing in the -Y direction (backward/back)
- "向左开" (open left): means opening in the -X direction
- "向右开" (open right): means opening in the +X direction

"""
    
    # 构建AABB边界信息
    aabb_boundaries_info = ""
    if aabb_info:
        boundaries_analysis = _analyze_aabb_boundaries(aabb_info)
        if boundaries_analysis:
            boundaries = boundaries_analysis["boundaries"]
            aabb_boundaries_info = f"""
**Part AABB Boundary Information:**
- Center: [{boundaries_analysis['center'][0]:.4f}, {boundaries_analysis['center'][1]:.4f}, {boundaries_analysis['center'][2]:.4f}]
- Size: [{boundaries_analysis['size'][0]:.4f}, {boundaries_analysis['size'][1]:.4f}, {boundaries_analysis['size'][2]:.4f}] meters

**Boundary Faces (面):**
"""
            for key, boundary in boundaries.items():
                pos = boundary["position"]
                normal = boundary["normal"]
                aabb_boundaries_info += f"- {boundary['label']} ({boundary['english_label']}): position=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], normal={boundary['normal']}, direction={boundary['direction']}\n"
            
            aabb_boundaries_info += f"""
**Direction Interpretation Guide:**
- For doors that "open outward" (往外开): The door typically opens away from the cabinet body, which usually means opening in the +Y direction (forward/front face direction)
- For drawers that "pull outward" (往外拉): The drawer typically pulls out in the +Y direction (forward/front face direction)
- The "front face" (前面) is typically the y_max boundary, with normal pointing in +Y direction
- The "back face" (后面) is typically the y_min boundary, with normal pointing in -Y direction

"""
    
    # 构建空间上下文信息
    spatial_context_info = ""
    if spatial_context:
        spatial_context_info = "\n**Spatial Context Information:**\n"
        
        if spatial_context.get("adjacent_parts"):
            spatial_context_info += "- Adjacent Parts (相邻部件):\n"
            for adj_part in spatial_context["adjacent_parts"]:
                spatial_context_info += f"  * {adj_part.get('name', 'unknown')}: {adj_part.get('relationship', 'unknown')} - {adj_part.get('description', '')}\n"
        
        if spatial_context.get("opening_direction"):
            opening_dir = spatial_context["opening_direction"]
            spatial_context_info += f"- Opening Direction (开口方向): {opening_dir.get('description', 'unknown')}\n"
            if opening_dir.get("direction_vector"):
                vec = opening_dir["direction_vector"]
                spatial_context_info += f"  Direction vector: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]\n"
        
        if spatial_context.get("part_position_relative_to_scene"):
            spatial_context_info += f"- Part Position Relative to Scene: {spatial_context['part_position_relative_to_scene']}\n"
        
        spatial_context_info += "\n"
    
    # 创建 PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=AxisSelectionLLMResponse)
    
    # 根据运动类型构建不同的提示
    if motion_type == "edge_rotation":
        axis_type_name = "edge旋转轴"
        axis_description = "每条edge包含：edge_id（唯一标识）、midpoint（中点坐标）、direction（方向向量）、length（长度）、alignment_axis（对齐的轴：x/y/z）、alignment_score（对齐度分数）、importance_score（重要性分数）"
    elif motion_type == "centerline_rotation":
        axis_type_name = "中心线旋转轴"
        axis_description = "每条中心线包含：axis_id（唯一标识）、point（通过的点坐标）、direction（方向向量）、axis_type（轴类型：principal/centroid/diagonal）"
    elif motion_type == "sliding":
        axis_type_name = "滑动方向"
        axis_description = "每个方向包含：direction_id（唯一标识）、direction（方向向量）、magnitude（大小）、axis（主轴：x/y/z）"
    else:
        axis_type_name = "运动轴"
        axis_description = "每个轴包含基本信息"
    
    # 构建候选轴列表文本（按重要性排序，突出关键信息）
    candidate_axes_text = []
    for i, axis in enumerate(candidate_axes):
        axis_info = f"Candidate {i} (Index: {i}):\n"
        
        # 优先显示关键信息
        if "alignment_score" in axis:
            axis_info += f"  - alignment_score: {axis['alignment_score']:.4f} (对齐度，越高越好，>0.8为优秀)\n"
        if "importance_score" in axis:
            axis_info += f"  - importance_score: {axis['importance_score']:.4f} (重要性分数，越高越好)\n"
        if "alignment_axis" in axis:
            axis_info += f"  - alignment_axis: {axis['alignment_axis']} (对齐的轴：x/y/z)\n"
        if "is_axis_aligned" in axis:
            axis_info += f"  - is_axis_aligned: {axis['is_axis_aligned']} (是否与标准轴对齐)\n"
        if "boundary_type" in axis and axis.get("boundary_type"):
            axis_info += f"  - boundary_type: {axis['boundary_type']} (边界位置，如x_max/y_min等)\n"
        if "is_on_boundary" in axis:
            axis_info += f"  - is_on_boundary: {axis['is_on_boundary']} (是否在AABB边界上)\n"
        
        # 显示其他信息
        for key, value in axis.items():
            if key in ["alignment_score", "importance_score", "alignment_axis", "is_axis_aligned", 
                       "boundary_type", "is_on_boundary"]:
                continue  # 已经显示过了
            if isinstance(value, (list, tuple)) and len(value) == 3:
                # 格式化3D向量
                axis_info += f"  - {key}: [{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}]\n"
            elif isinstance(value, (int, float)):
                axis_info += f"  - {key}: {value:.4f}\n"
            else:
                axis_info += f"  - {key}: {value}\n"
        candidate_axes_text.append(axis_info)
    
    candidate_list_text = "\n".join(candidate_axes_text)
    
    # 构建序号映射信息
    index_mapping_text = ""
    if index_mapping:
        index_mapping_text = "\n**Index Mapping (序号映射) - Number to Direction/Axis Mapping:**\n"
        index_mapping_text += "In the visualization image, each candidate axis/direction is labeled with a number. Here's what each number represents:\n\n"
        for seq_num in sorted(index_mapping.keys()):
            info = index_mapping[seq_num]
            index_mapping_text += f"- **Number {seq_num}**:\n"
            if "semantic_info" in info:
                index_mapping_text += f"  - Semantic Information: {info['semantic_info']}\n"
            if "direction" in info:
                dir_vec = info["direction"]
                index_mapping_text += f"  - Direction Vector: [{dir_vec[0]:.4f}, {dir_vec[1]:.4f}, {dir_vec[2]:.4f}]\n"
            if "edge_id" in info:
                index_mapping_text += f"  - Edge ID: {info['edge_id']}\n"
            if "axis_id" in info:
                index_mapping_text += f"  - Axis ID: {info['axis_id']}\n"
            if "direction_id" in info:
                index_mapping_text += f"  - Direction ID: {info['direction_id']}\n"
            if "reference_direction_id" in info:
                index_mapping_text += f"  - Reference Direction ID: {info['reference_direction_id']} (original ID, e.g., positive_y)\n"
            if "length" in info:
                index_mapping_text += f"  - Length: {info['length']:.3f}m\n"
            if "importance_score" in info:
                index_mapping_text += f"  - Importance Score: {info['importance_score']:.4f}\n"
            if "alignment_axis" in info:
                index_mapping_text += f"  - Alignment Axis: {info['alignment_axis']}\n"
            if "alignment_score" in info:
                index_mapping_text += f"  - Alignment Score: {info['alignment_score']:.4f}\n"
            index_mapping_text += "\n"
        index_mapping_text += "**Important**: Use these number labels in the visualization image to identify and select the most appropriate axis/direction.\n\n"
    
    # 构建 prompt
    visualization_hint = ""
    if visualization_path:
        visualization_hint = f"\nAdditionally, a 3D visualization image is provided showing all candidate axes with different colors and numbered labels. The visualization file is at: {visualization_path}\n"
        if index_mapping:
            visualization_hint += "Each axis/direction in the visualization is labeled with a number. Refer to the Index Mapping section below to understand what each number represents.\n"
    
    # 构建更详细的part分析信息
    part_analysis_details = f"""
Part Analysis Summary:
- Function: {part_analysis.function}
- Motion Type: {part_analysis.motion_type}
- Position: {part_analysis.position}
- Motion Description: {part_analysis.motion_description}
- Motion Axis: {part_analysis.motion_axis or 'unknown'}
- Connection Type: {part_analysis.connection_type}
- Detailed Position: {part_analysis.detailed_position}
- Specific Function: {getattr(part_analysis, 'specific_function', 'unknown')}
- Interaction Method: {getattr(part_analysis, 'interaction_method', 'unknown')}
- Relative to Ground: {getattr(part_analysis, 'relative_to_ground', 'unknown')}
- Importance Level: {getattr(part_analysis, 'importance_level', 'unknown')}
"""
    
    # 构建图像说明
    image_description = """
Image Context:
- The first image is the overall scene image showing the complete device structure and all parts
- The second image is the highlighted render of the specific part (highlighted parts are marked with bright colors)
"""
    if visualization_path:
        image_description += f"- The third image is a 3D visualization showing all candidate axes with different colors and numbered labels (file: {visualization_path})\n"
    
    prompt = f"""
Analyze and select the most appropriate {axis_type_name} for part "{part_name}" from the candidate list:

{coordinate_system_info}{aabb_boundaries_info}{spatial_context_info}{scene_context}{part_analysis_details}
{image_description}
{index_mapping_text}
Candidate {axis_type_name} List:
{candidate_list_text}
{visualization_hint}
Please carefully analyze each candidate and select the most appropriate one based on:

**CRITICAL SELECTION CRITERIA (in priority order):**

1. **Alignment Score (MOST IMPORTANT)**: 
   - **PRIORITIZE candidates with alignment_score > 0.8** (high alignment with standard axes X/Y/Z)
   - For edge rotation: **PREFER vertical edges (alignment_axis='z')** with high alignment_score, as most doors/cabinets rotate around vertical hinges
   - For sliding: prefer edges aligned with the sliding direction

2. **Geometric Alignment**: 
   - The selected axis should align with the part's actual structure (visible in images)
   - For rotation: the axis should be at a logical hinge location (typically at edges/boundaries)
   - Check if the axis is on a boundary (is_on_boundary=True) - this is often important for hinges

3. **Direction Consistency with AABB Boundaries and Spatial Context**: 
   - **CRITICAL**: Use the AABB boundary information and spatial context to determine the correct direction
   - For doors that "open outward" (往外开): 
     * Typically opens in the +Y direction (forward/front face direction)
     * The rotation axis should be at the edge opposite to the opening direction (e.g., if opening forward, axis at back edge)
   - For drawers that "pull outward" (往外拉):
     * Typically pulls in the +Y direction (forward/front face direction)
     * Select the direction vector that matches positive_y [+1.0, 0.0, 0.0] or the direction closest to the front face normal
   - Match the candidate direction vectors with the boundary face normals to determine which direction is "outward"
   - Consider the spatial context: if the part is adjacent to a fixed part, the opening direction is typically away from that fixed part

4. **Importance Score**: 
   - Higher importance_score indicates better combination of length, alignment, and boundary position
   - Use this as a tie-breaker when alignment_scores are similar

5. **Consistency with Motion Description**: 
   - The axis should match the part's motion description and function
   - For doors: vertical rotation axis (Z-axis) is standard
   - **For drawers/sliding parts**: 
     - **PREFER Y-axis (positive_y or negative_y) for front-back sliding** - this is the most common direction for drawers
     - X-axis (left-right) is less common unless the part is clearly oriented horizontally
     - Consider the part's position: drawers typically slide forward/backward (Y-axis), not left/right (X-axis)
     - Look at the scene images to determine the logical sliding direction based on the cabinet structure

6. **Visual Evidence**: 
   - Use the visualization image to verify the axis location matches the expected hinge/sliding mechanism
   - The axis should align with visible structural features in the images

**IMPORTANT**: 
- **ALWAYS prefer candidates with alignment_score >= 0.8** over those with lower scores
- For edge rotation, **strongly prefer vertical edges (Z-axis)** unless there's clear evidence otherwise
- **For sliding (drawers)**: 
  - **STRONGLY PREFER Y-axis directions (positive_y or negative_y)** unless there's clear evidence the drawer slides left-right
  - Consider the part's function: drawers slide in/out (front-back), which is typically Y-axis
  - X-axis (left-right) is less common for drawers unless clearly oriented horizontally
  - Use the scene images and visualization to verify the logical direction
- **For direction selection, CRITICALLY IMPORTANT**: 
  - Match the motion description (e.g., "往外开", "往外拉") with the coordinate system and AABB boundaries
  - "往外" (outward) typically means +Y direction (forward/front)
  - Compare candidate direction vectors with boundary face normals to determine the correct "outward" direction
- Pay special attention to the visualization image (if provided) where each candidate axis is shown in a different color

Provide:
1. selected_axis_id: The ID of the selected axis (must match one of the candidate IDs)
2. selected_index: The index of the selected axis in the candidate list (0-based)
3. confidence: Your confidence in this selection (0.0-1.0)
4. reasoning: Detailed reasoning process explaining why this axis is the most appropriate, including references to the images and scene context
5. alternative_axis_ids: (Optional) List of alternative axis IDs if you're not completely certain

{parser.get_format_instructions()}
"""
    return prompt


def build_user_hint_parsing_prompt(
    user_hint: str,
    available_parts: List[str]
) -> str:
    """构建用户提示解析的prompt
    
    Args:
        user_hint: 用户的自然语言提示
        available_parts: 可用的部件名称列表
        
    Returns:
        用户提示解析的prompt文本
    """
    parts_list = "\n".join([f"- {part}" for part in available_parts])
    
    # 构建部件名称映射提示（帮助LLM匹配）
    part_mapping_hints = []
    for part in available_parts:
        part_lower = part.lower()
        hints = []
        if "button" in part_lower or "knob" in part_lower:
            hints.append("按钮/按键")
        if "lid" in part_lower or "cover" in part_lower:
            hints.append("盖子/盖")
        if "door" in part_lower:
            hints.append("门")
        if "drawer" in part_lower:
            hints.append("抽屉")
        if hints:
            part_mapping_hints.append(f"  - {part}: 可能对应 {', '.join(hints)}")
    
    mapping_text = "\n".join(part_mapping_hints) if part_mapping_hints else "  （无特殊映射）"
    
    prompt = f"""请解析以下用户提示，提取出部件名称、运动类型、方向/旋转类型和运动范围等信息。

用户提示："{user_hint}"

可用的部件名称列表：
{parts_list}

部件名称映射提示（帮助匹配）：
{mapping_text}

**重要提示**：
- 如果用户提到"按钮"、"按键"，请从包含"button"或"knob"的部件名称中选择
- 如果用户提到"盖子"、"盖"，请从包含"lid"或"cover"的部件名称中选择
- 如果用户提到"门"，请从包含"door"的部件名称中选择
- 如果用户提到"抽屉"，请从包含"drawer"的部件名称中选择
- **必须从上述可用部件列表中选择一个确切的部件名称，不能返回中文名称**

请根据用户提示，识别：
1. 部件名称（part_name）：**必须**从可用部件列表中选择一个部件名称。如果用户提示中提到的是部件的功能或特征（如"按钮"、"盖子"、"门"等），请根据语义匹配到最接近的部件名称：
   - "按钮"、"按键"、"button" -> 通常对应 "knob" 或包含 "button" 的部件
   - "盖子"、"盖"、"lid" -> 通常对应 "lid" 或包含 "lid" 的部件
   - "门"、"door" -> 通常对应 "door" 或包含 "door" 的部件
   - "抽屉"、"drawer" -> 通常对应 "drawer" 或包含 "drawer" 的部件
   如果无法确定，请选择语义最接近的部件名称。**重要：返回的part_name必须是可用部件列表中的一个。**
2. 运动类型（motion_type）：
   - "fixed": 固定部件，不运动
   - "sliding": 滑动运动（如抽屉、推拉门、按钮按下等）
   - "rotating": 旋转运动（如门、盖子、旋钮等）
3. 滑动方向（sliding_direction，仅当motion_type=sliding时）：
   - "x": 左右滑动
   - "y": 前后滑动
   - "z": 上下滑动（如按钮按下）
4. 旋转类型（rotation_type，仅当motion_type=rotating时）：
   - "centerline": 绕中心线旋转
   - "edge": 绕边旋转
   - "custom_axis": 绕自定义轴旋转
5. 运动范围（motion_range，可选）：
   - 滑动：距离（米），如0.1表示±0.1米
   - 旋转：角度（度），如90表示±90度
   - 如果用户提示中没有明确提到范围，则为None

注意：
- 用户提示是绝对正确的，请严格按照用户提示解析
- 如果用户说"按下去"、"按下"、"推"等，通常是sliding运动，方向为z（上下）
- 如果用户说"旋转"、"转动"、"打开"（门）等，通常是rotating运动
- 如果用户说"固定"、"不动"等，通常是fixed运动
- 请仔细分析用户提示中的关键词，准确识别运动类型和方向

请以JSON格式输出解析结果。"""
    
    return prompt


__all__ = [
    "build_scene_analysis_prompt",
    "build_part_analysis_prompt",
    "build_part_motion_type_prompt",
    "build_sliding_constraint_prompt",
    "build_rotating_constraint_prompt",
    "build_axis_selection_prompt",
    "build_user_hint_parsing_prompt",
]

