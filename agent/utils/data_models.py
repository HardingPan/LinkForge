"""
数据模型定义
使用Pydantic定义part分析的结构化输出
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum


class RelationshipType(str, Enum):
    """关系类型枚举"""
    FIXED = "fixed"            # 固定连接
    SLIDING = "sliding"        # 滑动连接
    HINGE = "hinge"           # 铰连接
    NO_RELATION = "no_relation"  # 无明显连接关系


class PartMotionType(str, Enum):
    """部件运动类型枚举"""
    FIXED = "fixed"            # 固定部件（本体）
    ROTATING = "rotating"      # 旋转部件
    SLIDING = "sliding"        # 滑动部件
    UNKNOWN = "unknown"        # 未知运动类型


class GeometryInfo(BaseModel):
    """几何信息"""
    aabb_min: Tuple[float, float, float] = Field(..., description="包围盒最小点")
    aabb_max: Tuple[float, float, float] = Field(..., description="包围盒最大点")
    aabb_center: Tuple[float, float, float] = Field(..., description="包围盒中心点")
    aabb_size: Tuple[float, float, float] = Field(..., description="包围盒尺寸")
    
    @validator('aabb_min', 'aabb_max', 'aabb_center', 'aabb_size')
    def validate_tuple_length(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("坐标必须是3元组")
        return v


class MotionAnalysisInfo(BaseModel):
    """运动分析信息"""
    motion_type: PartMotionType = Field(..., description="运动类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="运动类型判断置信度")
    reasoning: str = Field(..., description="运动类型判断的推理过程")
    motion_axis: Optional[Tuple[float, float, float]] = Field(None, description="运动轴方向（旋转轴或滑动方向）")
    motion_center: Optional[Tuple[float, float, float]] = Field(None, description="运动中心点")
    motion_range: Optional[Tuple[float, float]] = Field(None, description="运动范围 [min, max]")
    
    class Config:
        use_enum_values = True


class PartInfo(BaseModel):
    """部件信息"""
    name: str = Field(..., description="部件名称")
    function: str = Field(..., description="部件功能描述")
    shape_description: str = Field(..., description="形状描述")
    position_description: str = Field(..., description="位置描述")
    geometry: GeometryInfo = Field(..., description="几何信息")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分析置信度")
    motion_analysis: Optional[MotionAnalysisInfo] = Field(None, description="运动分析信息")


class PartRelationship(BaseModel):
    """部件关系"""
    source_part: str = Field(..., description="源部件名称")
    target_part: str = Field(..., description="目标部件名称")
    relationship_types: List[RelationshipType] = Field(..., description="关系类型列表（可能多种）")
    description: str = Field(..., description="关系描述")
    confidence: float = Field(..., ge=0.0, le=1.0, description="关系置信度")
    reasoning: Optional[str] = Field(None, description="关系判断的推理过程")
    
    class Config:
        use_enum_values = True


class TopologyGraph(BaseModel):
    """拓扑图结构"""
    nodes: List[str] = Field(..., description="节点列表（部件名称）")
    edges: List[PartRelationship] = Field(..., description="边列表（部件关系）")
    main_structure: Optional[str] = Field(None, description="主体结构部件名称")
    
    def get_relationships_for_part(self, part_name: str) -> List[PartRelationship]:
        """获取与指定部件相关的所有关系"""
        return [
            rel for rel in self.edges 
            if rel.source_part == part_name or rel.target_part == part_name
        ]
    
    def get_connected_parts(self, part_name: str) -> List[str]:
        """获取与指定部件直接连接的其他部件"""
        connected = []
        for rel in self.edges:
            if rel.source_part == part_name:
                connected.append(rel.target_part)
            elif rel.target_part == part_name:
                connected.append(rel.source_part)
        return connected


class AssemblyInfo(BaseModel):
    """装配信息"""
    device_name: str = Field(..., description="设备名称")
    device_type: str = Field(..., description="设备类型")
    device_description: str = Field(..., description="设备整体描述")
    main_function: str = Field(..., description="主要功能")
    total_parts: int = Field(..., description="总部件数")
    main_components: List[str] = Field(..., description="主要组件列表")
    complexity_score: float = Field(..., ge=0.0, le=10.0, description="复杂度评分")


class MotionAnalysisResult(BaseModel):
    """运动分析结果"""
    fixed_parts: List[str] = Field(..., description="固定部件（本体）列表")
    rotating_parts: List[str] = Field(..., description="旋转部件列表")
    sliding_parts: List[str] = Field(..., description="滑动部件列表")
    unknown_parts: List[str] = Field(..., description="未知运动类型部件列表")
    motion_relationships: List[PartRelationship] = Field(..., description="运动关系列表")
    
    def get_motion_parts(self) -> List[str]:
        """获取所有运动部件"""
        return self.rotating_parts + self.sliding_parts
    
    def get_all_parts(self) -> List[str]:
        """获取所有部件"""
        return self.fixed_parts + self.rotating_parts + self.sliding_parts + self.unknown_parts


class AnalysisResult(BaseModel):
    """分析结果"""
    task_instruction: str = Field(..., description="任务指令")
    model_path: str = Field(..., description="模型文件路径")
    assembly: AssemblyInfo = Field(..., description="装配信息")
    parts: List[PartInfo] = Field(..., description="部件列表")
    topology: TopologyGraph = Field(..., description="拓扑图结构")
    motion_analysis: Optional[MotionAnalysisResult] = Field(None, description="运动分析结果")
    analysis_summary: str = Field(..., description="分析总结")
    confidence_overall: float = Field(..., ge=0.0, le=1.0, description="整体分析置信度")
    processing_time: float = Field(..., description="处理时间（秒）")
    
    def get_part_by_name(self, name: str) -> Optional[PartInfo]:
        """根据名称获取部件信息"""
        for part in self.parts:
            if part.name == name:
                return part
        return None
    
    def get_high_confidence_parts(self, threshold: float = 0.7) -> List[PartInfo]:
        """获取高置信度的部件"""
        return [part for part in self.parts if part.confidence >= threshold]
    
    def get_motion_parts(self) -> List[PartInfo]:
        """获取运动部件"""
        return [part for part in self.parts if part.motion_analysis and part.motion_analysis.motion_type in [PartMotionType.ROTATING, PartMotionType.SLIDING]]
    
    def get_fixed_parts(self) -> List[PartInfo]:
        """获取固定部件"""
        return [part for part in self.parts if part.motion_analysis and part.motion_analysis.motion_type == PartMotionType.FIXED]


class PartAnalysisRequest(BaseModel):
    """part分析请求"""
    xml_path: str = Field(..., description="XML文件路径")
    task_instruction: str = Field(..., description="任务指令")
    target_parts: Optional[List[str]] = Field(None, description="指定要分析的部件名称列表，None表示分析所有部件")
    device_type_hint: Optional[str] = Field(None, description="设备类型提示（如：柜子、桌子、机械装置等），如果提供则用于指导分析")
    analysis_options: Dict[str, Any] = Field(default_factory=dict, description="分析选项")


class LLMDeviceInfo(BaseModel):
    """LLM输出的设备信息"""
    name: str = Field(..., description="设备名称")
    type: str = Field(..., description="设备类型")
    description: str = Field(..., description="设备描述")
    main_function: str = Field(..., description="主要功能")


class LLMPartInfo(BaseModel):
    """LLM输出的部件信息"""
    name: str = Field(..., description="部件名称")
    function: str = Field(..., description="功能描述")
    shape_description: str = Field(..., description="形状描述")
    position_description: str = Field(..., description="位置描述")


class LLMRelationshipInfo(BaseModel):
    """LLM输出的关系信息"""
    source: str = Field(..., description="源部件名称")
    target: str = Field(..., description="目标部件名称")
    relationship_types: List[RelationshipType] = Field(..., description="关系类型列表")
    description: str = Field(..., description="关系描述")
    reasoning: str = Field(..., description="推理过程")


class LLMAnalysisResult(BaseModel):
    """LLM输出的完整分析结果"""
    device_info: LLMDeviceInfo = Field(..., description="设备信息")
    parts: List[LLMPartInfo] = Field(..., description="部件列表")
    relationships: List[LLMRelationshipInfo] = Field(..., description="关系列表")


class PartMotionTypeResponse(BaseModel):
    """快速分析响应模型 - 仅包含运动类型"""
    motion_type: str = Field(..., description="Motion type: fixed/sliding/rotating")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    brief_reasoning: str = Field(..., description="Brief reasoning for motion type determination")


class SlidingDirectionResponse(BaseModel):
    """滑动方向推理响应模型"""
    sliding_direction: str = Field(..., description="Sliding direction: x (left-right), y (front-back), or z (up-down)")
    sliding_orientation: str = Field(..., description="Sliding orientation description (e.g., front-to-back, left-to-right)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    reasoning: str = Field(..., description="Reasoning for sliding direction determination")


class RotatingAxisTypeResponse(BaseModel):
    """旋转轴类型推理响应模型"""
    rotation_type: str = Field(..., description="Rotation type: centerline/edge/custom_axis")
    axis_description: str = Field(..., description="Axis description (e.g., vertical centerline, top edge, custom axis)")
    axis_location: str = Field(..., description="Axis location description (e.g., center, top edge, bottom edge, left edge, right edge)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    reasoning: str = Field(..., description="Reasoning for rotation type determination")


class SlidingDirectionType(str, Enum):
    """滑动方向类型（简化为三个轴）"""
    X_AXIS = "x"      # X轴（左右）
    Y_AXIS = "y"      # Y轴（前后）
    Z_AXIS = "z"      # Z轴（上下）


class RotationType(str, Enum):
    """旋转类型"""
    CENTERLINE = "centerline"  # 绕中心线旋转
    EDGE = "edge"             # 绕边旋转
    CUSTOM_AXIS = "custom_axis"  # 绕自定义轴旋转


class MotionConstraintLLMResponse(BaseModel):
    """LLM运动约束推理响应模型"""
    # 滑动相关（当motion_type=sliding时）
    sliding_direction: Optional[str] = Field(
        None, 
        description="Sliding direction: x (left-right), y (front-back), or z (up-down)"
    )
    sliding_orientation: Optional[str] = Field(
        None, 
        description="Detailed sliding orientation description (e.g., 'slides horizontally from left to right')"
    )
    
    # 旋转相关（当motion_type=rotating时）
    rotation_type: Optional[str] = Field(
        None, 
        description="Rotation type: centerline/edge/custom_axis. "
                   "centerline: rotates around a centerline through the part; "
                   "edge: rotates around an edge of the part; "
                   "custom_axis: rotates around a custom axis (not centerline or edge)"
    )
    axis_description: Optional[str] = Field(
        None, 
        description="Description of the rotation axis (e.g., 'vertical centerline', 'horizontal edge at top')"
    )
    axis_location: Optional[str] = Field(
        None, 
        description="Location description of the rotation axis (e.g., 'at the center', 'along the top edge')"
    )
    
    # 运动范围（新增，简化为单个值，系统会自动转换为对称范围）
    motion_range: Optional[float] = Field(
        None,
        description="Motion range value (single number). "
                   "For sliding: maximum sliding distance in meters (e.g., 0.4 means ±0.4m, total range 0.8m). "
                   "For rotating: maximum rotation angle in DEGREES (NOT radians) (e.g., 90 means ±90 degrees, total range 180 degrees). "
                   "IMPORTANT: Use degrees for rotation angles - it's more intuitive and less error-prone. "
                   "The system will automatically convert this to symmetric range [-value, value]. "
                   "Examples: 0.4 for sliding (±0.4m), 90 for rotation (±90 degrees), 180 for full rotation (±180 degrees). "
                   "Reason about realistic values based on scene context, geometry, and use case. "
                   "Most doors should open at least 90 degrees. "
                   "DO NOT choose very small values (like 30-45 degrees) unless there's clear evidence of space constraints."
    )
    motion_range_description: Optional[str] = Field(
        None,
        description="Description of the motion range reasoning (e.g., 'cabinet door opens 90 degrees', 'drawer slides 0.3m based on AABB size')"
    )
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Reasoning confidence")
    reasoning: str = Field(..., description="Detailed reasoning process for the constraint inference")


class AxisSelectionLLMResponse(BaseModel):
    """LLM轴选择响应模型"""
    selected_axis_id: str = Field(..., description="选中的轴ID（从候选轴列表中选择）")
    selected_index: int = Field(..., description="选中的轴在候选列表中的索引（从0开始）")
    confidence: float = Field(..., ge=0.0, le=1.0, description="选择置信度")
    reasoning: str = Field(..., description="选择该轴的详细推理过程")
    alternative_axis_ids: Optional[List[str]] = Field(None, description="备选轴ID列表（如果主要选择不确定）")


class MotionConstraintResult(BaseModel):
    """运动约束推理结果"""
    part_name: str = Field(..., description="部件名称")
    motion_type: str = Field(..., description="运动类型: fixed/sliding/rotating")
    
    # 滑动相关
    sliding_direction: Optional[str] = Field(None, description="滑动方向类型")
    sliding_orientation: Optional[str] = Field(None, description="滑动方向详细描述")
    
    # 旋转相关
    rotation_type: Optional[str] = Field(None, description="旋转类型: centerline/edge/custom_axis")
    axis_description: Optional[str] = Field(None, description="旋转轴描述")
    axis_location: Optional[str] = Field(None, description="旋转轴位置描述")
    
    # 新增：选中的轴信息（从工具分析结果中选择）
    selected_axis: Optional[Dict[str, Any]] = Field(None, description="选中的轴详细信息（包含point、direction等）")
    selected_axis_id: Optional[str] = Field(None, description="选中的轴ID")
    selected_axis_info: Optional[Dict[str, Any]] = Field(None, description="选中的轴详细信息（结构化格式，便于显示）")
    all_candidate_axes: Optional[List[Dict[str, Any]]] = Field(None, description="所有候选轴列表")
    axis_selection_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="轴选择置信度")
    axis_selection_reasoning: Optional[str] = Field(None, description="轴选择推理过程")
    visualization_path: Optional[str] = Field(None, description="可视化图像路径")
    
    # 运动范围（新增，简化为单个值，系统会自动转换为对称范围）
    motion_range: Optional[Tuple[float, float]] = Field(None, description="运动范围 [min, max]。滑动：距离（米）；旋转：角度（度数，非弧度）。注意：内部存储为对称范围，但LLM只需输出单个值")
    motion_range_description: Optional[str] = Field(None, description="运动范围描述和推理")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="推理置信度")
    reasoning: str = Field(..., description="推理过程")
    timestamp: float = Field(..., description="推理时间戳")


class PartAnalysisLLMResponse(BaseModel):
    """LLM分析响应的结构化模型"""
    function: str = Field(..., description="Part function")
    motion_type: str = Field(..., description="Motion type: fixed/sliding/rotating")
    position: str = Field(..., description="Position")
    material: str = Field(..., description="Material")
    detailed_position: str = Field(..., description="Detailed position")
    specific_function: str = Field(..., description="Specific function")
    motion_description: str = Field(..., description="Motion description")
    motion_axis: Optional[str] = Field(None, description="Motion axis direction")
    motion_range: Optional[str] = Field(None, description="Motion range description")
    interaction_method: str = Field(..., description="Interaction method")
    relative_to_ground: str = Field(..., description="Relative to ground relationship")
    connection_type: str = Field(..., description="Connection type")
    importance_level: str = Field(..., description="Importance level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    analysis_summary: str = Field(..., description="Analysis summary")


class UserHintParsedResult(BaseModel):
    """用户提示解析结果"""
    part_name: str = Field(..., description="部件名称（从用户提示中识别）")
    motion_type: str = Field(..., description="运动类型: fixed/sliding/rotating")
    sliding_direction: Optional[str] = Field(
        None, 
        description="滑动方向（仅当motion_type=sliding时）: x/y/z。x=左右，y=前后，z=上下"
    )
    rotation_type: Optional[str] = Field(
        None,
        description="旋转类型（仅当motion_type=rotating时）: centerline/edge/custom_axis"
    )
    motion_range: Optional[float] = Field(
        None,
        description="运动范围。滑动：距离（米），旋转：角度（度）。如果用户提示中没有明确提到，则为None"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="解析置信度")
    reasoning: str = Field(..., description="解析推理过程")


class PartAnalysisResult(BaseModel):
    """单个part分析结果"""
    part_name: str = Field(..., description="部件名称")
    function: str = Field(..., description="部件功能")
    motion_type: str = Field(..., description="运动类型")
    position: str = Field(..., description="位置")
    material: str = Field(..., description="材质")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分析置信度")
    analysis_text: str = Field(..., description="详细分析文本")
    image_path: str = Field(..., description="高亮渲染图像路径")
    processing_time: float = Field(..., description="处理时间（秒）")
    timestamp: float = Field(..., description="分析时间戳")
    
    # 新增详细语义信息
    detailed_position: str = Field(..., description="详细位置描述")
    specific_function: str = Field(..., description="具体功能描述")
    motion_description: str = Field(..., description="运动方式详细描述")
    motion_axis: Optional[str] = Field(None, description="运动轴方向")
    motion_range: Optional[str] = Field(None, description="运动范围描述")
    interaction_method: str = Field(..., description="交互方式")
    relative_to_ground: str = Field(..., description="相对于地面的关系")
    connection_type: str = Field(..., description="连接方式")
    importance_level: str = Field(..., description="重要性级别")


class ScenePartAnalysisResult(BaseModel):
    """场景所有part分析结果"""
    task_id: str = Field(..., description="任务ID")
    scene_info: Dict[str, Any] = Field(..., description="场景信息")
    parts_analysis: List[PartAnalysisResult] = Field(..., description="所有part分析结果")
    motion_parts: List[str] = Field(..., description="运动部件列表")
    fixed_parts: List[str] = Field(..., description="固定部件列表")
    unknown_parts: List[str] = Field(..., description="未知运动类型部件列表")
    total_processing_time: float = Field(..., description="总处理时间（秒）")
    analysis_timestamp: float = Field(..., description="分析完成时间戳")
    
    def get_part_by_name(self, name: str) -> Optional[PartAnalysisResult]:
        """根据名称获取part分析结果"""
        for part in self.parts_analysis:
            if part.part_name == name:
                return part
        return None
    
    def get_motion_parts_by_type(self, motion_type: str) -> List[PartAnalysisResult]:
        """根据运动类型获取part列表"""
        return [part for part in self.parts_analysis if part.motion_type == motion_type]
    
    def get_high_confidence_parts(self, threshold: float = 0.7) -> List[PartAnalysisResult]:
        """获取高置信度的part"""
        return [part for part in self.parts_analysis if part.confidence >= threshold]


class PartAnalysisResponse(BaseModel):
    """part分析响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    result: Optional[AnalysisResult] = Field(None, description="分析结果")
    error_details: Optional[str] = Field(None, description="错误详情")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")


__all__ = [
    "RelationshipType", 
    "PartMotionType",
    "SlidingDirectionType",
    "RotationType",
    "MotionAnalysisInfo",
    "MotionAnalysisResult",
    "GeometryInfo",
    "PartInfo",
    "PartRelationship",
    "TopologyGraph",
    "AssemblyInfo",
    "AnalysisResult",
    "PartAnalysisRequest",
    "PartAnalysisResponse",
    "PartMotionTypeResponse",
    "AxisSelectionLLMResponse",
    "MotionConstraintLLMResponse",
    "MotionConstraintResult",
    "PartAnalysisLLMResponse",
    "PartAnalysisResult",
    "ScenePartAnalysisResult",
    "LLMDeviceInfo",
    "LLMPartInfo", 
    "LLMRelationshipInfo",
    "LLMAnalysisResult",
    "UserHintParsedResult",
]
