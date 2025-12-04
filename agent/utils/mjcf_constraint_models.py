"""
MJCF约束标准格式定义
定义旋转、滑动等约束的标准格式，用于将运动信息转换为MJCF约束
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class JointType(str, Enum):
    """关节类型枚举（只支持三种基本类型）"""
    HINGE = "hinge"      # 旋转关节（铰链）- 用于edge旋转和centerline旋转
    SLIDE = "slide"     # 滑动关节（平移）- 用于滑动
    FIXED = "fixed"     # 固定关节（无自由度）- 用于固定部件


class ConstraintType(str, Enum):
    """约束类型枚举"""
    JOINT = "joint"           # 关节约束
    EQUALITY_CONNECT = "connect"  # 等式约束：球铰连接
    EQUALITY_WELD = "weld"    # 等式约束：刚接
    EQUALITY_JOINT = "joint_coupling"  # 等式约束：关节耦合


class MJCFJointSpec(BaseModel):
    """MJCF关节规范"""
    name: str = Field(..., description="关节名称")
    type: JointType = Field(..., description="关节类型")
    body_name: str = Field(..., description="关节所属的body名称")
    
    # 轴信息
    axis: Tuple[float, float, float] = Field(..., description="关节轴方向向量（归一化）")
    pos: Optional[Tuple[float, float, float]] = Field(None, description="关节位置（相对于body）")
    
    # 限制和物理参数
    limited: bool = Field(False, description="是否限制范围")
    range: Optional[Tuple[float, float]] = Field(None, description="关节范围 [min, max]")
    damping: Optional[float] = Field(None, description="阻尼系数")
    stiffness: Optional[float] = Field(None, description="刚度系数")
    armature: Optional[float] = Field(None, description="惯性系数")
    
    # 父body（用于连接）
    parent_body: Optional[str] = Field(None, description="父body名称（如果关节连接两个body）")


class MJCFSiteSpec(BaseModel):
    """MJCF站点规范（用于约束连接点）"""
    name: str = Field(..., description="站点名称")
    body_name: str = Field(..., description="站点所属的body名称")
    pos: Optional[Tuple[float, float, float]] = Field(None, description="站点位置（相对于body），如果设置了fromto则可以为None")
    size: Optional[float] = Field(0.005, description="站点大小（用于可视化）")
    type: str = Field("sphere", description="站点类型：sphere/capsule/box")
    
    # 对于capsule类型
    fromto: Optional[Tuple[float, float, float, float, float, float]] = Field(
        None, description="胶囊的起点和终点 (x1 y1 z1 x2 y2 z2)"
    )


class MJCFEqualityConstraintSpec(BaseModel):
    """MJCF等式约束规范"""
    name: str = Field(..., description="约束名称")
    type: ConstraintType = Field(..., description="约束类型")
    
    # 连接两个site
    site1: str = Field(..., description="第一个站点名称")
    site2: str = Field(..., description="第二个站点名称")
    
    # 物理参数
    solref: Optional[Tuple[float, float]] = Field(
        (0.01, 1.0), description="求解器参考参数 (timeconst, dampratio)"
    )
    solimp: Optional[Tuple[float, float, float, float, float]] = Field(
        (0.9, 0.95, 0.001, 0.5, 2.0), 
        description="求解器阻抗参数 (dmin, dmax, width, midpoint, power)"
    )


class MJCFBodySpec(BaseModel):
    """MJCF Body规范（用于创建feature_frame）"""
    name: str = Field(..., description="Body名称")
    parent_body: str = Field(..., description="父body名称")
    pos: Tuple[float, float, float] = Field((0.0, 0.0, 0.0), description="Body位置（相对于父body）")
    quat: Optional[Tuple[float, float, float, float]] = Field(
        (1.0, 0.0, 0.0, 0.0), description="Body四元数旋转"
    )


class MJCFConstraintPlan(BaseModel):
    """MJCF约束方案（完整的约束定义）"""
    part_name: str = Field(..., description="部件名称")
    motion_type: str = Field(..., description="运动类型：rotating/sliding/fixed")
    rotation_type: Optional[str] = Field(None, description="旋转类型：edge/centerline（仅当motion_type=rotating时）")
    
    # 关节定义
    joint: Optional[MJCFJointSpec] = Field(None, description="关节规范（fixed类型不需要joint）")
    
    # 站点定义（用于约束连接和可视化）
    sites: List[MJCFSiteSpec] = Field(default_factory=list, description="站点列表")
    
    # 等式约束（如果需要连接两个body）
    equality_constraints: List[MJCFEqualityConstraintSpec] = Field(
        default_factory=list, description="等式约束列表"
    )
    
    # Feature frame（如果需要）
    feature_frame: Optional[MJCFBodySpec] = Field(None, description="特征框架body")
    
    # 元数据
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="约束方案置信度")
    reasoning: str = Field("", description="约束方案推理过程")


class MJCFGenerationResult(BaseModel):
    """MJCF生成结果"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果消息")
    xml_path: Optional[str] = Field(None, description="生成的XML文件路径")
    constraint_plans: List[MJCFConstraintPlan] = Field(
        default_factory=list, description="约束方案列表"
    )
    modifications: List[str] = Field(
        default_factory=list, description="修改说明列表"
    )


__all__ = [
    "JointType",
    "ConstraintType",
    "MJCFJointSpec",
    "MJCFSiteSpec",
    "MJCFEqualityConstraintSpec",
    "MJCFBodySpec",
    "MJCFConstraintPlan",
    "MJCFGenerationResult",
]

