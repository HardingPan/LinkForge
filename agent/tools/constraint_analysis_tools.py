"""
约束分析工具
用于分析mesh的几何特征，找到可能的旋转轴和滑动方向
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import numpy as np
    import trimesh
except ImportError:
    raise RuntimeError(
        "需要依赖 numpy 与 trimesh。请先安装：pip install numpy trimesh"
    )

from .base_tool import Tool, ToolResult
from ..utils.mesh_analyzer import MeshAnalyzer


Vector3 = Tuple[float, float, float]
Vector3Array = List[Vector3]


class FindCenterlineAxesTool(Tool):
    """查找中心线轴工具
    
    分析mesh的几何结构，找到所有可能的中心线作为旋转轴候选。
    中心线包括：
    - 通过AABB中心的3条主轴（X、Y、Z）
    - 通过mesh质心的主轴
    - 其他可能的中心线（如对角中心线）
    """
    
    def __init__(self):
        super().__init__(
            name="find_centerline_axes",
            description="查找mesh的所有可能的中心线作为旋转轴候选"
        )
    
    def execute(
        self,
        xml_path: str,
        part_name: str,
        include_diagonal: bool = False,
        **kwargs
    ) -> ToolResult:
        """执行中心线轴查找
        
        Args:
            xml_path: XML文件路径
            part_name: part名称（mesh名称）
            include_diagonal: 是否包含对角中心线
            
        Returns:
            ToolResult包含所有可能的中心线轴信息
        """
        try:
            # 验证参数
            if not xml_path or not Path(xml_path).exists():
                return ToolResult(
                    success=False,
                    message=f"XML文件不存在: {xml_path}"
                )
            
            # 使用MeshAnalyzer加载mesh信息
            analyzer = MeshAnalyzer(xml_path)
            mesh_info_dict = analyzer.analyze()
            
            if part_name not in mesh_info_dict:
                return ToolResult(
                    success=False,
                    message=f"未找到part: {part_name}"
                )
            
            mesh_info = mesh_info_dict[part_name]
            
            # 1. 从AABB获取3条主轴中心线
            principal_axes = self._extract_principal_centerlines(mesh_info)
            
            # 2. 从mesh几何体提取质心中心线
            centroid_axes = self._extract_centroid_centerlines(mesh_info)
            
            # 3. 提取对角中心线（如果启用）
            diagonal_axes = []
            if include_diagonal:
                diagonal_axes = self._extract_diagonal_centerlines(mesh_info)
            
            # 合并所有中心线
            all_centerlines = {
                "principal_axes": principal_axes,
                "centroid_axes": centroid_axes,
                "diagonal_axes": diagonal_axes,
                "total_count": len(principal_axes) + len(centroid_axes) + len(diagonal_axes)
            }
            
            return ToolResult(
                success=True,
                message=f"找到 {all_centerlines['total_count']} 条可能的中心线轴",
                data={
                    "part_name": part_name,
                    "centerlines": all_centerlines,
                    "mesh_info": {
                        "aabb_center": mesh_info.aabb.center,
                        "aabb_size": mesh_info.aabb.size
                    },
                    "mesh_info_dict": mesh_info_dict  # 返回完整的mesh_info_dict
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"查找中心线轴失败: {str(e)}"
            )
    
    def _extract_principal_centerlines(self, mesh_info) -> List[Dict[str, Any]]:
        """提取3条主轴中心线（通过AABB中心）"""
        aabb_center = mesh_info.aabb.center
        
        centerlines = [
            {
                "axis_id": "x_axis_centerline",
                "axis": "x",
                "point": aabb_center,
                "direction": [1.0, 0.0, 0.0],
                "description": "X轴中心线（通过AABB中心）"
            },
            {
                "axis_id": "y_axis_centerline",
                "axis": "y",
                "point": aabb_center,
                "direction": [0.0, 1.0, 0.0],
                "description": "Y轴中心线（通过AABB中心）"
            },
            {
                "axis_id": "z_axis_centerline",
                "axis": "z",
                "point": aabb_center,
                "direction": [0.0, 0.0, 1.0],
                "description": "Z轴中心线（通过AABB中心）"
            }
        ]
        
        return centerlines
    
    def _extract_centroid_centerlines(self, mesh_info) -> List[Dict[str, Any]]:
        """提取通过mesh质心的中心线"""
        try:
            mesh_file = mesh_info.file_path
            if not Path(mesh_file).exists():
                return []
            
            mesh_obj = trimesh.load(mesh_file)
            if not isinstance(mesh_obj, trimesh.Trimesh):
                return []
            
            # 获取质心
            centroid = mesh_obj.centroid
            
            # 计算主惯性轴（Principal Axes of Inertia）
            # 使用trimesh的principal_inertia_vectors
            try:
                principal_vectors = mesh_obj.principal_inertia_vectors
                if principal_vectors is not None and len(principal_vectors) >= 3:
                    # Convert centroid to tuple of floats (not numpy types)
                    centroid_tuple = tuple(float(x) for x in centroid)
                    centroid_axes = [
                        {
                            "axis_id": "centroid_axis_1",
                            "axis": "principal_1",
                            "point": centroid_tuple,
                            "direction": principal_vectors[0].tolist(),
                            "description": "通过质心的主惯性轴1"
                        },
                        {
                            "axis_id": "centroid_axis_2",
                            "axis": "principal_2",
                            "point": centroid_tuple,
                            "direction": principal_vectors[1].tolist(),
                            "description": "通过质心的主惯性轴2"
                        },
                        {
                            "axis_id": "centroid_axis_3",
                            "axis": "principal_3",
                            "point": centroid_tuple,
                            "direction": principal_vectors[2].tolist(),
                            "description": "通过质心的主惯性轴3"
                        }
                    ]
                    return centroid_axes
            except:
                pass
            
            # 如果无法获取主惯性轴，使用AABB中心线但通过质心
            # Convert centroid to tuple of floats (not numpy types)
            centroid_tuple = tuple(float(x) for x in centroid)
            return [
                {
                    "axis_id": "centroid_x_axis",
                    "axis": "x",
                    "point": centroid_tuple,
                    "direction": [1.0, 0.0, 0.0],
                    "description": "通过质心的X轴中心线"
                },
                {
                    "axis_id": "centroid_y_axis",
                    "axis": "y",
                    "point": centroid_tuple,
                    "direction": [0.0, 1.0, 0.0],
                    "description": "通过质心的Y轴中心线"
                },
                {
                    "axis_id": "centroid_z_axis",
                    "axis": "z",
                    "point": centroid_tuple,
                    "direction": [0.0, 0.0, 1.0],
                    "description": "通过质心的Z轴中心线"
                }
            ]
            
        except Exception as e:
            return []
    
    def _extract_diagonal_centerlines(self, mesh_info) -> List[Dict[str, Any]]:
        """提取对角中心线"""
        aabb = mesh_info.aabb
        keypoints = mesh_info.keypoints
        
        # 计算对角向量
        corners = keypoints.corners
        min_corner = corners["min_min_min"]
        max_corner = corners["max_max_max"]
        
        # 主对角线
        diagonal_vector = np.array(max_corner) - np.array(min_corner)
        diagonal_length = np.linalg.norm(diagonal_vector)
        if diagonal_length > 1e-6:
            diagonal_direction = (diagonal_vector / diagonal_length).tolist()
        else:
            diagonal_direction = [1.0, 1.0, 1.0]
        
        center = mesh_info.aabb.center
        
        diagonal_axes = [
            {
                "axis_id": "diagonal_main",
                "axis": "diagonal",
                "point": tuple(center),
                "direction": diagonal_direction,
                "length": float(diagonal_length),
                "description": "主对角线中心线"
            }
        ]
        
        return diagonal_axes


class AnalyzeSlidingDirectionTool(Tool):
    """分析滑动方向工具
    
    分析mesh的几何结构，找到所有可能的滑动方向。
    滑动方向可以是：
    - 沿着3个主轴的方向（±X, ±Y, ±Z）
    - 沿着面的法线方向
    - 沿着其他可能的滑动轨迹
    """
    
    def __init__(self):
        super().__init__(
            name="analyze_sliding_direction",
            description="分析mesh的所有可能的滑动方向"
        )
    
    def execute(
        self,
        xml_path: str,
        part_name: str,
        include_face_normals: bool = True,
        **kwargs
    ) -> ToolResult:
        """执行滑动方向分析
        
        Args:
            xml_path: XML文件路径
            part_name: part名称（mesh名称）
            include_face_normals: 是否包含面的法线方向
            
        Returns:
            ToolResult包含所有可能的滑动方向信息
        """
        try:
            # 验证参数
            if not xml_path or not Path(xml_path).exists():
                return ToolResult(
                    success=False,
                    message=f"XML文件不存在: {xml_path}"
                )
            
            # 使用MeshAnalyzer加载mesh信息
            analyzer = MeshAnalyzer(xml_path)
            mesh_info_dict = analyzer.analyze()
            
            if part_name not in mesh_info_dict:
                return ToolResult(
                    success=False,
                    message=f"未找到part: {part_name}"
                )
            
            mesh_info = mesh_info_dict[part_name]
            
            # 1. 提取主轴方向（6个方向：±X, ±Y, ±Z）
            principal_directions = self._extract_principal_directions(mesh_info)
            
            # 2. 只使用主轴方向（不再使用面的法线方向）
            # 合并所有滑动方向
            all_directions = {
                "principal_directions": principal_directions,
                "face_normal_directions": [],  # 不再使用面法线方向
                "total_count": len(principal_directions)
            }
            
            return ToolResult(
                success=True,
                message=f"找到 {all_directions['total_count']} 个可能的滑动方向",
                data={
                    "part_name": part_name,
                    "sliding_directions": all_directions,
                    "mesh_info": mesh_info,  # 返回完整的MeshInfo对象，避免重复分析
                    "mesh_info_dict": mesh_info_dict  # 也返回完整的字典，方便后续使用
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"分析滑动方向失败: {str(e)}"
            )
    
    def _extract_principal_directions(self, mesh_info) -> List[Dict[str, Any]]:
        """提取主轴方向（6个方向：±X, ±Y, ±Z）"""
        aabb_size = mesh_info.aabb.size
        
        directions = [
            {
                "direction_id": "positive_x",
                "axis": "x",
                "direction": [1.0, 0.0, 0.0],
                "magnitude": float(aabb_size[0]),
                "description": "正X方向（向右）"
            },
            {
                "direction_id": "negative_x",
                "axis": "x",
                "direction": [-1.0, 0.0, 0.0],
                "magnitude": float(aabb_size[0]),
                "description": "负X方向（向左）"
            },
            {
                "direction_id": "positive_y",
                "axis": "y",
                "direction": [0.0, 1.0, 0.0],
                "magnitude": float(aabb_size[1]),
                "description": "正Y方向（向上）"
            },
            {
                "direction_id": "negative_y",
                "axis": "y",
                "direction": [0.0, -1.0, 0.0],
                "magnitude": float(aabb_size[1]),
                "description": "负Y方向（向下）"
            },
            {
                "direction_id": "positive_z",
                "axis": "z",
                "direction": [0.0, 0.0, 1.0],
                "magnitude": float(aabb_size[2]),
                "description": "正Z方向（向前）"
            },
            {
                "direction_id": "negative_z",
                "axis": "z",
                "direction": [0.0, 0.0, -1.0],
                "magnitude": float(aabb_size[2]),
                "description": "负Z方向（向后）"
            }
        ]
        
        return directions
    
    def _extract_face_normal_directions(self, mesh_info) -> List[Dict[str, Any]]:
        """提取面的法线方向"""
        try:
            mesh_file = mesh_info.file_path
            if not Path(mesh_file).exists():
                return []
            
            mesh_obj = trimesh.load(mesh_file)
            if not isinstance(mesh_obj, trimesh.Trimesh):
                return []
            
            # 获取面的法线
            face_normals = mesh_obj.face_normals
            
            # 统计主要的法线方向（使用聚类或统计）
            # 这里简化处理：提取不同的法线方向
            unique_normals = {}
            for i, normal in enumerate(face_normals[:50]):  # 限制数量
                normal_normalized = normal / np.linalg.norm(normal)
                normal_key = tuple(np.round(normal_normalized, 2))
                
                if normal_key not in unique_normals:
                    unique_normals[normal_key] = {
                        "direction_id": f"face_normal_{len(unique_normals)}",
                        "direction": normal_normalized.tolist(),
                        "face_count": 1,
                        "description": f"面法线方向 {len(unique_normals)}"
                    }
                else:
                    unique_normals[normal_key]["face_count"] += 1
            
            return list(unique_normals.values())
            
        except Exception as e:
            return []


class FindPartEdgesTool(Tool):
    """查找part的合适棱（edge）工具
    
    分析mesh中某个part的几何特征，找到合适的棱作为旋转轴候选。
    这个工具专门用于分析单个part的edge特征，不依赖于housing。
    找到的edge可以用于edge转动轴的分析。
    """
    
    def __init__(self):
        super().__init__(
            name="find_part_edges",
            description="查找mesh中某个part合适的棱（edge），用于edge转动轴分析"
        )
    
    def execute(
        self,
        xml_path: str,
        part_name: str,
        max_candidates: int = 15,
        min_length_ratio: float = 0.1,
        alignment_threshold: float = 0.7,
        visualize: bool = True,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """执行part edge查找
        
        Args:
            xml_path: XML文件路径
            part_name: part名称（mesh名称）
            max_candidates: 最多返回的候选边数量
            min_length_ratio: 最小长度比例（相对于AABB最大尺寸）
            alignment_threshold: 主轴对齐阈值（0-1，越高要求越严格）
            visualize: 是否生成3D可视化
            output_dir: 可视化输出目录（如果为None，使用XML文件所在目录）
            
        Returns:
            ToolResult包含所有可能的edge信息
        """
        try:
            # 验证参数
            if not xml_path or not Path(xml_path).exists():
                return ToolResult(
                    success=False,
                    message=f"XML文件不存在: {xml_path}"
                )
            
            # 使用MeshAnalyzer加载mesh信息
            analyzer = MeshAnalyzer(xml_path)
            mesh_info_dict = analyzer.analyze()
            
            if part_name not in mesh_info_dict:
                return ToolResult(
                    success=False,
                    message=f"未找到part: {part_name}"
                )
            
            part_mesh_info = mesh_info_dict[part_name]
            
            # 提取合适的edge
            candidate_edges = self._extract_suitable_edges(
                part_mesh_info,
                max_candidates=max_candidates,
                min_length_ratio=min_length_ratio,
                alignment_threshold=alignment_threshold
            )
            
            print(f"✓ 找到 {len(candidate_edges)} 条候选edge")
            
            # 准备返回数据
            result_data = {
                "part_name": part_name,
                "edges": candidate_edges,
                "total_count": len(candidate_edges),
                "mesh_info": {
                    "aabb": {
                        "min": part_mesh_info.aabb.minimum,
                        "max": part_mesh_info.aabb.maximum,
                        "size": part_mesh_info.aabb.size,
                        "center": part_mesh_info.aabb.center
                    }
                },
                "mesh_info_dict": mesh_info_dict  # 返回完整的mesh_info_dict
            }
            
            # 如果启用可视化，生成3D可视化
            visualization_path = None
            if visualize:
                visualization_path = self._visualize_edges(
                    part_mesh_info,
                    candidate_edges,
                    output_dir=output_dir or str(Path(xml_path).parent),
                    part_name=part_name
                )
                if visualization_path:
                    result_data["visualization_path"] = visualization_path
            
            return ToolResult(
                success=True,
                message=f"找到 {len(candidate_edges)} 条合适的edge",
                data=result_data
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(
                success=False,
                message=f"查找part edge失败: {str(e)}"
            )
    
    def _extract_suitable_edges(
        self,
        mesh_info: Any,
        max_candidates: int = 15,
        min_length_ratio: float = 0.1,
        alignment_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """提取合适的edge
        
        Args:
            mesh_info: mesh信息对象
            max_candidates: 最多返回的候选边数量
            min_length_ratio: 最小长度比例（相对于AABB最大尺寸）
            alignment_threshold: 主轴对齐阈值
            
        Returns:
            候选边列表
        """
        try:
            mesh_file = mesh_info.file_path
            if not Path(mesh_file).exists():
                return []
            
            mesh_obj = trimesh.load(mesh_file)
            if not isinstance(mesh_obj, trimesh.Trimesh):
                # 如果是Scene，尝试合并
                if isinstance(mesh_obj, trimesh.Scene):
                    mesh_obj = trimesh.util.concatenate([m for m in mesh_obj.geometry.values() 
                                                         if isinstance(m, trimesh.Trimesh)])
                else:
                    return []
            
            # 获取AABB信息
            aabb_min = np.array(mesh_info.aabb.minimum)
            aabb_max = np.array(mesh_info.aabb.maximum)
            aabb_size = np.array(mesh_info.aabb.size)
            max_size = max(aabb_size)
            min_length = max_size * min_length_ratio
            
            # 提取所有唯一的边
            edges = mesh_obj.edges_unique
            vertices = mesh_obj.vertices
            
            # 计算每条边的信息
            edge_info_list = []
            for edge in edges:
                v1 = np.array(vertices[edge[0]])
                v2 = np.array(vertices[edge[1]])
                
                # 计算边的向量和长度
                edge_vector = v2 - v1
                length = np.linalg.norm(edge_vector)
                
                # 过滤太短的边
                if length < min_length:
                    continue
                
                direction = edge_vector / length
                midpoint = (v1 + v2) / 2
                
                # 计算边与主轴的对齐程度
                abs_direction = np.abs(direction)
                max_align_idx = np.argmax(abs_direction)
                alignment_score = abs_direction[max_align_idx]
                alignment_axis = ["x", "y", "z"][max_align_idx]
                is_axis_aligned = alignment_score >= alignment_threshold
                
                # 判断边是否在边界上
                is_on_boundary = False
                boundary_type = None
                
                # 计算到各个面的距离
                dist_to_faces = {
                    "x_min": abs(midpoint[0] - aabb_min[0]),
                    "x_max": abs(midpoint[0] - aabb_max[0]),
                    "y_min": abs(midpoint[1] - aabb_min[1]),
                    "y_max": abs(midpoint[1] - aabb_max[1]),
                    "z_min": abs(midpoint[2] - aabb_min[2]),
                    "z_max": abs(midpoint[2] - aabb_max[2]),
                }
                
                # 改进边界检测：使用更精确的阈值
                # 如果边的中点距离某个AABB面很近，且边的方向与该面垂直，认为在边界上
                threshold = min(aabb_size) * 0.03  # 降低阈值，更严格
                for face_name, dist in dist_to_faces.items():
                    if dist < threshold:
                        # 检查边的方向是否与该面垂直
                        # 例如：如果边界是x_min或x_max，边应该沿Y或Z方向
                        face_axis = face_name.split("_")[0]  # "x", "y", "z"
                        if alignment_axis != face_axis:  # 边的方向与面的法线垂直
                            is_on_boundary = True
                            boundary_type = face_name
                            break
                
                # 计算边的"重要性"分数
                # 综合考虑：对齐度（最重要）、长度、是否在边界上
                length_score = length / max_size  # 归一化长度分数
                
                # 对齐度权重提高（从40%提高到50%），因为对齐度是最重要的特征
                # 对于旋转部件，垂直边（Z轴）应该优先
                axis_bonus = 0.0
                if alignment_axis == "z" and alignment_score > 0.8:
                    # 垂直边（Z轴）且对齐度高，给予额外奖励（适合旋转门）
                    axis_bonus = 0.1
                elif alignment_score > 0.9:
                    # 非常高的对齐度，给予奖励
                    axis_bonus = 0.05
                
                importance_score = (
                    alignment_score * 0.5 +  # 对齐度占比50%（提高）
                    length_score * 0.3 +  # 长度占比30%（降低）
                    (1.0 if is_on_boundary else 0.0) * 0.2 +  # 边界占比20%
                    axis_bonus  # 轴奖励
                )
                
                edge_info_list.append({
                    "edge": edge,
                    "v1": v1,
                    "v2": v2,
                    "midpoint": tuple(float(x) for x in midpoint),  # 确保是Python原生类型
                    "direction": direction.tolist(),
                    "length": float(length),
                    "alignment_axis": alignment_axis,
                    "alignment_score": float(alignment_score),
                    "is_axis_aligned": bool(is_axis_aligned),
                    "is_on_boundary": bool(is_on_boundary),
                    "boundary_type": boundary_type,
                    "importance_score": float(importance_score),
                    "length_score": float(length_score)
                })
            
            # 按重要性分数排序
            edge_info_list.sort(key=lambda x: x["importance_score"], reverse=True)
            
            # 按轴分组，确保不同轴方向的边都能被保留
            # 优先考虑垂直轴（Z轴），因为大多数旋转部件（如门）使用垂直轴
            edges_by_axis = {"x": [], "y": [], "z": []}
            for e in edge_info_list:
                if e["is_axis_aligned"]:
                    edges_by_axis[e["alignment_axis"]].append(e)
            
            # 从每个轴方向选择最重要的几条边
            # 对于旋转部件，优先选择垂直边（Z轴）
            unique_candidates = []
            edges_per_axis = max(2, max_candidates // 3)  # 每个轴至少保留2条
            
            # 优先处理Z轴（垂直边），然后是X和Y轴
            axis_priority = ["z", "x", "y"]  # Z轴优先（适合旋转门）
            
            for axis in axis_priority:
                axis_edges = edges_by_axis[axis]
                if not axis_edges:
                    continue
                
                # 去重：如果有多条边在相似位置，只保留一条
                axis_unique = []
                for edge_info in axis_edges:
                    is_duplicate = False
                    for existing in axis_unique:
                        midpoint_dist = np.linalg.norm(
                            np.array(edge_info["midpoint"]) - np.array(existing["midpoint"])
                        )
                        direction_similarity = np.abs(np.dot(
                            np.array(edge_info["direction"]),
                            np.array(existing["direction"])
                        ))
                        
                        # 如果中点距离很近且方向非常相似，认为是重复的
                        if midpoint_dist < max_size * 0.05 and direction_similarity > 0.95:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        axis_unique.append(edge_info)
                    
                    if len(axis_unique) >= edges_per_axis:
                        break
                
                unique_candidates.extend(axis_unique)
            
            # 如果还没有足够的候选边，从剩余边中补充（按重要性排序）
            if len(unique_candidates) < max_candidates:
                remaining_edges = []
                for edge_info in edge_info_list:
                    # 检查是否已经在unique_candidates中
                    already_included = False
                    for existing in unique_candidates:
                        if (np.allclose(edge_info["midpoint"], existing["midpoint"], atol=max_size * 0.01) and
                            np.abs(np.dot(edge_info["direction"], existing["direction"])) > 0.95):
                            already_included = True
                            break
                    
                    if not already_included:
                        remaining_edges.append(edge_info)
                
                # 按重要性排序，补充到max_candidates
                remaining_edges.sort(key=lambda x: x["importance_score"], reverse=True)
                for edge_info in remaining_edges:
                    if len(unique_candidates) >= max_candidates:
                        break
                    unique_candidates.append(edge_info)
            
            # 最终按重要性排序
            unique_candidates.sort(key=lambda x: x["importance_score"], reverse=True)
            
            # 转换为输出格式
            candidate_edges = []
            for i, edge_info in enumerate(unique_candidates):
                p1, p2 = edge_info["v1"], edge_info["v2"]
                
                description_parts = [
                    f"{edge_info['alignment_axis'].upper()}轴方向",
                    f"长度{edge_info['length']:.3f}",
                    f"对齐度{edge_info['alignment_score']:.2f}"
                ]
                if edge_info["is_on_boundary"]:
                    description_parts.append(f"边界({edge_info['boundary_type']})")
                description_parts.append(f"重要性{edge_info['importance_score']:.2f}")
                
                # 确保midpoint是Python原生类型（不是numpy类型）
                midpoint_tuple = edge_info["midpoint"]
                midpoint_clean = tuple(float(x) for x in midpoint_tuple)
                
                candidate_edges.append({
                    "edge_id": f"part_edge_{i}",
                    "midpoint": midpoint_clean,
                    "direction": edge_info["direction"],
                    "length": float(edge_info["length"]),
                    "alignment_axis": edge_info["alignment_axis"],
                    "alignment_score": float(edge_info["alignment_score"]),
                    "is_axis_aligned": bool(edge_info["is_axis_aligned"]),
                    "is_on_boundary": bool(edge_info["is_on_boundary"]),
                    "boundary_type": edge_info.get("boundary_type"),
                    "importance_score": float(edge_info["importance_score"]),
                    "length_score": float(edge_info["length_score"]),
                    "edge_vertices": [p1.tolist(), p2.tolist()],
                    "description": f"Part edge {i}（{', '.join(description_parts)}）"
                })
            
            return candidate_edges
            
        except Exception as e:
            print(f"提取edge失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _visualize_edges(
        self,
        mesh_info: Any,
        edges: List[Dict[str, Any]],
        output_dir: str,
        part_name: str
    ) -> Optional[str]:
        """生成3D可视化
        
        Args:
            mesh_info: mesh信息对象
            edges: edge列表
            output_dir: 输出目录
            part_name: part名称
            
        Returns:
            可视化文件路径，如果失败则返回None
        """
        try:
            # 设置非交互式后端，避免在多线程环境中创建GUI窗口（macOS要求）
            import matplotlib
            matplotlib.use('Agg')  # 必须在导入pyplot之前设置
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.patches import Rectangle
        except ImportError:
            print("✗ 缺少可视化依赖（matplotlib），跳过可视化")
            return None
        
        try:
            # 加载mesh
            mesh_file = mesh_info.file_path
            mesh_obj = trimesh.load(mesh_file)
            if not isinstance(mesh_obj, trimesh.Trimesh):
                if isinstance(mesh_obj, trimesh.Scene):
                    mesh_obj = trimesh.util.concatenate([m for m in mesh_obj.geometry.values() 
                                                         if isinstance(m, trimesh.Trimesh)])
                else:
                    print(f"无法加载mesh: {mesh_file}")
                    return None
            
            # 生成颜色映射（带序号）
            colors = self._generate_distinct_colors(len(edges))
            color_mapping = {}
            for i, edge_info in enumerate(edges):
                r, g, b = colors[i]
                color_hex = self._rgb_to_hex(r, g, b)
                sequence_number = i + 1  # 序号从1开始
                color_mapping[color_hex] = {
                    "edge_id": edge_info["edge_id"],
                    "rgb": [r, g, b],
                    "hex": color_hex,
                    "index": i,
                    "sequence_number": sequence_number,
                    **edge_info
                }
            
            # 创建图形（移除右侧图例，只显示3D图）
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制mesh
            ax.plot_trisurf(
                mesh_obj.vertices[:, 0],
                mesh_obj.vertices[:, 1],
                mesh_obj.vertices[:, 2],
                triangles=mesh_obj.faces,
                alpha=0.15,
                color='lightblue',
                edgecolor='none'
            )
            
            # 绘制edge
            for edge in edges:
                edge_id = edge["edge_id"]
                
                # 查找颜色
                color_info = None
                for hex_color, info in color_mapping.items():
                    if info["edge_id"] == edge_id:
                        color_info = info
                        break
                
                if not color_info:
                    continue
                
                midpoint = np.array(edge["midpoint"])
                direction = np.array(edge["direction"])
                length = edge["length"]
                
                # 计算edge的起点和终点
                start = midpoint - direction * length / 2
                end = midpoint + direction * length / 2
                
                # 获取颜色和序号
                r, g, b = color_info["rgb"]
                sequence_number = color_info.get("sequence_number", color_info.get("index", 0) + 1)
                
                # 绘制edge
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=(r, g, b),
                    linewidth=3.5,
                    alpha=0.9,
                    linestyle='-',
                    label=f"Edge {sequence_number}"
                )
                
                # 在edge端点标注序号（使用edge的颜色，避免遮挡）
                # 选择edge的一个端点作为标注位置，避免在中点重叠
                use_start = (sequence_number % 2 == 0)
                
                if use_start:
                    # 使用起点，并向外偏移
                    base_pos = start
                    offset_dir = -direction  # 向外偏移
                else:
                    # 使用终点，并向外偏移
                    base_pos = end
                    offset_dir = direction  # 向外偏移
                
                # 计算偏移量（垂直于edge方向，避免遮挡edge本身）
                if abs(direction[2]) < 0.9:  # 不是纯垂直的edge
                    perp_vec = np.cross(direction, [0, 0, 1])
                    if np.linalg.norm(perp_vec) < 0.1:  # 如果平行，使用Y轴
                        perp_vec = np.cross(direction, [0, 1, 0])
                else:  # 垂直edge，使用X轴
                    perp_vec = np.cross(direction, [1, 0, 0])
                
                perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-10)  # 归一化
                offset_distance = length * 0.08  # 偏移距离
                text_pos = base_pos + perp_vec * offset_distance + offset_dir * (length * 0.05)
                
                ax.text(
                    text_pos[0], text_pos[1], text_pos[2],
                    str(sequence_number),
                    fontsize=14,
                    fontweight='bold',
                    color=(r, g, b),  # 使用edge的颜色
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=(r, g, b), linewidth=2.0)
                )
            
            # 设置等比例
            aabb = mesh_info.aabb
            aabb_size = np.array(aabb.size)
            max_range = max(aabb_size) / 2.0
            center = np.array(aabb.center)
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            # 设置坐标轴标签（英文）
            ax.set_xlabel('X', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z', fontsize=12, fontweight='bold')
            ax.set_title(f"{part_name} - Part Edges Visualization", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图像
            output_path = Path(output_dir) / f"{part_name}_part_edges_visualization.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存到: {output_path}")
            
            plt.close(fig)
            return str(output_path)
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """生成n个不同的颜色，使用更多、更易区分的颜色方案"""
        # 扩展的基础调色板，包含更多易区分的颜色
        base_palette = [
            (1.0, 0.0, 0.0),      # Red - 红色
            (0.0, 1.0, 0.0),      # Green - 绿色
            (0.0, 0.0, 1.0),      # Blue - 蓝色
            (1.0, 1.0, 0.0),      # Yellow - 黄色
            (1.0, 0.0, 1.0),      # Magenta - 洋红
            (0.0, 1.0, 1.0),      # Cyan - 青色
            (1.0, 0.5, 0.0),      # Orange - 橙色
            (0.5, 0.0, 1.0),      # Purple - 紫色
            (1.0, 0.75, 0.8),     # Pink - 粉色
            (0.0, 0.5, 0.5),      # Teal - 青绿色
            (0.5, 1.0, 0.5),      # Light Green - 浅绿色
            (1.0, 0.5, 0.5),      # Light Red - 浅红色
            (0.5, 0.5, 1.0),      # Light Blue - 浅蓝色
            (1.0, 0.8, 0.0),      # Gold - 金色
            (0.8, 0.2, 0.8),      # Violet - 紫罗兰色
            (0.2, 0.8, 0.8),      # Turquoise - 青蓝色
            (0.8, 0.8, 0.2),      # Olive - 橄榄色
            (1.0, 0.4, 0.4),      # Coral - 珊瑚色
            (0.4, 1.0, 0.4),      # Lime - 酸橙绿
            (0.4, 0.4, 1.0),      # Sky Blue - 天蓝色
            (1.0, 0.6, 0.2),      # Dark Orange - 深橙色
            (0.6, 0.2, 1.0),      # Indigo - 靛蓝色
            (0.2, 1.0, 0.6),      # Spring Green - 春绿色
            (1.0, 0.2, 0.6),      # Hot Pink - 热粉色
            (0.2, 0.6, 1.0),      # Royal Blue - 皇家蓝
            (1.0, 0.8, 0.4),      # Peach - 桃色
            (0.8, 0.4, 1.0),      # Lavender - 薰衣草色
            (0.4, 1.0, 0.8),      # Aqua - 水色
            (1.0, 0.4, 0.8),      # Rose - 玫瑰色
            (0.4, 0.8, 1.0),      # Light Sky Blue - 浅天蓝色
        ]
        
        if n <= len(base_palette):
            return base_palette[:n]
        
        colors = list(base_palette)
        remain = n - len(base_palette)
        # 使用HSL生成额外颜色，确保高饱和度和中等亮度以便区分
        for k in range(remain):
            # 使用黄金角度（约137.5度）来最大化颜色差异
            golden_angle = 0.618033988749895  # 黄金比例
            h = (k * golden_angle) % 1.0
            s = 0.9  # 高饱和度
            l = 0.5  # 中等亮度
            r, g, b = self._hsl_to_rgb(h, s, l)
            colors.append((r, g, b))
        return colors
    
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> Tuple[float, float, float]:
        """HSL转RGB"""
        def hue_to_rgb(p: float, q: float, t: float) -> float:
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        if s == 0:
            return l, l, l
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        return r, g, b
    
    def _rgb_to_hex(self, r: float, g: float, b: float) -> str:
        """RGB转HEX"""
        r_int = int(round(r * 255))
        g_int = int(round(g * 255))
        b_int = int(round(b * 255))
        return f"#{r_int:02X}{g_int:02X}{b_int:02X}"


class AnalyzeMotionTypeTool(Tool):
    """分析运动类型工具
    
    根据输入的运动类型（edge旋转、中心线旋转、滑动），自动选择相应的工具进行分析，
    并生成带颜色轴的3D可视化图像，返回每个颜色对应的轴方向信息。
    """
    
    # 运动类型映射
    MOTION_TYPES = {
        "edge_rotation": "edge旋转",
        "centerline_rotation": "中心线旋转",
        "sliding": "滑动",
        "edge": "edge旋转",
        "centerline": "中心线旋转",
        "slide": "滑动"
    }
    
    def __init__(self):
        super().__init__(
            name="analyze_motion_type",
            description="根据运动类型分析part的运动轴或方向，生成带颜色轴的3D可视化"
        )
        # 初始化子工具
        self.edge_tool = FindPartEdgesTool()
        self.centerline_tool = FindCenterlineAxesTool()
        self.sliding_tool = AnalyzeSlidingDirectionTool()
    
    def execute(
        self,
        xml_path: str,
        part_name: str,
        motion_type: str,
        visualize: bool = True,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """执行运动类型分析
        
        Args:
            xml_path: XML文件路径
            part_name: part名称（mesh名称）
            motion_type: 运动类型，可选值：
                - "edge_rotation" 或 "edge": edge旋转
                - "centerline_rotation" 或 "centerline": 中心线旋转
                - "sliding" 或 "slide": 滑动
            visualize: 是否生成3D可视化
            output_dir: 可视化输出目录（如果为None，使用XML文件所在目录）
            **kwargs: 其他参数，会传递给相应的子工具
            
        Returns:
            ToolResult包含：
            - motion_type: 运动类型
            - axes/directions: 轴或方向列表
            - color_mapping: 颜色映射（hex -> axis_info）
            - visualization_path: 可视化图像路径
        """
        try:
            # 验证参数
            if not xml_path or not Path(xml_path).exists():
                return ToolResult(
                    success=False,
                    message=f"XML文件不存在: {xml_path}"
                )
            
            # 规范化运动类型
            motion_type_lower = motion_type.lower().strip()
            if motion_type_lower not in self.MOTION_TYPES:
                return ToolResult(
                    success=False,
                    message=f"不支持的运动类型: {motion_type}。支持的类型: {', '.join(set(self.MOTION_TYPES.keys()))}"
                )
            
            normalized_motion_type = self.MOTION_TYPES[motion_type_lower]
            print(f"分析运动类型: {normalized_motion_type}")
            
            # 根据运动类型调用相应的工具
            if normalized_motion_type == "edge旋转":
                return self._analyze_edge_rotation(
                    xml_path, part_name, visualize, output_dir, **kwargs
                )
            elif normalized_motion_type == "中心线旋转":
                return self._analyze_centerline_rotation(
                    xml_path, part_name, visualize, output_dir, **kwargs
                )
            elif normalized_motion_type == "滑动":
                return self._analyze_sliding(
                    xml_path, part_name, visualize, output_dir, **kwargs
                )
            else:
                return ToolResult(
                    success=False,
                    message=f"未知的运动类型: {normalized_motion_type}"
                )
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(
                success=False,
                message=f"分析运动类型失败: {str(e)}"
            )
    
    def _analyze_edge_rotation(
        self,
        xml_path: str,
        part_name: str,
        visualize: bool,
        output_dir: Optional[str],
        **kwargs
    ) -> ToolResult:
        """分析edge旋转"""
        # 调用FindPartEdgesTool
        result = self.edge_tool.execute(
            xml_path=xml_path,
            part_name=part_name,
            visualize=visualize,
            output_dir=output_dir,
            **kwargs
        )
        
        if not result.success:
            return result
        
        # 提取edges数据
        edges_raw = result.data["edges"]
        
        # 获取part功能信息（如果可能）
        part_function = kwargs.get("part_function")
        
        # 生成颜色映射和序号映射
        color_mapping, index_mapping = self._create_edge_color_mapping(edges_raw, part_function)
        
        # 创建使用序号ID的edge列表（用于LLM选择）
        edges = []
        for i, edge_info in enumerate(edges_raw):
            sequence_number = i + 1
            new_edge_id = f"part_edge_{sequence_number}"
            # 创建新的edge信息，使用序号ID
            updated_edge = edge_info.copy()
            updated_edge["edge_id"] = new_edge_id  # 使用序号ID
            updated_edge["original_edge_id"] = edge_info.get("edge_id")  # 保留原始ID
            # 从index_mapping获取语义信息
            if sequence_number in index_mapping:
                updated_edge["semantic_info"] = index_mapping[sequence_number].get("semantic_info", "")
            edges.append(updated_edge)
        
        # 如果还没有可视化，生成一个
        if visualize and not result.data.get("visualization_path"):
            visualization_path = self._visualize_edges_unified(
                xml_path, part_name, edges_raw, color_mapping, output_dir
            )
        else:
            visualization_path = result.data.get("visualization_path")
        
        # 获取mesh_info_dict（从edge_tool的结果中）
        mesh_info_dict = result.data.get("mesh_info_dict")
        
        data = {
            "motion_type": "edge_rotation",
            "motion_type_cn": "edge旋转",
            "axes": edges,  # 使用序号ID的edge列表
            "color_mapping": color_mapping,
            "index_mapping": index_mapping,  # 新增：序号映射
            "visualization_path": visualization_path,
            "total_count": len(edges)
        }
        if mesh_info_dict:
            data["mesh_info_dict"] = mesh_info_dict
        
        return ToolResult(
            success=True,
            message=f"找到 {len(edges)} 条edge旋转轴候选",
            data=data
        )
    
    def _analyze_centerline_rotation(
        self,
        xml_path: str,
        part_name: str,
        visualize: bool,
        output_dir: Optional[str],
        **kwargs
    ) -> ToolResult:
        """分析中心线旋转"""
        # 调用FindCenterlineAxesTool
        result = self.centerline_tool.execute(
            xml_path=xml_path,
            part_name=part_name,
            **kwargs
        )
        
        if not result.success:
            return result
        
        # 提取centerlines数据
        centerlines_data = result.data["centerlines"]
        part_function = kwargs.get("part_function")
        
        # 收集所有中心线（原始数据）
        all_centerlines_raw = []
        all_centerlines_raw.extend(centerlines_data.get("principal_axes", []))
        all_centerlines_raw.extend(centerlines_data.get("centroid_axes", []))
        all_centerlines_raw.extend(centerlines_data.get("diagonal_axes", []))
        
        # 生成颜色映射和序号映射
        color_mapping, index_mapping = self._create_centerline_color_mapping(centerlines_data, part_function)
        
        # 创建使用序号ID的中心线列表（用于LLM选择）
        all_centerlines = []
        for i, axis_info in enumerate(all_centerlines_raw):
            sequence_number = i + 1
            new_axis_id = f"part_centerline_{sequence_number}"
            # 创建新的轴信息，使用序号ID
            updated_axis = axis_info.copy()
            updated_axis["axis_id"] = new_axis_id  # 使用序号ID
            updated_axis["original_axis_id"] = axis_info.get("axis_id")  # 保留原始ID
            # 从index_mapping获取语义信息
            if sequence_number in index_mapping:
                updated_axis["semantic_info"] = index_mapping[sequence_number].get("semantic_info", "")
            all_centerlines.append(updated_axis)
        
        # 生成可视化
        visualization_path = None
        if visualize:
            visualization_path = self._visualize_centerlines_unified(
                xml_path, part_name, centerlines_data, color_mapping, output_dir
            )
        
        # 获取mesh_info_dict（从centerline_tool的结果中）
        mesh_info_dict = result.data.get("mesh_info_dict")
        
        data = {
            "motion_type": "centerline_rotation",
            "motion_type_cn": "中心线旋转",
            "axes": all_centerlines,  # 使用序号ID的中心线列表
            "centerlines": centerlines_data,
            "color_mapping": color_mapping,
            "index_mapping": index_mapping,  # 新增：序号映射
            "visualization_path": visualization_path,
            "total_count": len(all_centerlines)
        }
        if mesh_info_dict:
            data["mesh_info_dict"] = mesh_info_dict
        
        return ToolResult(
            success=True,
            message=f"找到 {len(all_centerlines)} 条中心线旋转轴候选",
            data=data
        )
    
    def _analyze_sliding(
        self,
        xml_path: str,
        part_name: str,
        visualize: bool,
        output_dir: Optional[str],
        **kwargs
    ) -> ToolResult:
        """分析滑动"""
        # 调用AnalyzeSlidingDirectionTool
        result = self.sliding_tool.execute(
            xml_path=xml_path,
            part_name=part_name,
            **kwargs
        )
        
        if not result.success:
            return result
        
        # 提取滑动方向数据
        directions_data = result.data["sliding_directions"]
        
        # 获取part功能信息（如果可能）
        part_function = kwargs.get("part_function")
        
        # 生成颜色映射和序号映射（只包含3个轴）
        color_mapping, index_mapping = self._create_sliding_color_mapping(directions_data, part_function)
        
        # 创建使用序号ID的轴列表（用于LLM选择，只包含3个轴）
        # 将6个方向（±X, ±Y, ±Z）合并为3个轴
        principal_directions = directions_data.get("principal_directions", [])
        
        # 按轴分组
        axes_dict = {}  # {axis_name: {"positive": {...}, "negative": {...}}}
        for direction_info in principal_directions:
            axis = direction_info.get("axis", "").lower()
            direction_id = direction_info.get("direction_id", "")
            
            if axis not in axes_dict:
                axes_dict[axis] = {}
            
            if "positive" in direction_id:
                axes_dict[axis]["positive"] = direction_info
            elif "negative" in direction_id:
                axes_dict[axis]["negative"] = direction_info
        
        # 创建轴列表（只包含3个轴）
        all_directions = []
        for axis_name in ["x", "y", "z"]:
            if axis_name not in axes_dict:
                continue
            
            axis_dirs = axes_dict[axis_name]
            positive_dir = axis_dirs.get("positive")
            
            if not positive_dir:
                continue
            
            # 从index_mapping获取对应的序号和语义信息
            sequence_number = None
            for seq_num, info in index_mapping.items():
                if info.get("axis") == axis_name.upper():
                    sequence_number = seq_num
                    break
            
            if sequence_number is None:
                continue
            
            # 创建轴信息（使用正方向作为代表，但说明是双向的）
            axis_info = positive_dir.copy()
            axis_info["direction_id"] = f"sliding_axis_{sequence_number}"  # 使用轴序号ID
            axis_info["reference_direction_id"] = positive_dir.get("direction_id")  # 保留原始ID
            axis_info["axis"] = axis_name
            axis_info["description"] = f"{axis_name.upper()}-axis (bidirectional sliding along ±{axis_name.upper()})"
            
            # 从index_mapping获取语义信息
            if sequence_number in index_mapping:
                axis_info["semantic_info"] = index_mapping[sequence_number].get("semantic_info", "")
                axis_info["description"] = index_mapping[sequence_number].get("description", axis_info["description"])
            
            all_directions.append(axis_info)
        
        # 生成可视化（使用更新后的方向列表和颜色映射）
        visualization_path = None
        if visualize:
            # 创建一个包含更新后方向的数据结构用于可视化
            # 使用原始directions_data，因为可视化函数需要原始结构
            visualization_path = self._visualize_sliding_unified(
                xml_path, part_name, directions_data, color_mapping, output_dir
            )
            
            # 确保可视化路径被正确设置
            if visualization_path:
                print(f"✓ 滑动方向可视化已生成: {visualization_path}")
            else:
                print(f"⚠ 滑动方向可视化生成失败")
        
        return ToolResult(
            success=True,
            message=f"找到 {len(all_directions)} 个滑动轴候选（每个轴支持双向滑动）",
            data={
                "motion_type": "sliding",
                "motion_type_cn": "滑动",
                "directions": all_directions,  # 使用序号ID的轴列表（只包含3个轴）
                "sliding_directions": directions_data,
                "color_mapping": color_mapping,
                "index_mapping": index_mapping,  # 新增：序号映射（只包含3个轴）
                "visualization_path": visualization_path,
                "total_count": len(all_directions)
            }
        )
    
    def _generate_edge_semantic_info(
        self,
        edge_info: Dict[str, Any],
        part_function: Optional[str] = None
    ) -> str:
        """为edge生成语义信息
        
        Args:
            edge_info: edge信息字典
            part_function: 部件功能描述（如"door"、"drawer"等）
            
        Returns:
            语义信息字符串
        """
        alignment_axis = edge_info.get("alignment_axis", "")
        boundary_type = edge_info.get("boundary_type", "")
        is_on_boundary = edge_info.get("is_on_boundary", False)
        alignment_score = edge_info.get("alignment_score", 0)
        importance_score = edge_info.get("importance_score", 0)
        length = edge_info.get("length", 0)
        
        semantic_parts = []
        
        # 根据alignment_axis和boundary_type生成语义信息
        if alignment_axis == "z":
            # 垂直边
            semantic_parts.append("vertical edge (Z-axis aligned)")
            if boundary_type:
                if "y_max" in boundary_type:
                    semantic_parts.append("at front face of cabinet")
                    if part_function and ("door" in part_function.lower() or "门" in part_function):
                        semantic_parts.append("typical hinge location for door opening outward")
                        semantic_parts.append("most common rotation axis for cabinet doors")
                elif "y_min" in boundary_type:
                    semantic_parts.append("at back face of cabinet")
                    semantic_parts.append("hinge location for inward-opening doors")
                elif "x_max" in boundary_type:
                    semantic_parts.append("at right side of cabinet")
                elif "x_min" in boundary_type:
                    semantic_parts.append("at left side of cabinet")
        elif alignment_axis == "x":
            # 水平X方向边
            semantic_parts.append("horizontal edge (X-axis aligned, left-right)")
            if boundary_type:
                if "z_max" in boundary_type:
                    semantic_parts.append("at top face")
                elif "z_min" in boundary_type:
                    semantic_parts.append("at bottom face")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("potential sliding rail location")
        elif alignment_axis == "y":
            # 水平Y方向边
            semantic_parts.append("horizontal edge (Y-axis aligned, front-back)")
            if boundary_type:
                if "z_max" in boundary_type:
                    semantic_parts.append("at top face")
                elif "z_min" in boundary_type:
                    semantic_parts.append("at bottom face")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("potential sliding rail location")
        
        # 添加边界位置信息
        if is_on_boundary:
            semantic_parts.append("located on AABB boundary")
            if boundary_type:
                semantic_parts.append(f"boundary type: {boundary_type}")
        
        # 添加对齐信息
        if alignment_score > 0.9:
            semantic_parts.append("excellent alignment with standard axis")
        elif alignment_score > 0.8:
            semantic_parts.append("well-aligned with standard axis")
        elif alignment_score > 0.7:
            semantic_parts.append("moderately aligned")
        else:
            semantic_parts.append("low alignment score")
        
        # 添加长度信息
        if length > 0:
            semantic_parts.append(f"length: {length:.3f}m")
            if length > 0.3:
                semantic_parts.append("long edge")
            elif length < 0.1:
                semantic_parts.append("short edge")
        
        # 添加重要性分数信息
        if importance_score > 0.8:
            semantic_parts.append("high importance score")
        elif importance_score > 0.6:
            semantic_parts.append("moderate importance")
        
        if not semantic_parts:
            return "edge candidate"
        
        return ", ".join(semantic_parts)
    
    def _create_edge_color_mapping(
        self, 
        edges: List[Dict[str, Any]], 
        part_function: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """为edge创建颜色映射和序号映射
        
        Args:
            edges: edge列表
            part_function: 部件功能描述
            
        Returns:
            (color_mapping, index_mapping) 颜色映射和序号映射（序号从1开始）
        """
        colors = self._generate_distinct_colors(len(edges))
        color_mapping = {}
        index_mapping = {}  # 序号映射：序号(从1开始) -> 轴信息
        
        for i, edge_info in enumerate(edges):
            r, g, b = colors[i]
            color_hex = self._rgb_to_hex(r, g, b)
            sequence_number = i + 1  # 序号从1开始
            
            # 生成语义信息
            semantic_info = self._generate_edge_semantic_info(edge_info, part_function)
            
            color_mapping[color_hex] = {
                "edge_id": edge_info["edge_id"],
                "rgb": [r, g, b],
                "hex": color_hex,
                "index": i,
                "sequence_number": sequence_number,  # 序号（从1开始）
                "midpoint": edge_info["midpoint"],
                "direction": edge_info["direction"],
                "length": edge_info["length"],
                "alignment_axis": edge_info["alignment_axis"],
                "alignment_score": edge_info["alignment_score"],
                "description": edge_info.get("description", ""),
                "semantic_info": semantic_info,  # 语义信息
                "axis_type": "edge"
            }
            
            # 创建序号映射
            index_mapping[sequence_number] = {
                "edge_id": edge_info["edge_id"],
                "direction": edge_info["direction"],
                "alignment_axis": edge_info["alignment_axis"],
                "alignment_score": edge_info["alignment_score"],
                "importance_score": edge_info.get("importance_score", 0),
                "length": edge_info.get("length", 0),
                "semantic_info": semantic_info,
                "description": edge_info.get("description", ""),
                "boundary_type": edge_info.get("boundary_type", ""),
                "is_on_boundary": edge_info.get("is_on_boundary", False)
            }
        
        return color_mapping, index_mapping
    
    def _generate_centerline_semantic_info(
        self,
        axis_info: Dict[str, Any],
        part_function: Optional[str] = None
    ) -> str:
        """为中心线生成语义信息
        
        Args:
            axis_info: 轴信息字典
            part_function: 部件功能描述
            
        Returns:
            语义信息字符串
        """
        axis_id = axis_info.get("axis_id", "")
        axis_type = "principal"
        if axis_id.startswith("centroid"):
            axis_type = "centroid"
        elif axis_id.startswith("diagonal"):
            axis_type = "diagonal"
        
        direction = np.array(axis_info.get("direction", [0, 0, 0]))
        point = axis_info.get("point", [0, 0, 0])
        
        semantic_parts = []
        
        # 根据轴类型生成语义
        if axis_type == "principal":
            semantic_parts.append("principal axis (main geometric axis)")
            # 判断方向
            if abs(direction[0]) > 0.9:
                semantic_parts.append("X-axis aligned (left-right)")
                if direction[0] > 0:
                    semantic_parts.append("rightward orientation")
                else:
                    semantic_parts.append("leftward orientation")
                if part_function and ("door" in part_function.lower() or "门" in part_function):
                    semantic_parts.append("horizontal rotation axis (uncommon for doors)")
            elif abs(direction[1]) > 0.9:
                semantic_parts.append("Y-axis aligned (front-back)")
                if direction[1] > 0:
                    semantic_parts.append("forward orientation (toward front face)")
                    if part_function and ("door" in part_function.lower() or "门" in part_function):
                        semantic_parts.append("typical opening direction for doors")
                        semantic_parts.append("outward opening motion")
                else:
                    semantic_parts.append("backward orientation")
            elif abs(direction[2]) > 0.9:
                semantic_parts.append("Z-axis aligned (vertical, up-down)")
                if direction[2] > 0:
                    semantic_parts.append("upward orientation")
                else:
                    semantic_parts.append("downward orientation")
                if part_function and ("door" in part_function.lower() or "门" in part_function):
                    semantic_parts.append("vertical rotation axis")
                    semantic_parts.append("most common rotation axis for cabinet doors")
        elif axis_type == "centroid":
            semantic_parts.append("centroid axis")
            semantic_parts.append("passes through geometric center of part")
            semantic_parts.append("center of mass alignment")
        elif axis_type == "diagonal":
            semantic_parts.append("diagonal axis")
            semantic_parts.append("diagonal orientation")
            semantic_parts.append("non-standard axis direction")
        
        # 添加位置信息
        if point:
            semantic_parts.append(f"passes through point: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        
        if not semantic_parts:
            return "centerline rotation axis"
        
        return ", ".join(semantic_parts)
    
    def _create_centerline_color_mapping(
        self, 
        centerlines: Dict[str, List[Dict[str, Any]]], 
        part_function: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """为中心线创建颜色映射和序号映射
        
        Args:
            centerlines: 中心线数据字典
            part_function: 部件功能描述
            
        Returns:
            (color_mapping, index_mapping) 颜色映射和序号映射（序号从1开始）
        """
        all_axes = []
        all_axes.extend(centerlines.get("principal_axes", []))
        all_axes.extend(centerlines.get("centroid_axes", []))
        all_axes.extend(centerlines.get("diagonal_axes", []))
        
        colors = self._generate_distinct_colors(len(all_axes))
        color_mapping = {}
        index_mapping = {}  # 序号映射：序号(从1开始) -> 轴信息
        
        for i, axis_info in enumerate(all_axes):
            r, g, b = colors[i]
            color_hex = self._rgb_to_hex(r, g, b)
            sequence_number = i + 1  # 序号从1开始
            
            # 确定轴类型
            axis_type = "principal"
            if axis_info.get("axis_id", "").startswith("centroid"):
                axis_type = "centroid"
            elif axis_info.get("axis_id", "").startswith("diagonal"):
                axis_type = "diagonal"
            
            # 生成语义信息
            semantic_info = self._generate_centerline_semantic_info(axis_info, part_function)
            
            color_mapping[color_hex] = {
                "axis_id": axis_info["axis_id"],
                "rgb": [r, g, b],
                "hex": color_hex,
                "index": i,
                "sequence_number": sequence_number,  # 序号（从1开始）
                "point": axis_info["point"],
                "direction": axis_info["direction"],
                "description": axis_info.get("description", ""),
                "axis_type": axis_type,
                "semantic_info": semantic_info  # 语义信息
            }
            
            # 创建序号映射
            index_mapping[sequence_number] = {
                "axis_id": axis_info["axis_id"],
                "direction": axis_info["direction"],
                "point": axis_info["point"],
                "axis_type": axis_type,
                "semantic_info": semantic_info,
                "description": axis_info.get("description", "")
            }
        
        return color_mapping, index_mapping
    
    def _generate_sliding_semantic_info(
        self,
        direction_info: Dict[str, Any],
        part_function: Optional[str] = None
    ) -> str:
        """为滑动方向生成语义信息
        
        Args:
            direction_info: 方向信息字典
            part_function: 部件功能描述
            
        Returns:
            语义信息字符串
        """
        direction_id = direction_info.get("direction_id", "")
        reference_direction_id = direction_info.get("reference_direction_id", direction_id)  # 使用原始ID
        axis = direction_info.get("axis", "")
        magnitude = direction_info.get("magnitude", 0)
        direction_vec = direction_info.get("direction", [0, 0, 0])
        
        semantic_parts = []
        
        # 根据方向ID生成语义（使用reference_direction_id，因为这是原始的语义ID）
        direction_id_for_semantic = reference_direction_id if reference_direction_id else direction_id
        
        # 方向类型判断
        if "principal" in direction_id_for_semantic:
            semantic_parts.append("principal direction (main sliding axis)")
        elif "face_normal" in direction_id_for_semantic:
            semantic_parts.append("face normal direction (perpendicular to cabinet face)")
        
        # 根据方向向量判断主要方向
        if "positive_y" in direction_id_for_semantic or (len(direction_vec) == 3 and abs(direction_vec[1]) > 0.7 and direction_vec[1] > 0):
            semantic_parts.append("forward direction (+Y axis)")
            semantic_parts.append("toward front face of cabinet")
            semantic_parts.append("outward opening/pull-out direction")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("typical drawer pull-out direction")
                semantic_parts.append("most common sliding direction for drawers")
                semantic_parts.append("allows drawer to extend forward from cabinet")
            elif part_function and ("door" in part_function.lower() or "门" in part_function):
                semantic_parts.append("sliding door opening direction (forward)")
        elif "negative_y" in direction_id_for_semantic or (len(direction_vec) == 3 and abs(direction_vec[1]) > 0.7 and direction_vec[1] < 0):
            semantic_parts.append("backward direction (-Y axis)")
            semantic_parts.append("toward back face of cabinet")
            semantic_parts.append("inward sliding direction")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("drawer push-in direction")
        elif "positive_x" in direction_id_for_semantic or (len(direction_vec) == 3 and abs(direction_vec[0]) > 0.7 and direction_vec[0] > 0):
            semantic_parts.append("rightward direction (+X axis)")
            semantic_parts.append("toward right side of cabinet")
            semantic_parts.append("lateral sliding direction")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("lateral drawer sliding (less common)")
                semantic_parts.append("side-opening drawer mechanism")
        elif "negative_x" in direction_id_for_semantic or (len(direction_vec) == 3 and abs(direction_vec[0]) > 0.7 and direction_vec[0] < 0):
            semantic_parts.append("leftward direction (-X axis)")
            semantic_parts.append("toward left side of cabinet")
            semantic_parts.append("lateral sliding direction")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("lateral drawer sliding (less common)")
                semantic_parts.append("side-opening drawer mechanism")
        elif "positive_z" in direction_id_for_semantic or (len(direction_vec) == 3 and abs(direction_vec[2]) > 0.7 and direction_vec[2] > 0):
            semantic_parts.append("upward direction (+Z axis)")
            semantic_parts.append("toward top of cabinet")
            semantic_parts.append("vertical sliding direction")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("vertical sliding (uncommon for drawers)")
                semantic_parts.append("typically used for vertical storage systems")
        elif "negative_z" in direction_id_for_semantic or (len(direction_vec) == 3 and abs(direction_vec[2]) > 0.7 and direction_vec[2] < 0):
            semantic_parts.append("downward direction (-Z axis)")
            semantic_parts.append("toward bottom of cabinet")
            semantic_parts.append("vertical sliding direction")
            if part_function and ("drawer" in part_function.lower() or "抽屉" in part_function):
                semantic_parts.append("vertical sliding (uncommon for drawers)")
        
        # 添加幅度信息
        if magnitude > 0:
            semantic_parts.append(f"magnitude: {magnitude:.3f} (relative to part size)")
            # 根据幅度判断重要性
            if magnitude > 0.5:
                semantic_parts.append("significant sliding distance")
            elif magnitude < 0.1:
                semantic_parts.append("limited sliding distance")
        
        # 添加轴信息
        if axis:
            semantic_parts.append(f"aligned with {axis.upper()}-axis")
            # 添加坐标系统说明
            if axis.lower() == "x":
                semantic_parts.append("left-right axis in MuJoCo coordinate system")
            elif axis.lower() == "y":
                semantic_parts.append("front-back axis in MuJoCo coordinate system")
            elif axis.lower() == "z":
                semantic_parts.append("up-down axis in MuJoCo coordinate system")
        
        # 添加方向向量的详细信息
        if len(direction_vec) == 3:
            vec_str = f"[{direction_vec[0]:.3f}, {direction_vec[1]:.3f}, {direction_vec[2]:.3f}]"
            semantic_parts.append(f"direction vector: {vec_str}")
        
        # 添加使用场景说明
        if part_function:
            if "drawer" in part_function.lower() or "抽屉" in part_function:
                semantic_parts.append("commonly used for drawer mechanisms")
                semantic_parts.append("enables smooth linear motion")
            elif "door" in part_function.lower() or "门" in part_function:
                semantic_parts.append("can be used for sliding doors")
                semantic_parts.append("provides linear opening/closing motion")
            elif "panel" in part_function.lower() or "面板" in part_function:
                semantic_parts.append("suitable for sliding panels")
        
        # 添加机械约束说明
        semantic_parts.append("requires guide rails or tracks for smooth motion")
        semantic_parts.append("typically constrained to single degree of freedom")
        
        if not semantic_parts:
            return f"sliding direction along {axis} axis"
        
        return ", ".join(semantic_parts)
    
    def _create_sliding_color_mapping(
        self, 
        directions: Dict[str, List[Dict[str, Any]]], 
        part_function: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """为滑动方向创建颜色映射和序号映射
        
        Args:
            directions: 方向数据字典
            part_function: 部件功能描述
            
        Returns:
            (color_mapping, index_mapping) 颜色映射和序号映射（序号从1开始，只包含3个轴）
        """
        # 只使用主轴方向，将6个方向（±X, ±Y, ±Z）合并为3个轴
        principal_directions = directions.get("principal_directions", [])
        
        # 按轴分组：每个轴只生成一个序号映射
        axes_dict = {}  # {axis_name: {"positive": {...}, "negative": {...}}}
        for direction_info in principal_directions:
            axis = direction_info.get("axis", "").lower()
            direction_id = direction_info.get("direction_id", "")
            
            if axis not in axes_dict:
                axes_dict[axis] = {}
            
            if "positive" in direction_id:
                axes_dict[axis]["positive"] = direction_info
            elif "negative" in direction_id:
                axes_dict[axis]["negative"] = direction_info
        
        # 只处理3个轴（X, Y, Z）
        axes_list = []
        for axis_name in ["x", "y", "z"]:
            if axis_name in axes_dict:
                axes_list.append((axis_name, axes_dict[axis_name]))
        
        colors = self._generate_distinct_colors(len(axes_list))
        color_mapping = {}
        index_mapping = {}  # 序号映射：序号(从1开始) -> 轴信息（只包含3个轴）
        
        for i, (axis_name, axis_dirs) in enumerate(axes_list):
            r, g, b = colors[i]
            color_hex = self._rgb_to_hex(r, g, b)
            sequence_number = i + 1  # 序号从1开始（只有3个：1=X轴, 2=Y轴, 3=Z轴）
            
            # 使用正方向作为代表（如果存在）
            positive_dir = axis_dirs.get("positive")
            negative_dir = axis_dirs.get("negative")
            
            if not positive_dir:
                continue
            
            # 生成语义信息（基于正方向，但说明是双向的）
            semantic_info = self._generate_sliding_semantic_info(positive_dir, part_function)
            # 修改语义信息，说明这是双向轴
            semantic_info = semantic_info.replace("direction", "axis (bidirectional sliding)")
            if "forward" in semantic_info or "backward" in semantic_info or "upward" in semantic_info or "downward" in semantic_info:
                semantic_info = f"{axis_name.upper()}-axis (bidirectional sliding), " + semantic_info
            
            # 使用轴名称作为direction_id（格式：sliding_axis_1, sliding_axis_2等）
            new_direction_id = f"sliding_axis_{sequence_number}"
            
            # 为两个方向都创建颜色映射（使用相同的颜色）
            if positive_dir:
                pos_color_hex = color_hex
                color_mapping[pos_color_hex] = {
                    "direction_id": new_direction_id,  # 使用轴序号作为ID
                    "reference_direction_id": positive_dir["direction_id"],  # 保留原始ID作为参考
                    "rgb": [r, g, b],
                    "hex": pos_color_hex,
                    "index": i,
                    "sequence_number": sequence_number,  # 序号（从1开始）
                    "direction": positive_dir["direction"],
                    "magnitude": positive_dir.get("magnitude", 0),
                    "axis": axis_name,
                    "description": f"{axis_name.upper()}-axis (bidirectional)",
                    "semantic_info": semantic_info,
                    "direction_type": "principal"
                }
            
            if negative_dir:
                neg_color_hex = color_hex  # 使用相同的颜色
                color_mapping[neg_color_hex] = {
                    "direction_id": new_direction_id,  # 使用相同的轴序号ID
                    "reference_direction_id": negative_dir["direction_id"],  # 保留原始ID作为参考
                    "rgb": [r, g, b],
                    "hex": neg_color_hex,
                    "index": i,
                    "sequence_number": sequence_number,  # 相同的序号
                    "direction": negative_dir["direction"],
                    "magnitude": negative_dir.get("magnitude", 0),
                    "axis": axis_name,
                    "description": f"{axis_name.upper()}-axis (bidirectional)",
                    "semantic_info": semantic_info,
                    "direction_type": "principal"
                }
            
            # 创建序号映射（每个轴只有一个条目）
            index_mapping[sequence_number] = {
                "axis_id": new_direction_id,  # 使用轴序号作为ID
                "axis": axis_name.upper(),  # 轴名称（X, Y, Z）
                "reference_direction_id": positive_dir["direction_id"],  # 保留正方向的原始ID
                "direction": positive_dir["direction"],  # 正方向向量
                "semantic_info": semantic_info,
                "description": f"{axis_name.upper()}-axis (bidirectional sliding along ±{axis_name.upper()})"
            }
        
        return color_mapping, index_mapping
    
    def _layout_text_labels_avoid_overlap(
        self,
        text_positions: List[np.ndarray],
        base_positions: List[np.ndarray],
        perp_vectors: List[np.ndarray],
        offset_directions: List[np.ndarray],
        min_distance: float = 0.05,
        max_iterations: int = 50
    ) -> List[np.ndarray]:
        """智能布局文本标注，避免重叠
        
        Args:
            text_positions: 初始文本位置列表
            base_positions: 基础位置列表（edge/axis的端点）
            perp_vectors: 垂直于轴/边的向量列表
            offset_directions: 沿轴/边的偏移方向列表
            min_distance: 最小间距（相对于场景大小）
            max_iterations: 最大迭代次数
            
        Returns:
            调整后的文本位置列表
        """
        if len(text_positions) <= 1:
            return text_positions
        
        # 转换为numpy数组
        positions = np.array(text_positions)
        base_pos = np.array(base_positions)
        perp_vecs = np.array(perp_vectors)
        offset_dirs = np.array(offset_directions)
        
        # 计算场景大小用于归一化距离
        scene_size = np.max(np.ptp(positions, axis=0)) if len(positions) > 0 else 1.0
        min_dist_normalized = min_distance * scene_size
        
        # 增大最小距离，更激进地分离
        min_dist_normalized = max(min_dist_normalized, scene_size * 0.15)  # 至少15%的场景大小
        
        # 迭代调整位置
        for iteration in range(max_iterations):
            has_overlap = False
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    
                    if dist < min_dist_normalized:
                        has_overlap = True
                        
                        # 计算分离方向
                        separation_dir = positions[i] - positions[j]
                        if np.linalg.norm(separation_dir) < 1e-10:
                            # 如果位置完全相同，使用随机方向
                            separation_dir = np.random.randn(3)
                        separation_dir = separation_dir / (np.linalg.norm(separation_dir) + 1e-10)
                        
                        # 计算需要移动的距离 - 更激进的分离
                        # 不仅分离到最小距离，还要额外增加50%的缓冲
                        target_distance = min_dist_normalized * 1.5
                        move_distance = (target_distance - dist) / 2.0
                        # 确保移动距离足够大
                        move_distance = max(move_distance, min_dist_normalized * 0.3)
                        
                        # 移动两个位置，使其分离
                        # 优先沿垂直于轴/边的方向移动
                        move_i = separation_dir * move_distance
                        move_j = -separation_dir * move_distance
                        
                        # 尝试沿垂直于轴/边的方向移动
                        if i < len(perp_vecs) and np.linalg.norm(perp_vecs[i]) > 0.1:
                            perp_component_i = np.dot(move_i, perp_vecs[i]) * perp_vecs[i]
                            if np.linalg.norm(perp_component_i) > 0.01:
                                # 放大垂直分量，使其更明显
                                move_i = perp_component_i * (move_distance * 1.5 / np.linalg.norm(perp_component_i))
                        
                        if j < len(perp_vecs) and np.linalg.norm(perp_vecs[j]) > 0.1:
                            perp_component_j = np.dot(move_j, perp_vecs[j]) * perp_vecs[j]
                            if np.linalg.norm(perp_component_j) > 0.01:
                                # 放大垂直分量，使其更明显
                                move_j = perp_component_j * (move_distance * 1.5 / np.linalg.norm(perp_component_j))
                        
                        positions[i] += move_i
                        positions[j] += move_j
            
            if not has_overlap:
                break
        
        # 对于仍然重叠的位置，使用垂直排列
        # 检测密集区域
        clusters = []
        used = set()
        
        for i in range(len(positions)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(positions)):
                if j in used:
                    continue
                
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_dist_normalized * 1.5:  # 如果距离很近，认为是同一簇
                    cluster.append(j)
                    used.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        # 对每个簇进行垂直排列
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            
            # 找到簇的中心和主要方向
            cluster_positions = positions[cluster]
            cluster_center = np.mean(cluster_positions, axis=0)
            
            # 使用第一个元素的垂直向量作为排列方向
            if cluster[0] < len(perp_vecs) and np.linalg.norm(perp_vecs[cluster[0]]) > 0.1:
                arrange_dir = perp_vecs[cluster[0]]
            else:
                # 如果没有垂直向量，使用Y轴方向
                arrange_dir = np.array([0, 1, 0])
            
            arrange_dir = arrange_dir / (np.linalg.norm(arrange_dir) + 1e-10)
            
            # 计算排列间距 - 增大间距，避免重叠
            arrange_spacing = min_dist_normalized * 2.0  # 增加到2倍，更激进
            
            # 垂直排列
            for idx, pos_idx in enumerate(cluster):
                offset = (idx - len(cluster) / 2.0) * arrange_spacing
                positions[pos_idx] = cluster_center + arrange_dir * offset
        
        return positions.tolist()
    
    def _visualize_edges_unified(
        self,
        xml_path: str,
        part_name: str,
        edges: List[Dict[str, Any]],
        color_mapping: Dict[str, Dict[str, Any]],
        output_dir: Optional[str]
    ) -> Optional[str]:
        """统一的edge可视化"""
        try:
            # 设置非交互式后端，避免在多线程环境中创建GUI窗口（macOS要求）
            import matplotlib
            matplotlib.use('Agg')  # 必须在导入pyplot之前设置
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.patches import Rectangle
        except ImportError:
            print("✗ 缺少可视化依赖（matplotlib），跳过可视化")
            return None
        
        try:
            from ..utils.mesh_analyzer import MeshAnalyzer
            
            # 加载mesh信息
            analyzer = MeshAnalyzer(xml_path)
            mesh_info_dict = analyzer.analyze()
            
            if part_name not in mesh_info_dict:
                return None
            
            mesh_info = mesh_info_dict[part_name]
            mesh_file = mesh_info.file_path
            mesh_obj = trimesh.load(mesh_file)
            
            if not isinstance(mesh_obj, trimesh.Trimesh):
                if isinstance(mesh_obj, trimesh.Scene):
                    mesh_obj = trimesh.util.concatenate([m for m in mesh_obj.geometry.values() 
                                                         if isinstance(m, trimesh.Trimesh)])
                else:
                    return None
            
            # 创建图形（移除右侧图例，只显示3D图）
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制mesh
            ax.plot_trisurf(
                mesh_obj.vertices[:, 0],
                mesh_obj.vertices[:, 1],
                mesh_obj.vertices[:, 2],
                triangles=mesh_obj.faces,
                alpha=0.15,
                color='lightblue',
                edgecolor='none'
            )
            
            # 绘制edge
            for edge in edges:
                edge_id = edge["edge_id"]
                # 查找对应的颜色信息
                color_info = None
                for hex_color, info in color_mapping.items():
                    if info.get("edge_id") == edge_id:
                        color_info = info
                        break
                
                if not color_info:
                    continue
                
                midpoint = np.array(edge["midpoint"])
                direction = np.array(edge["direction"])
                length = edge["length"]
                
                start = midpoint - direction * length / 2
                end = midpoint + direction * length / 2
                
                r, g, b = color_info["rgb"]
                sequence_number = color_info.get("sequence_number", color_info.get("index", 0) + 1)
                
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=(r, g, b),
                    linewidth=3.5,
                    alpha=0.9,
                    label=f"Edge {sequence_number}"
                )
            
            # 收集所有edge信息用于智能布局
            edge_data = []
            for edge in edges:
                edge_id = edge["edge_id"]
                color_info = None
                for hex_color, info in color_mapping.items():
                    if info.get("edge_id") == edge_id:
                        color_info = info
                        break
                
                if not color_info:
                    continue
                
                midpoint = np.array(edge["midpoint"])
                direction = np.array(edge["direction"])
                length = edge["length"]
                start = midpoint - direction * length / 2
                end = midpoint + direction * length / 2
                sequence_number = color_info.get("sequence_number", color_info.get("index", 0) + 1)
                
                # 计算垂直于edge方向的向量
                if abs(direction[2]) < 0.9:
                    perp_base = np.cross(direction, [0, 0, 1])
                    if np.linalg.norm(perp_base) < 0.1:
                        perp_base = np.cross(direction, [0, 1, 0])
                else:
                    perp_base = np.cross(direction, [1, 0, 0])
                perp_base = perp_base / (np.linalg.norm(perp_base) + 1e-10)
                
                # 初始位置：使用黄金角度分布
                golden_angle = 2.399963229728653
                angle_offset = (sequence_number - 1) * golden_angle
                use_start = (sequence_number % 2 == 0)
                radial_offset_base = 0.15
                radial_offset_variation = (sequence_number % 5) * 0.05
                radial_offset = radial_offset_base + radial_offset_variation
                along_edge_offset = (sequence_number % 3) * 0.08
                
                if use_start:
                    base_pos = start
                    offset_dir = -direction
                else:
                    base_pos = end
                    offset_dir = direction
                
                perp_vec = perp_base * np.cos(angle_offset) + np.cross(direction, perp_base) * np.sin(angle_offset)
                perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-10)
                
                offset_distance = length * radial_offset
                initial_text_pos = base_pos + perp_vec * offset_distance + offset_dir * (length * along_edge_offset)
                
                edge_data.append({
                    "sequence_number": sequence_number,
                    "color": color_info["rgb"],
                    "base_pos": base_pos,
                    "perp_vec": perp_vec,
                    "offset_dir": offset_dir,
                    "initial_pos": initial_text_pos
                })
            
            # 使用智能布局算法调整位置
            if edge_data:
                text_positions = [ed["initial_pos"] for ed in edge_data]
                base_positions = [ed["base_pos"] for ed in edge_data]
                perp_vectors = [ed["perp_vec"] for ed in edge_data]
                offset_directions = [ed["offset_dir"] for ed in edge_data]
                
                # 计算场景大小用于设置最小距离 - 增大最小距离
                all_positions = np.array([ed["base_pos"] for ed in edge_data])
                scene_size = np.max(np.ptp(all_positions, axis=0)) if len(all_positions) > 0 else 1.0
                min_distance = 0.15 * scene_size  # 增加到15%的场景大小，更激进
                
                adjusted_positions = self._layout_text_labels_avoid_overlap(
                    text_positions, base_positions, perp_vectors, offset_directions,
                    min_distance=min_distance / scene_size  # 归一化
                )
                
                # 绘制调整后的文本
                for ed, text_pos in zip(edge_data, adjusted_positions):
                    ax.text(
                        text_pos[0], text_pos[1], text_pos[2],
                        str(ed["sequence_number"]),
                        fontsize=14,
                        fontweight='bold',
                        color=tuple(ed["color"]),
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=tuple(ed["color"]), linewidth=2.0)
                    )
            
            # 设置坐标轴
            aabb = mesh_info.aabb
            aabb_size = np.array(aabb.size)
            max_range = max(aabb_size) / 2.0
            center = np.array(aabb.center)
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            # 设置坐标轴标签（英文）
            ax.set_xlabel('X', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z', fontsize=12, fontweight='bold')
            ax.set_title(f"{part_name} - Edge Rotation Axes", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图像
            output_path = Path(output_dir or str(Path(xml_path).parent)) / f"{part_name}_edge_rotation_axes.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存到: {output_path}")
            
            plt.close(fig)
            return str(output_path)
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_centerlines_unified(
        self,
        xml_path: str,
        part_name: str,
        centerlines: Dict[str, List[Dict[str, Any]]],
        color_mapping: Dict[str, Dict[str, Any]],
        output_dir: Optional[str]
    ) -> Optional[str]:
        """统一的中心线可视化"""
        try:
            # 设置非交互式后端，避免在多线程环境中创建GUI窗口（macOS要求）
            import matplotlib
            matplotlib.use('Agg')  # 必须在导入pyplot之前设置
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.patches import Rectangle
        except ImportError:
            print("✗ 缺少可视化依赖（matplotlib），跳过可视化")
            return None
        
        try:
            from ..utils.mesh_analyzer import MeshAnalyzer
            
            # 加载mesh信息
            analyzer = MeshAnalyzer(xml_path)
            mesh_info_dict = analyzer.analyze()
            
            if part_name not in mesh_info_dict:
                return None
            
            mesh_info = mesh_info_dict[part_name]
            mesh_file = mesh_info.file_path
            mesh_obj = trimesh.load(mesh_file)
            
            if not isinstance(mesh_obj, trimesh.Trimesh):
                if isinstance(mesh_obj, trimesh.Scene):
                    mesh_obj = trimesh.util.concatenate([m for m in mesh_obj.geometry.values() 
                                                         if isinstance(m, trimesh.Trimesh)])
                else:
                    return None
            
            # 收集所有中心线
            all_axes = []
            all_axes.extend(centerlines.get("principal_axes", []))
            all_axes.extend(centerlines.get("centroid_axes", []))
            all_axes.extend(centerlines.get("diagonal_axes", []))
            
            # 创建图形（移除右侧图例，只显示3D图）
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制mesh
            ax.plot_trisurf(
                mesh_obj.vertices[:, 0],
                mesh_obj.vertices[:, 1],
                mesh_obj.vertices[:, 2],
                triangles=mesh_obj.faces,
                alpha=0.15,
                color='lightgray',
                edgecolor='none'
            )
            
            # 获取AABB用于缩放
            aabb = mesh_info.aabb
            aabb_size = np.array(aabb.size)
            max_size = max(aabb_size)
            axis_length = max_size * 0.6
            
            # 绘制中心线
            for axis_info in all_axes:
                axis_id = axis_info["axis_id"]
                # 查找对应的颜色信息
                color_info = None
                for hex_color, info in color_mapping.items():
                    if info.get("axis_id") == axis_id:
                        color_info = info
                        break
                
                if not color_info:
                    continue
                
                point = np.array(axis_info["point"])
                direction = np.array(axis_info["direction"])
                length = axis_info.get("length", axis_length)
                
                start = point - direction * length / 2
                end = point + direction * length / 2
                
                r, g, b = color_info["rgb"]
                sequence_number = color_info.get("sequence_number", color_info.get("index", 0) + 1)
                
                # 根据轴类型选择线型
                linestyle = '-'
                if color_info["axis_type"] == "centroid":
                    linestyle = '--'
                elif color_info["axis_type"] == "diagonal":
                    linestyle = ':'
                
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=(r, g, b),
                    linewidth=3.0,
                    alpha=0.9,
                    linestyle=linestyle,
                    label=f"Axis {sequence_number}"
                )
                
                ax.scatter(
                    [point[0]],
                    [point[1]],
                    [point[2]],
                    color=(r, g, b),
                    s=100,
                    marker='o',
                    alpha=0.9,
                    edgecolors='black',
                    linewidths=1.0
                )
            
            # 收集所有轴信息用于智能布局
            axis_data = []
            for axis_info in all_axes:
                axis_id = axis_info["axis_id"]
                color_info = None
                for hex_color, info in color_mapping.items():
                    if info.get("axis_id") == axis_id:
                        color_info = info
                        break
                
                if not color_info:
                    continue
                
                point = np.array(axis_info["point"])
                direction = np.array(axis_info["direction"])
                length = axis_info.get("length", axis_length)
                start = point - direction * length / 2
                end = point + direction * length / 2
                sequence_number = color_info.get("sequence_number", color_info.get("index", 0) + 1)
                
                # 计算垂直于轴方向的向量
                if abs(direction[2]) < 0.9:
                    perp_base = np.cross(direction, [0, 0, 1])
                    if np.linalg.norm(perp_base) < 0.1:
                        perp_base = np.cross(direction, [0, 1, 0])
                else:
                    perp_base = np.cross(direction, [1, 0, 0])
                perp_base = perp_base / (np.linalg.norm(perp_base) + 1e-10)
                
                # 初始位置：使用黄金角度分布
                golden_angle = 2.399963229728653
                angle_offset = (sequence_number - 1) * golden_angle
                use_start = (sequence_number % 2 == 0)
                radial_offset_base = 0.15
                radial_offset_variation = (sequence_number % 5) * 0.05
                radial_offset = radial_offset_base + radial_offset_variation
                along_axis_offset = (sequence_number % 3) * 0.08
                
                if use_start:
                    base_pos = start
                    offset_dir = -direction
                else:
                    base_pos = end
                    offset_dir = direction
                
                perp_vec = perp_base * np.cos(angle_offset) + np.cross(direction, perp_base) * np.sin(angle_offset)
                perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-10)
                
                offset_distance = axis_length * radial_offset
                initial_text_pos = base_pos + perp_vec * offset_distance + offset_dir * (axis_length * along_axis_offset)
                
                axis_data.append({
                    "sequence_number": sequence_number,
                    "color": color_info["rgb"],
                    "base_pos": base_pos,
                    "perp_vec": perp_vec,
                    "offset_dir": offset_dir,
                    "initial_pos": initial_text_pos
                })
            
            # 使用智能布局算法调整位置
            if axis_data:
                text_positions = [ad["initial_pos"] for ad in axis_data]
                base_positions = [ad["base_pos"] for ad in axis_data]
                perp_vectors = [ad["perp_vec"] for ad in axis_data]
                offset_directions = [ad["offset_dir"] for ad in axis_data]
                
                # 计算场景大小用于设置最小距离 - 增大最小距离
                all_positions = np.array([ad["base_pos"] for ad in axis_data])
                scene_size = np.max(np.ptp(all_positions, axis=0)) if len(all_positions) > 0 else 1.0
                min_distance = 0.15 * scene_size  # 增加到15%的场景大小，更激进
                
                adjusted_positions = self._layout_text_labels_avoid_overlap(
                    text_positions, base_positions, perp_vectors, offset_directions,
                    min_distance=min_distance / scene_size
                )
                
                # 绘制调整后的文本
                for ad, text_pos in zip(axis_data, adjusted_positions):
                    ax.text(
                        text_pos[0], text_pos[1], text_pos[2],
                        str(ad["sequence_number"]),
                        fontsize=14,
                        fontweight='bold',
                        color=tuple(ad["color"]),
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=tuple(ad["color"]), linewidth=2.0)
                    )
            
            # 设置坐标轴
            max_range = max(aabb_size) / 2.0
            center = np.array(aabb.center)
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)
            ax.set_title(f"{part_name} - Centerline Rotation Axes", fontsize=14, fontweight='bold')
            
            # 不再创建颜色图例
            
            plt.tight_layout()
            
            # 保存图像
            output_path = Path(output_dir or str(Path(xml_path).parent)) / f"{part_name}_centerline_rotation_axes.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存到: {output_path}")
            
            plt.close(fig)
            return str(output_path)
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_sliding_unified(
        self,
        xml_path: str,
        part_name: str,
        directions: Dict[str, List[Dict[str, Any]]],
        color_mapping: Dict[str, Dict[str, Any]],
        output_dir: Optional[str]
    ) -> Optional[str]:
        """统一的滑动方向可视化"""
        try:
            # 设置非交互式后端，避免在多线程环境中创建GUI窗口（macOS要求）
            import matplotlib
            matplotlib.use('Agg')  # 必须在导入pyplot之前设置
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.patches import Rectangle
        except ImportError:
            print("✗ 缺少可视化依赖（matplotlib），跳过可视化")
            return None
        
        try:
            from ..utils.mesh_analyzer import MeshAnalyzer
            
            # 加载mesh信息
            analyzer = MeshAnalyzer(xml_path)
            mesh_info_dict = analyzer.analyze()
            
            if part_name not in mesh_info_dict:
                return None
            
            mesh_info = mesh_info_dict[part_name]
            mesh_file = mesh_info.file_path
            mesh_obj = trimesh.load(mesh_file)
            
            if not isinstance(mesh_obj, trimesh.Trimesh):
                if isinstance(mesh_obj, trimesh.Scene):
                    mesh_obj = trimesh.util.concatenate([m for m in mesh_obj.geometry.values() 
                                                         if isinstance(m, trimesh.Trimesh)])
                else:
                    return None
            
            # 收集主轴方向（只考虑principal_directions，忽略face_normal_directions）
            principal_directions = directions.get("principal_directions", [])
            
            if not principal_directions:
                print(f"⚠ 没有找到滑动方向，跳过可视化")
                return None
            
            # 将6个方向（±X, ±Y, ±Z）合并为3个轴
            # 按轴分组：每个轴只显示一个双向箭头
            axes_dict = {}  # {axis_name: {"positive": {...}, "negative": {...}}}
            for direction_info in principal_directions:
                axis = direction_info.get("axis", "").lower()
                direction_id = direction_info.get("direction_id", "")
                
                if axis not in axes_dict:
                    axes_dict[axis] = {}
                
                if "positive" in direction_id:
                    axes_dict[axis]["positive"] = direction_info
                elif "negative" in direction_id:
                    axes_dict[axis]["negative"] = direction_info
            
            # 创建图形（移除右侧图例，只显示3D图）
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制mesh
            ax.plot_trisurf(
                mesh_obj.vertices[:, 0],
                mesh_obj.vertices[:, 1],
                mesh_obj.vertices[:, 2],
                triangles=mesh_obj.faces,
                alpha=0.15,
                color='lightblue',
                edgecolor='none'
            )
            
            # 获取AABB用于缩放
            aabb = mesh_info.aabb
            aabb_center = np.array(aabb.center)
            aabb_size = np.array(aabb.size)
            max_size = max(aabb_size)
            base_arrow_length = max_size * 0.5  # 增加箭头长度，使其更明显
            
            # 绘制三个轴的双向箭头
            axis_data = []  # 用于存储轴标签信息
            axis_colors = {
                "x": (1.0, 0.0, 0.0),  # 红色
                "y": (0.0, 1.0, 0.0),  # 绿色
                "z": (0.0, 0.0, 1.0)   # 蓝色
            }
            
            for axis_idx, (axis_name, axis_dirs) in enumerate(sorted(axes_dict.items())):
                positive_dir = axis_dirs.get("positive")
                negative_dir = axis_dirs.get("negative")
                
                if not positive_dir or not negative_dir:
                    continue
                
                # 获取正方向向量和大小
                pos_vec = np.array(positive_dir["direction"])
                pos_magnitude = positive_dir.get("magnitude", max_size)
                pos_arrow_length = base_arrow_length * (pos_magnitude / max_size)
                
                # 获取负方向向量和大小
                neg_vec = np.array(negative_dir["direction"])
                neg_magnitude = negative_dir.get("magnitude", max_size)
                neg_arrow_length = base_arrow_length * (neg_magnitude / max_size)
                
                # 使用轴对应的颜色，或从color_mapping中查找
                color = axis_colors.get(axis_name, (0.5, 0.5, 0.5))
                
                # 尝试从color_mapping中获取正方向的颜色
                pos_direction_id = positive_dir.get("direction_id", "")
                for hex_color, info in color_mapping.items():
                    if info.get("direction_id") == pos_direction_id:
                        color = info["rgb"]
                        break
                    if info.get("reference_direction_id") == pos_direction_id:
                        color = info["rgb"]
                        break
                
                r, g, b = color
                
                # 绘制正方向箭头
                ax.quiver(
                    aabb_center[0], aabb_center[1], aabb_center[2],
                    pos_vec[0] * pos_arrow_length,
                    pos_vec[1] * pos_arrow_length,
                    pos_vec[2] * pos_arrow_length,
                    color=(r, g, b),
                    arrow_length_ratio=0.15,
                    linewidth=3.0,
                    alpha=0.9
                )
                
                # 绘制负方向箭头
                ax.quiver(
                    aabb_center[0], aabb_center[1], aabb_center[2],
                    neg_vec[0] * neg_arrow_length,
                    neg_vec[1] * neg_arrow_length,
                    neg_vec[2] * neg_arrow_length,
                    color=(r, g, b),
                    arrow_length_ratio=0.15,
                    linewidth=3.0,
                    alpha=0.9
                )
                
                # 计算标签位置（在正方向箭头的末端）
                pos_arrow_end = aabb_center + pos_vec * pos_arrow_length
                
                # 计算垂直于方向的偏移向量用于标签位置
                if abs(pos_vec[2]) < 0.9:
                    perp_base = np.cross(pos_vec, [0, 0, 1])
                    if np.linalg.norm(perp_base) < 0.1:
                        perp_base = np.cross(pos_vec, [0, 1, 0])
                else:
                    perp_base = np.cross(pos_vec, [1, 0, 0])
                perp_base = perp_base / (np.linalg.norm(perp_base) + 1e-10)
                
                # 标签位置：在箭头末端稍微偏移
                label_offset = pos_arrow_length * 0.15
                label_pos = pos_arrow_end + perp_base * label_offset
                
                axis_data.append({
                    "axis_name": axis_name.upper(),
                    "color": color,
                    "label_pos": label_pos
                })
            
            # 绘制轴标签
            for axis_info in axis_data:
                ax.text(
                    axis_info["label_pos"][0],
                    axis_info["label_pos"][1],
                    axis_info["label_pos"][2],
                    axis_info["axis_name"],
                    fontsize=16,
                    fontweight='bold',
                    color=tuple(axis_info["color"]),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                            edgecolor=tuple(axis_info["color"]), linewidth=2.5)
                )
            
            # 标记中心点
            ax.scatter(
                [aabb_center[0]],
                [aabb_center[1]],
                [aabb_center[2]],
                color='black',
                s=100,
                marker='o',
                alpha=0.9,
                edgecolors='black',
                linewidths=1.0
            )
            
            # 设置坐标轴
            max_range = max(aabb_size) / 2.0
            
            ax.set_xlim(aabb_center[0] - max_range, aabb_center[0] + max_range)
            ax.set_ylim(aabb_center[1] - max_range, aabb_center[1] + max_range)
            ax.set_zlim(aabb_center[2] - max_range, aabb_center[2] + max_range)
            
            # 设置坐标轴标签（英文）
            ax.set_xlabel('X', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z', fontsize=12, fontweight='bold')
            ax.set_title(f"{part_name} - Sliding Directions", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图像
            output_path = Path(output_dir or str(Path(xml_path).parent)) / f"{part_name}_sliding_directions.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存到: {output_path}")
            
            plt.close(fig)
            return str(output_path)
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """生成n个不同的颜色，使用更多、更易区分的颜色方案"""
        # 扩展的基础调色板，包含更多易区分的颜色
        base_palette = [
            (1.0, 0.0, 0.0),      # Red - 红色
            (0.0, 1.0, 0.0),      # Green - 绿色
            (0.0, 0.0, 1.0),      # Blue - 蓝色
            (1.0, 1.0, 0.0),      # Yellow - 黄色
            (1.0, 0.0, 1.0),      # Magenta - 洋红
            (0.0, 1.0, 1.0),      # Cyan - 青色
            (1.0, 0.5, 0.0),      # Orange - 橙色
            (0.5, 0.0, 1.0),      # Purple - 紫色
            (1.0, 0.75, 0.8),     # Pink - 粉色
            (0.0, 0.5, 0.5),      # Teal - 青绿色
            (0.5, 1.0, 0.5),      # Light Green - 浅绿色
            (1.0, 0.5, 0.5),      # Light Red - 浅红色
            (0.5, 0.5, 1.0),      # Light Blue - 浅蓝色
            (1.0, 0.8, 0.0),      # Gold - 金色
            (0.8, 0.2, 0.8),      # Violet - 紫罗兰色
            (0.2, 0.8, 0.8),      # Turquoise - 青蓝色
            (0.8, 0.8, 0.2),      # Olive - 橄榄色
            (1.0, 0.4, 0.4),      # Coral - 珊瑚色
            (0.4, 1.0, 0.4),      # Lime - 酸橙绿
            (0.4, 0.4, 1.0),      # Sky Blue - 天蓝色
            (1.0, 0.6, 0.2),      # Dark Orange - 深橙色
            (0.6, 0.2, 1.0),      # Indigo - 靛蓝色
            (0.2, 1.0, 0.6),      # Spring Green - 春绿色
            (1.0, 0.2, 0.6),      # Hot Pink - 热粉色
            (0.2, 0.6, 1.0),      # Royal Blue - 皇家蓝
            (1.0, 0.8, 0.4),      # Peach - 桃色
            (0.8, 0.4, 1.0),      # Lavender - 薰衣草色
            (0.4, 1.0, 0.8),      # Aqua - 水色
            (1.0, 0.4, 0.8),      # Rose - 玫瑰色
            (0.4, 0.8, 1.0),      # Light Sky Blue - 浅天蓝色
        ]
        
        if n <= len(base_palette):
            return base_palette[:n]
        
        colors = list(base_palette)
        remain = n - len(base_palette)
        # 使用HSL生成额外颜色，确保高饱和度和中等亮度以便区分
        for k in range(remain):
            # 使用黄金角度（约137.5度）来最大化颜色差异
            golden_angle = 0.618033988749895  # 黄金比例
            h = (k * golden_angle) % 1.0
            s = 0.9  # 高饱和度
            l = 0.5  # 中等亮度
            r, g, b = self._hsl_to_rgb(h, s, l)
            colors.append((r, g, b))
        return colors
    
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> Tuple[float, float, float]:
        """HSL转RGB"""
        def hue_to_rgb(p: float, q: float, t: float) -> float:
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        if s == 0:
            return l, l, l
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        return r, g, b
    
    def _rgb_to_hex(self, r: float, g: float, b: float) -> str:
        """RGB转HEX"""
        r_int = int(round(r * 255))
        g_int = int(round(g * 255))
        b_int = int(round(b * 255))
        return f"#{r_int:02X}{g_int:02X}{b_int:02X}"


__all__ = [
    "FindCenterlineAxesTool",
    "AnalyzeSlidingDirectionTool",
    "FindPartEdgesTool",
    "AnalyzeMotionTypeTool",
]

