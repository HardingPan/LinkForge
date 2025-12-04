from __future__ import annotations

import os
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET


try:
    import numpy as np
    import trimesh
except Exception as import_error:  # pragma: no cover
    raise RuntimeError(
        "需要依赖 numpy 与 trimesh。请先安装：pip install numpy trimesh"
    ) from import_error


Vector3 = Tuple[float, float, float]


@dataclass
class AABB:
    minimum: Vector3
    maximum: Vector3

    @property
    def size(self) -> Vector3:
        return (
            self.maximum[0] - self.minimum[0],
            self.maximum[1] - self.minimum[1],
            self.maximum[2] - self.minimum[2],
        )

    @property
    def center(self) -> Vector3:
        return (
            (self.minimum[0] + self.maximum[0]) / 2.0,
            (self.minimum[1] + self.maximum[1]) / 2.0,
            (self.minimum[2] + self.maximum[2]) / 2.0,
        )


@dataclass
class MeshKeyPoints:
    corners: Dict[str, Vector3]
    face_centers: Dict[str, Vector3]
    edge_midpoints: Dict[str, Vector3]


@dataclass
class MeshInfo:
    name: str
    file_path: str
    scale: Optional[Vector3]
    aabb: AABB
    keypoints: MeshKeyPoints

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "scale": self.scale,
            "aabb": {
                "min": self.aabb.minimum,
                "max": self.aabb.maximum,
                "size": self.aabb.size,
                "center": self.aabb.center,
            },
            "keypoints": {
                "corners": self.keypoints.corners,
                "face_centers": self.keypoints.face_centers,
                "edge_midpoints": self.keypoints.edge_midpoints,
            },
        }


class MeshAnalyzer:
    """
    解析 MJCF(XML) 中被使用到的 mesh 资产，加载其网格并计算：
    - AABB 包围盒（最小点、最大点、尺寸、中心）
    - 包围盒的 8 个角点
    - 6 个面中心点（±X, ±Y, ±Z）
    - 12 条边的中点

    注意：结果坐标均在 mesh 本地坐标系（已考虑 <mesh> 的 scale）。
    不包含 MuJoCo 中各个 geom 的位姿变换。
    """

    def __init__(self, xml_path: str) -> None:
        if not os.path.isabs(xml_path):
            xml_path = os.path.abspath(xml_path)
        self.xml_path: str = xml_path
        self.xml_dir: str = os.path.dirname(xml_path)
        self._mesh_name_to_file: Dict[str, str] = {}
        self._mesh_name_to_scale: Dict[str, Optional[Vector3]] = {}
        self._used_mesh_names: List[str] = []
        self._mesh_info: Dict[str, MeshInfo] = {}

    # --------------- Public API ---------------
    def analyze(self) -> Dict[str, MeshInfo]:
        """解析 XML、筛选被使用的 mesh，加载并计算包围盒与关键点。"""
        self._parse_mjcf()
        for mesh_name in self._used_mesh_names:
            file_rel = self._mesh_name_to_file[mesh_name]
            scale = self._mesh_name_to_scale.get(mesh_name)
            file_abs = os.path.abspath(os.path.join(self.xml_dir, file_rel))

            mesh = self._load_mesh(file_abs)
            if scale is not None:
                # 支持各向异性缩放
                mesh.apply_scale(np.asarray(scale, dtype=float))

            aabb = self._compute_aabb(mesh)
            keypoints = self._compute_keypoints(aabb)
            self._mesh_info[mesh_name] = MeshInfo(
                name=mesh_name,
                file_path=file_abs,
                scale=scale,
                aabb=aabb,
                keypoints=keypoints,
            )

        return self._mesh_info

    def get_info(self, mesh_name: str) -> Optional[MeshInfo]:
        return self._mesh_info.get(mesh_name)

    def get_all(self) -> Dict[str, MeshInfo]:
        return self._mesh_info

    # --------------- MJCF Parsing ---------------
    def _parse_mjcf(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # 1) 收集 <asset><mesh> 声明
        asset = root.find("asset")
        if asset is None:
            return

        for mesh_elem in asset.findall("mesh"):
            name = mesh_elem.get("name")
            file_attr = mesh_elem.get("file")
            if not name or not file_attr:
                continue

            scale_attr = mesh_elem.get("scale")
            scale_vec: Optional[Vector3] = None
            if scale_attr:
                parts = [p for p in scale_attr.replace(",", " ").split() if p]
                if len(parts) == 3:
                    scale_vec = (float(parts[0]), float(parts[1]), float(parts[2]))

            self._mesh_name_to_file[name] = file_attr
            self._mesh_name_to_scale[name] = scale_vec

        # 2) 找出被使用到的 mesh（任意层级下的 <geom type="mesh" mesh="...">）
        used: List[str] = []
        for geom in root.iter("geom"):
            if geom.get("type") == "mesh":
                mesh_ref = geom.get("mesh")
                if mesh_ref and mesh_ref in self._mesh_name_to_file and mesh_ref not in used:
                    used.append(mesh_ref)
        self._used_mesh_names = used

    # --------------- Geometry Utils ---------------
    def _load_mesh(self, file_path: str) -> trimesh.Trimesh:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到网格文件: {file_path}")

        loaded = trimesh.load(file_path, force='mesh')
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        # 某些格式会返回 Scene，此时合并为单个 Mesh
        if hasattr(loaded, 'geometry'):
            geometries = list(loaded.geometry.values())
            if len(geometries) == 0:
                raise ValueError(f"文件未包含几何: {file_path}")
            return trimesh.util.concatenate(geometries)

        raise TypeError(f"无法识别的网格类型: {type(loaded)} 来自 {file_path}")

    def _compute_aabb(self, mesh: trimesh.Trimesh) -> AABB:
        vmin = mesh.vertices.min(axis=0)
        vmax = mesh.vertices.max(axis=0)
        minimum: Vector3 = (float(vmin[0]), float(vmin[1]), float(vmin[2]))
        maximum: Vector3 = (float(vmax[0]), float(vmax[1]), float(vmax[2]))
        return AABB(minimum=minimum, maximum=maximum)

    def _compute_keypoints(self, aabb: AABB) -> MeshKeyPoints:
        xmin, ymin, zmin = aabb.minimum
        xmax, ymax, zmax = aabb.maximum
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0
        zmid = (zmin + zmax) / 2.0

        # 8 角点（用标签表明朝向，便于查阅）
        corners: Dict[str, Vector3] = {
            "min_min_min": (xmin, ymin, zmin),
            "min_min_max": (xmin, ymin, zmax),
            "min_max_min": (xmin, ymax, zmin),
            "min_max_max": (xmin, ymax, zmax),
            "max_min_min": (xmax, ymin, zmin),
            "max_min_max": (xmax, ymin, zmax),
            "max_max_min": (xmax, ymax, zmin),
            "max_max_max": (xmax, ymax, zmax),
        }

        # 6 面中心（按 ±X, ±Y, ±Z）
        face_centers: Dict[str, Vector3] = {
            "-x": (xmin, ymid, zmid),
            "+x": (xmax, ymid, zmid),
            "-y": (xmid, ymin, zmid),
            "+y": (xmid, ymax, zmid),
            "-z": (xmid, ymid, zmin),
            "+z": (xmid, ymid, zmax),
        }

        # 12 边中点（固定一轴在极值，另外两轴在中点）
        edge_midpoints: Dict[str, Vector3] = {
            # 固定 X
            "-x,-y": (xmin, ymin, zmid),
            "-x,+y": (xmin, ymax, zmid),
            "-x,-z": (xmin, ymid, zmin),
            "-x,+z": (xmin, ymid, zmax),
            "+x,-y": (xmax, ymin, zmid),
            "+x,+y": (xmax, ymax, zmid),
            "+x,-z": (xmax, ymid, zmin),
            "+x,+z": (xmax, ymid, zmax),
            # 固定 Y
            "-y,-z": (xmid, ymin, zmin),
            "-y,+z": (xmid, ymin, zmax),
            "+y,-z": (xmid, ymax, zmin),
            "+y,+z": (xmid, ymax, zmax),
        }

        return MeshKeyPoints(corners=corners, face_centers=face_centers, edge_midpoints=edge_midpoints)


__all__ = [
    "MeshAnalyzer",
]