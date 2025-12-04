from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .image_utils import (
    capture_mujoco_multiview_mosaic,
    capture_mujoco_multiview_to_data_urls,
    build_image_url,
)


class MujocoRenderController:
    """控制MuJoCo模型高亮与多视角渲染的助手类。

    用法示例：

        controller = MujocoRenderController("Examples/wardrobe/obj.xml")
        controller.set_highlights(["Cube001", "Cube011", "Plane001"])  # 指定需要高亮的mesh名
        data_url = controller.render(num_views=9, mosaic=True, save=True)  # 返回马赛克data URL，并保存PNG

    设计要点：
    - 不更改原XML；按需在同目录生成临时高亮版本XML，以保证相对路径mesh可被正确解析。
    - 非高亮构件统一半透明显示；高亮构件分配鲜艳且互异的颜色。
    - 渲染输出默认为9视角马赛克图（返回data URL），可切换为单独多图data URL列表。
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        default_other_alpha: float = 0.2,
        highlight_palette: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ) -> None:
        self.model_path: Path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo模型文件不存在: {self.model_path}")

        self._highlight_names: List[str] = []
        self.default_other_alpha: float = float(np.clip(default_other_alpha, 0.0, 1.0))

        # 基础调色板；若数量不够，会通过HSL色轮动态生成更多颜色
        self.highlight_palette: List[Tuple[float, float, float, float]] = list(
            highlight_palette
            or [
                (1.0, 0.2, 0.2, 1.0),
                (0.2, 0.8, 0.25, 1.0),
                (0.2, 0.4, 1.0, 1.0),
                (1.0, 0.4, 0.0, 1.0),
                (0.7, 0.2, 1.0, 1.0),
                (1.0, 0.2, 0.8, 1.0),
                (0.1, 0.9, 0.9, 1.0),
                (1.0, 0.9, 0.1, 1.0),
                (0.9, 0.6, 0.1, 1.0),
            ]
        )
        # 缓存：最后一次生成的映射（用于导出API）
        self._last_mapping_rgba: Dict[str, Tuple[float, float, float, float]] = {}

        # 9视角默认序列（与 image_utils._compute_fit_view_cameras 支持名称一致）
        self.default_views: List[str] = [
            "front",
            "front_right",
            "right",
            "front_left",
            "iso",
            "back_left",
            "left",
            "back",
            "top",
        ]

        # 临时XML路径（放置在与原文件同目录，确保mesh相对路径可用）
        self._tmp_xml_path: Optional[Path] = None
        # 记录所有生成的临时文件，用于批量清理
        self._all_tmp_xml_paths: List[Path] = []

    # ----------------------------- 公共API -----------------------------
    def set_highlights(self, mesh_names: Sequence[str]) -> None:
        """设置需要高亮显示的mesh名称列表。"""
        self._highlight_names = [str(n).strip() for n in mesh_names if str(n).strip()]
        # 使之前生成的临时XML失效，确保下次渲染应用新高亮
        if self._tmp_xml_path is not None and self._tmp_xml_path.exists():
            try:
                self._tmp_xml_path.unlink()
            except Exception:
                pass
        self._tmp_xml_path = None
        self._last_mapping_rgba = {}

    def render(
        self,
        num_views: int = 9,
        views: Optional[Sequence[str]] = None,
        mosaic: bool = True,
        width: int = 512,
        height: int = 384,
        save: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        tile_shape: Optional[Tuple[int, int]] = None,
        pad: int = 4,
    ) -> Union[str, List[str]]:
        """渲染多视角图像。

        - 当 mosaic=True：返回单张马赛克图的 data URL；若 save=True 同时保存 PNG。
        - 当 mosaic=False：返回每个视角的 data URL 列表；若 save=True 同时保存多张 PNG。
        """
        tmp_xml = self._ensure_tmp_highlighted_xml()

        if views is None:
            # 基于默认9视角，若需要更少则截断
            v = self.default_views
            if isinstance(num_views, int) and num_views > 0:
                v = v[: min(num_views, len(v))]
            views = v

        if mosaic:
            mosaic_arr = capture_mujoco_multiview_mosaic(
                model_path=tmp_xml,
                width=width,
                height=height,
                num_views=len(views),
                views=list(views),
                save_image=save,
                save_path=str(save_path) if save and save_path else None,
                tile_shape=tile_shape,
                pad=pad,
            )
            result = build_image_url(mosaic_arr)
            # 渲染完成后自动清理临时XML文件
            self.cleanup_temp_files()
            return result

        # 非马赛克：多图返回
        data_urls = capture_mujoco_multiview_to_data_urls(
            model_path=tmp_xml,
            width=width,
            height=height,
            num_views=len(views),
            views=list(views),
            save_images=save,
            save_prefix=(str(save_path) if save and save_path else None),
        )
        # 渲染完成后自动清理临时XML文件
        self.cleanup_temp_files()
        return data_urls

    def render_original(
        self,
        num_views: int = 9,
        views: Optional[Sequence[str]] = None,
        mosaic: bool = True,
        width: int = 512,
        height: int = 384,
        save: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        tile_shape: Optional[Tuple[int, int]] = None,
        pad: int = 4,
    ) -> Union[str, List[str]]:
        """渲染原始图像（无高亮效果）。

        - 当 mosaic=True：返回单张马赛克图的 data URL；若 save=True 同时保存 PNG。
        - 当 mosaic=False：返回每个视角的 data URL 列表；若 save=True 同时保存多张 PNG。
        """
        if views is None:
            # 基于默认9视角，若需要更少则截断
            v = self.default_views
            if isinstance(num_views, int) and num_views > 0:
                v = v[: min(num_views, len(v))]
            views = v

        if mosaic:
            mosaic_arr = capture_mujoco_multiview_mosaic(
                model_path=self.model_path,  # 直接使用原始XML，不应用高亮
                width=width,
                height=height,
                num_views=len(views),
                views=list(views),
                save_image=save,
                save_path=str(save_path) if save and save_path else None,
                tile_shape=tile_shape,
                pad=pad,
            )
            return build_image_url(mosaic_arr)

        # 非马赛克：多图返回
        data_urls = capture_mujoco_multiview_to_data_urls(
            model_path=self.model_path,  # 直接使用原始XML，不应用高亮
            width=width,
            height=height,
            num_views=len(views),
            views=list(views),
            save_images=save,
            save_prefix=(str(save_path) if save and save_path else None),
        )
        return data_urls

    # ----------------------------- 内部方法 -----------------------------
    def _ensure_tmp_highlighted_xml(self) -> Path:
        """生成或复用带高亮效果的临时XML路径。"""
        if self._tmp_xml_path is not None and self._tmp_xml_path.exists():
            return self._tmp_xml_path

        modified_xml = self._build_highlighted_xml_string()
        dir_path = self.model_path.parent
        stem = self.model_path.stem
        # 添加时间戳和随机数确保文件名唯一性，避免多线程竞争
        import time
        import random
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        random_id = random.randint(1000, 9999)
        tmp_name = f"{stem}.__highlight___{timestamp}_{random_id}.xml"
        tmp_path = dir_path / tmp_name
        tmp_path.write_text(modified_xml, encoding="utf-8")
        self._tmp_xml_path = tmp_path
        # 记录到所有临时文件列表
        if tmp_path not in self._all_tmp_xml_paths:
            self._all_tmp_xml_paths.append(tmp_path)
        return tmp_path

    def _build_highlighted_xml_string(self) -> str:
        """读取原XML并注入rgba以实现高亮与半透明效果。"""
        tree = ET.parse(str(self.model_path))
        root = tree.getroot()

        # 收集高亮颜色映射（动态扩展颜色）
        name_to_color: Dict[str, Tuple[float, float, float, float]] = {}
        if self._highlight_names:
            colors = self._generate_distinct_colors(len(self._highlight_names))
            for name, color in zip(self._highlight_names, colors):
                name_to_color[name] = color
        # 记录映射供导出API使用
        self._last_mapping_rgba = dict(name_to_color)

        # 遍历geom，按 mesh 名设置rgba
        for geom in root.iter("geom"):
            if geom.get("type") != "mesh":
                continue
            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue

            if mesh_name in name_to_color:
                r, g, b, a = name_to_color[mesh_name]
                geom.set("rgba", f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}")
            else:
                # 非高亮：统一为半透明中性灰（或保留原rgb，仅降低alpha）。
                rgba_attr = geom.get("rgba")
                if rgba_attr:
                    parts = [p for p in rgba_attr.strip().split() if p]
                    # 期望格式4元；否则回退为灰
                    if len(parts) == 4:
                        try:
                            r = float(parts[0])
                            g = float(parts[1])
                            b = float(parts[2])
                            geom.set("rgba", f"{r:.3f} {g:.3f} {b:.3f} {self.default_other_alpha:.3f}")
                            continue
                        except Exception:
                            pass
                # 回退：固定灰色半透
                geom.set("rgba", f"0.700 0.700 0.700 {self.default_other_alpha:.3f}")

        # 返回字符串
        return ET.tostring(root, encoding="unicode")

    # ----------------------------- 颜色生成与导出 -----------------------------
    def _generate_distinct_colors(self, n: int) -> List[Tuple[float, float, float, float]]:
        """生成 n 个高对比度颜色，返回 RGBA(0..1)。

        策略：优先使用基础调色板；若不足，按色轮等分生成HSL颜色（高饱和、高亮度）。
        """
        base = list(self.highlight_palette)
        if n <= len(base):
            return base[:n]

        colors = list(base)
        remain = n - len(base)
        # 通过等分色相生成额外颜色
        for k in range(remain):
            h = (k / max(1, remain)) % 1.0
            s = 0.85
            l = 0.55
            r, g, b = self._hsl_to_rgb(h, s, l)
            colors.append((r, g, b, 1.0))
        return colors

    @staticmethod
    def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
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
            r = g = b = l
            return r, g, b
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        return r, g, b

    def get_highlight_color_mapping(self, fmt: str = "rgba") -> Dict[str, Union[Tuple[float, float, float, float], Tuple[int, int, int], str]]:
        """获取最近一次设置的高亮“零件名→颜色”映射。

        Args:
            fmt: 输出格式：
                - "rgba": 返回 (r,g,b,a) 浮点0..1
                - "rgb": 返回 (R,G,B) 0..255 整数
                - "hex": 返回 "#RRGGBB" 字符串
        """
        mapping = dict(self._last_mapping_rgba)
        if fmt == "rgba":
            return mapping
        if fmt == "rgb":
            return {
                k: (int(round(v[0] * 255)), int(round(v[1] * 255)), int(round(v[2] * 255)))
                for k, v in mapping.items()
            }
        if fmt == "hex":
            def to_hex(v: Tuple[float, float, float, float]) -> str:
                r = int(round(v[0] * 255))
                g = int(round(v[1] * 255))
                b = int(round(v[2] * 255))
                return f"#{r:02X}{g:02X}{b:02X}"
            return {k: to_hex(v) for k, v in mapping.items()}
        raise ValueError("fmt 需为 'rgba' | 'rgb' | 'hex'")
    
    def cleanup_temp_files(self) -> None:
        """清理所有生成的临时XML文件"""
        cleaned_count = 0
        for tmp_path in self._all_tmp_xml_paths:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                    cleaned_count += 1
                except Exception as e:
                    # 忽略删除失败（可能文件已被其他进程删除）
                    pass
        
        # 清空列表
        self._all_tmp_xml_paths.clear()
        self._tmp_xml_path = None
        
        if cleaned_count > 0:
            # 可选：打印清理信息（仅在调试时使用）
            pass
    
    def __del__(self):
        """析构函数：确保临时文件被清理"""
        try:
            self.cleanup_temp_files()
        except Exception:
            # 忽略析构时的异常
            pass


__all__ = [
    "MujocoRenderController",
]


