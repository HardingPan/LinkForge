"""
渲染编排智能体
负责多线程渲染 overall 渲染图和所有 part 的高亮渲染图，并保存到记忆中
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils.render_controller import MujocoRenderController
from .memory import MemoryManager


class RenderOrchestrator:
    """渲染编排智能体
    
    功能：
    1. 接收 MJCF XML 文档路径
    2. 多线程渲染 overall 渲染图
    3. 多线程渲染所有 part 的高亮渲染图
    4. 将渲染结果保存到记忆中
    """
    
    def __init__(
        self,
        memory_storage_path: str = "./scene_memory"
    ):
        """初始化渲染编排智能体
        
        Args:
            memory_storage_path: 记忆存储路径，用于保存渲染图像和记忆记录
        """
        self.memory_path = memory_storage_path
        self.memory = MemoryManager(memory_storage_path)
        
        # 渲染配置
        self.render_options = {
            "num_views": 9,  # 3x3 马赛克
            "mosaic": True,
            "save": True,
            "image_quality": "medium"
        }
        
        # 存储渲染结果
        self.overall_image_path: Optional[str] = None
        self.part_images: Dict[str, str] = {}  # part_name -> image_path
    
    def orchestrate_rendering(self, xml_path: str, max_workers: int = 4, 
                            fast_mode: bool = False, clear_memory: bool = True) -> Dict[str, Any]:
        """编排渲染任务的主入口
        
        Args:
            xml_path: MJCF XML 文件路径
            max_workers: 最大线程数
            fast_mode: 是否使用快速模式（减少渲染视角）
            clear_memory: 是否在开始前清空记忆库和图片文件（默认True）
            
        Returns:
            包含渲染结果的字典，包括 overall 图像路径和所有 part 图像路径
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
            
            # 在任务开始前清空记忆库和图片文件
            if clear_memory:
                print("🧹 清空scene_memory文件夹...")
                memory_folder = Path(self.memory_path)
                memory_folder.mkdir(parents=True, exist_ok=True)
                
                # 清空所有图片文件
                deleted_count = 0
                for file in memory_folder.glob("*.png"):
                    try:
                        file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"  删除图片文件失败: {file.name}, 错误: {e}")
                
                # 清空记忆库 - 直接删除memories.json文件，确保彻底清空
                memory_file = memory_folder / "memories.json"
                if memory_file.exists():
                    try:
                        memory_file.unlink()
                        print(f"  ✓ 已删除memories.json文件")
                    except Exception as e:
                        print(f"  删除memories.json失败: {e}")
                
                # 重新初始化记忆库（确保内存中的记录也被清空）
                self.memory = MemoryManager(self.memory_path)
                print(f"  ✓ 已清空 {deleted_count} 个图片文件和所有记忆")
            
            # 获取所有 part 名称
            all_parts = self._get_all_parts_from_xml(xml_path)
            if not all_parts:
                return {
                    "success": False,
                    "message": "未找到任何part",
                    "error_details": "XML文件中没有有效的part"
                }
            
            print(f"🎨 渲染编排: {len(all_parts)} 个part，{max_workers} 线程")
            
            # 1. 渲染 overall 图像（单线程，优先级高）
            overall_result = self._render_overall(xml_path)
            if not overall_result.get("success"):
                return {
                    "success": False,
                    "message": "overall 渲染失败",
                    "error_details": overall_result.get("error", "未知错误")
                }
            
            self.overall_image_path = overall_result["image_path"]
            
            # 2. 多线程渲染所有 part 高亮图像
            part_results = self._render_parts_parallel(
                xml_path, all_parts, max_workers
            )
            
            # 统计成功和失败的part
            success_parts = [r["part_name"] for r in part_results if r.get("success")]
            failed_parts = [r["part_name"] for r in part_results if not r.get("success")]
            
            # 保存成功的part图像路径
            for result in part_results:
                if result.get("success"):
                    self.part_images[result["part_name"]] = result["image_path"]
            
            # 3. 存储渲染结果到记忆库（不再使用task_id）
            self._store_rendering_results(xml_path, overall_result, part_results)
            
            processing_time = time.time() - start_time
            
            print(f"✓ 渲染完成: overall + {len(success_parts)} 个part ({processing_time:.1f}秒)")
            if failed_parts:
                print(f"  ⚠ 失败: {len(failed_parts)} 个part - {failed_parts}")
            
            return {
                "success": True,
                "message": f"渲染编排完成，成功渲染 {len(success_parts)}/{len(all_parts)} 个part",
                "xml_path": xml_path,
                "overall_image_path": self.overall_image_path,
                "part_images": self.part_images,
                "success_parts": success_parts,
                "failed_parts": failed_parts,
                "processing_time": processing_time,
                "memory_path": self.memory_path
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "message": f"渲染编排失败: {str(e)}",
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
            
            return list(set(parts))  # 去重
        except Exception as e:
            print(f"解析XML文件失败: {e}")
            return []
    
    def _render_overall(self, xml_path: str) -> Dict[str, Any]:
        """渲染 overall 图像"""
        try:
            render_controller = MujocoRenderController(xml_path)
            
            # 不设置高亮，渲染原始场景
            memory_folder = Path(self.memory_path)
            memory_folder.mkdir(parents=True, exist_ok=True)
            # 使用更清晰的命名：overall_场景名.png
            xml_name = Path(xml_path).stem
            overall_path = f"{self.memory_path}/overall_{xml_name}.png"
            
            render_controller.render_original(
                num_views=self.render_options["num_views"],
                mosaic=self.render_options["mosaic"],
                save=self.render_options["save"],
                save_path=overall_path
            )
            
            return {
                "success": True,
                "image_path": overall_path,
                "image_type": "overall"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"overall渲染失败: {str(e)}"
            }
    
    def _render_parts_parallel(self, xml_path: str, all_parts: List[str], 
                              max_workers: int) -> List[Dict[str, Any]]:
        """并行渲染所有part的高亮图像"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_part = {
                executor.submit(self._render_single_part, xml_path, part_name): part_name
                for part_name in all_parts
            }
            
            # 收集结果
            for future in as_completed(future_to_part):
                part_name = future_to_part[future]
                try:
                    result = future.result()
                    if result.get("success"):
                        print(f"✓ 完成part渲染: {part_name}")
                    else:
                        print(f"✗ part渲染失败: {part_name} - {result.get('error', '未知错误')}")
                    results.append(result)
                except Exception as e:
                    print(f"✗ part {part_name} 渲染异常: {e}")
                    results.append({
                        "success": False,
                        "part_name": part_name,
                        "error": str(e)
                    })
        
        return results
    
    def _render_single_part(self, xml_path: str, part_name: str) -> Dict[str, Any]:
        """渲染单个part的高亮图像（用于并行渲染）"""
        try:
            # 为每个线程创建独立的渲染控制器
            render_controller = MujocoRenderController(xml_path)
            
            # 设置高亮
            render_controller.set_highlights([part_name])
            
            # 渲染高亮图像，使用更清晰的命名：highlighted_part名.png
            memory_folder = Path(self.memory_path)
            memory_folder.mkdir(parents=True, exist_ok=True)
            highlighted_path = f"{self.memory_path}/highlighted_{part_name}.png"
            
            render_controller.render(
                num_views=self.render_options["num_views"],
                mosaic=self.render_options["mosaic"],
                save=self.render_options["save"],
                save_path=highlighted_path
            )
            
            # 获取颜色映射
            color_mapping = render_controller.get_highlight_color_mapping(fmt="hex")
            
            # 清理临时XML文件
            render_controller.cleanup_temp_files()
            
            return {
                "success": True,
                "part_name": part_name,
                "image_path": highlighted_path,
                "image_type": "highlighted",
                "color_mapping": color_mapping
            }
            
        except Exception as e:
            # 即使渲染失败，也尝试清理临时文件
            try:
                if 'render_controller' in locals():
                    render_controller.cleanup_temp_files()
            except Exception:
                pass
            
            return {
                "success": False,
                "part_name": part_name,
                "error": f"part {part_name} 高亮渲染失败: {str(e)}"
            }
    
    def _store_rendering_results(self, xml_path: str, 
                                overall_result: Dict[str, Any],
                                part_results: List[Dict[str, Any]]) -> None:
        """存储渲染结果到记忆库（不再使用task_id）"""
        try:
            # 存储 overall 渲染结果
            overall_content = f"""
Overall渲染结果：
- 图像路径：{overall_result.get('image_path', '未知')}
- 图像类型：{overall_result.get('image_type', 'overall')}
"""
            overall_metadata = {
                "xml_path": xml_path,
                "image_path": overall_result.get("image_path", ""),
                "image_type": "overall",
                "rendering_type": "overall",
                "image_name": "overall",  # 用于检索
                "timestamp": time.time()
            }
            self.memory.store_long(overall_content, overall_metadata)
            
            # 存储每个part的渲染结果
            stored_count = 0
            for part_result in part_results:
                if part_result.get("success"):
                    part_name = part_result.get("part_name", "未知")
                    part_content = f"""
Part高亮渲染结果 - {part_name}：
- Part名称：{part_name}
- 图像路径：{part_result.get('image_path', '未知')}
- 图像类型：{part_result.get('image_type', 'highlighted')}
- 颜色映射：{part_result.get('color_mapping', {})}
"""
                    part_metadata = {
                        "xml_path": xml_path,
                        "part_name": part_name,
                        "image_path": part_result.get("image_path", ""),
                        "image_type": part_result.get("image_type", "highlighted"),
                        "rendering_type": "part_highlighted",
                        "image_name": f"highlighted_{part_name}",  # 用于检索
                        "color_mapping": part_result.get("color_mapping", {}),
                        "timestamp": time.time()
                    }
                    try:
                        self.memory.store_long(part_content, part_metadata)
                        stored_count += 1
                        print(f"    ✓ 已存储part '{part_name}' 的渲染结果到记忆")
                    except Exception as e:
                        print(f"    ✗ 存储part '{part_name}' 的渲染结果失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"渲染结果已存储到记忆库 (overall: 1, parts: {stored_count}/{len([r for r in part_results if r.get('success')])})")
            
        except Exception as e:
            print(f"存储渲染结果到记忆失败: {e}")
    
    def get_overall_image_path(self) -> Optional[str]:
        """获取 overall 图像路径"""
        return self.overall_image_path
    
    def get_part_image_path(self, part_name: str) -> Optional[str]:
        """获取指定part的图像路径"""
        return self.part_images.get(part_name)
    
    def get_all_part_images(self) -> Dict[str, str]:
        """获取所有part的图像路径"""
        return self.part_images.copy()
    
    def load_rendering_results_from_memory(self, verbose: bool = True) -> Dict[str, Any]:
        """从记忆库加载渲染结果（不再使用task_id，直接加载最新的）
        
        Args:
            verbose: 是否打印详细日志，默认True
        """
        try:
            # 查询 overall 渲染结果（选择最新的）
            overall_memories = self.memory.retrieve("Overall渲染结果", memory_type="long", limit=1000)
            overall_image_path = None
            xml_path = None
            
            if verbose:
                print(f"    🔍 查找渲染结果")
                print(f"      找到的overall记忆数量: {len(overall_memories)}")
            
            if overall_memories:
                # 选择最新的，并验证图片文件是否存在
                overall_memories.sort(key=lambda x: x.timestamp, reverse=True)
                for memory in overall_memories:
                    overall_image_path = memory.metadata.get("image_path")
                    if overall_image_path and Path(overall_image_path).exists():
                        xml_path = memory.metadata.get("xml_path")
                        if verbose:
                            print(f"      ✓ 找到overall渲染结果: {overall_image_path}")
                        break
                    else:
                        # 图片文件不存在，尝试查找overall图片（使用新的命名规则）
                        xml_path_from_memory = memory.metadata.get("xml_path")
                        if xml_path_from_memory:
                            xml_name = Path(xml_path_from_memory).stem
                            expected_path = Path(self.memory_path) / f"overall_{xml_name}.png"
                            if expected_path.exists():
                                overall_image_path = str(expected_path)
                                xml_path = xml_path_from_memory
                                if verbose:
                                    print(f"      💡 修复overall图片路径: {expected_path}")
                                break
                else:
                    # 所有记录都找不到有效的图片
                    overall_image_path = None
                    if verbose:
                        print(f"      ✗ 未找到有效的overall渲染结果")
            else:
                if verbose:
                    print(f"      ✗ 未找到overall渲染结果")
            
            # 如果没有找到xml_path，尝试从part渲染结果中获取
            if not xml_path:
                part_memories = self.memory.retrieve("Part高亮渲染结果", memory_type="long", limit=1000)
                if part_memories:
                    part_memories.sort(key=lambda x: x.timestamp, reverse=True)
                    xml_path = part_memories[0].metadata.get("xml_path")
            
            # 查询所有part渲染结果（选择最新的）
            part_memories = self.memory.retrieve("Part高亮渲染结果", memory_type="long", limit=1000)
            part_images = {}
            if verbose:
                print(f"      找到的part记忆数量: {len(part_memories)}")
            
            # 按part_name分组，选择每个part最新的渲染结果，并验证图片文件是否存在
            part_dict = {}
            for memory in part_memories:
                part_name = memory.metadata.get("part_name")
                image_path = memory.metadata.get("image_path")
                if part_name and image_path:
                    # 验证图片文件是否存在
                    if Path(image_path).exists():
                        if part_name not in part_dict or memory.timestamp > part_dict[part_name].timestamp:
                            part_dict[part_name] = memory
                    else:
                        # 图片文件不存在，尝试通过part_name直接查找图片
                        # 使用新的命名规则：highlighted_{part_name}.png
                        expected_path = Path(self.memory_path) / f"highlighted_{part_name}.png"
                        if expected_path.exists():
                            # 更新记忆中的图片路径
                            memory.metadata["image_path"] = str(expected_path)
                            if part_name not in part_dict or memory.timestamp > part_dict[part_name].timestamp:
                                part_dict[part_name] = memory
                                if verbose:
                                    print(f"      💡 修复part '{part_name}' 的图片路径: {expected_path}")
            
            # 如果记忆中没有找到某些part，直接扫描文件系统中的图片文件
            memory_folder = Path(self.memory_path)
            for highlighted_file in memory_folder.glob("highlighted_*.png"):
                # 从文件名提取part_name: highlighted_{part_name}.png
                part_name_from_file = highlighted_file.stem.replace("highlighted_", "")
                if part_name_from_file not in part_dict:
                    # 记忆中没有这个part的记录，但文件存在，直接使用
                    part_images[part_name_from_file] = str(highlighted_file)
                    if verbose:
                        print(f"      💡 从文件系统找到part '{part_name_from_file}' 的图片: {highlighted_file}")
            
            # 从记忆中加载的part
            for part_name, memory in part_dict.items():
                image_path = memory.metadata.get("image_path")
                if image_path and Path(image_path).exists():
                    part_images[part_name] = image_path
                    if verbose:
                        print(f"      ✓ 找到part '{part_name}' 的渲染结果: {image_path}")
                else:
                    if verbose:
                        print(f"      ✗ part '{part_name}' 的图片路径不存在: {image_path}")
            
            if verbose:
                print(f"      最终加载结果: overall={overall_image_path is not None}, parts={len(part_images)}")
            else:
                # 非详细模式下，只打印摘要
                print(f"  ✓ 已加载渲染结果: overall={overall_image_path is not None}, parts={len(part_images)}")
            
            return {
                "overall_image_path": overall_image_path,
                "part_images": part_images,
                "xml_path": xml_path
            }
            
        except Exception as e:
            print(f"从记忆库加载渲染结果失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "overall_image_path": None,
                "part_images": {},
                "xml_path": None
            }
    
    def get_part_image_path(self, part_name: str, image_type: str = "highlighted") -> Optional[str]:
        """根据part名字和类型获取图片路径
        
        Args:
            part_name: part名称
            image_type: 图片类型 ("overall", "highlighted", "visualization")
            
        Returns:
            图片路径，如果未找到则返回None
        """
        try:
            # 首先尝试从记忆中查找
            if image_type == "overall":
                memories = self.memory.retrieve("Overall渲染结果", memory_type="long", limit=10)
                # 选择最新的，并验证文件是否存在
                for memory in sorted(memories, key=lambda x: x.timestamp, reverse=True):
                    image_path = memory.metadata.get("image_path")
                    if image_path and Path(image_path).exists():
                        return image_path
                # 如果记忆中找不到，尝试直接查找文件（使用新的命名规则）
                # 查找所有overall_*.png文件
                memory_folder = Path(self.memory_path)
                overall_files = list(memory_folder.glob("overall_*.png"))
                if overall_files:
                    # 选择最新的
                    overall_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return str(overall_files[0])
                    
            elif image_type == "highlighted":
                memories = self.memory.retrieve(f"Part高亮渲染结果 - {part_name}", memory_type="long", limit=10)
                # 选择最新的，并验证文件是否存在
                for memory in sorted(memories, key=lambda x: x.timestamp, reverse=True):
                    image_path = memory.metadata.get("image_path")
                    if image_path and Path(image_path).exists():
                        return image_path
                # 如果记忆中找不到，尝试直接查找文件（使用新的命名规则）
                expected_path = Path(self.memory_path) / f"highlighted_{part_name}.png"
                if expected_path.exists():
                    return str(expected_path)
                    
            elif image_type == "visualization":
                memories = self.memory.retrieve(f"运动轴可视化结果 - {part_name}", memory_type="long", limit=10)
                # 选择最新的，并验证文件是否存在
                for memory in sorted(memories, key=lambda x: x.timestamp, reverse=True):
                    image_path = memory.metadata.get("visualization_path")
                    if image_path and Path(image_path).exists():
                        return image_path
                # 如果记忆中找不到，尝试直接查找文件
                memory_folder = Path(self.memory_path)
                # 查找包含part_name的可视化文件
                viz_files = list(memory_folder.glob(f"*{part_name}*visualization*.png"))
                if viz_files:
                    # 选择最新的
                    viz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return str(viz_files[0])
            else:
                return None
            
            return None
        except Exception as e:
            print(f"获取part图片路径失败: {e}")
            import traceback
            traceback.print_exc()
            return None


__all__ = [
    "RenderOrchestrator",
]

