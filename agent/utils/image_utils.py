import base64
import io
import mimetypes
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
from PIL import Image

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


def build_image_url(image_input) -> str:
    """Normalize various image inputs to a URL string that can be used in LLM vision messages.

    Automatically detects input type and converts to appropriate URL format.
    
    Accepted inputs:
    - numpy array (H, W, 3): treated as RGB image array
    - bytes: treated as PNG image bytes
    - str: can be
        * data URL (data:...)
        * http(s) URL
        * base64 string (without data: prefix)
        * local file path
    - Path: local file path

    Returns a string URL:
    - If remote URL: returns as-is
    - If local/bytes/base64/array: returns data URL (image/png)
    """

    if image_input is None:
        raise ValueError("图像输入不能为空")

    # numpy array (RGB)
    if isinstance(image_input, np.ndarray):
        return _array_to_data_url(image_input)

    # bytes
    if isinstance(image_input, bytes):
        return _bytes_to_data_url(image_input)

    # string
    if isinstance(image_input, str):
        if image_input.startswith("data:"):
            return image_input
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return image_input
        # treat as base64 or file path
        if len(image_input) > 100 and not Path(image_input).exists():
            # likely base64 string
            return _base64_to_data_url(image_input)
        # treat as file path
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"未找到本地图片: {image_input}")
        return _file_to_data_url(image_path)

    # Path object
    if isinstance(image_input, Path):
        if not image_path.exists():
            raise FileNotFoundError(f"未找到本地图片: {image_input}")
        return _file_to_data_url(image_input)

    raise TypeError(f"不支持的图像输入类型: {type(image_input)}")


def _array_to_data_url(img_array: "np.ndarray") -> str:
    """将numpy数组转换为data URL"""
    if img_array is None:
        raise ValueError("图像数组不能为空")
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        raise ValueError("图像数组形状应为 (H, W, 3)")
    
    # 直接使用RGB格式
    image = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _base64_to_data_url(b64_payload: str, mime_type: str = "image/png") -> str:
    if not b64_payload:
        raise ValueError("空的 base64 字符串")
    return f"data:{mime_type};base64,{b64_payload}"


def _bytes_to_data_url(img_bytes: bytes, mime_type: str = "image/png") -> str:
    if not img_bytes:
        raise ValueError("空的字节数据")
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _file_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _compute_fit_view_cameras(model: "mujoco.MjModel",
                             views: Optional[List[str]] = None,
                             distance_scale: float = 1.6) -> List["mujoco.MjvCamera"]:
    """基于模型统计信息生成多个适配视野(fit view)的自由相机。

    Args:
        model: 已加载的 MuJoCo 模型
        views: 视角列表，支持 ["front", "back", "left", "right", "iso", "top"]。
               默认为 ["front", "right", "back", "left"] 四视角。
        distance_scale: 相机距离系数，基于 model.stat.extent 放缩

    Returns:
        一组配置好的 `mujoco.MjvCamera` 对象
    """
    if views is None:
        views = ["front", "right", "back", "left"]

    center = np.array(getattr(model.stat, "center", np.zeros(3)))
    extent = float(getattr(model.stat, "extent", 1.0))
    distance = max(1e-6, distance_scale * extent)

    name_to_angles = {
        # azimuth(水平绕 z 旋转，度)，elevation(俯仰，度；负值俯视)
        "front": (0.0, -20.0),
        "back": (180.0, -20.0),
        "left": (90.0, -20.0),
        "right": (-90.0, -20.0),
        "iso": (45.0, -30.0),
        "top": (0.0, -90.0),
        "front_right": (-45.0, -20.0),
        "back_left": (135.0, -20.0),
        "front_left": (45.0, -20.0),
    }

    cameras: List[mujoco.MjvCamera] = []
    for view in views:
        az, el = name_to_angles.get(view, (0.0, -20.0))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = center
        cam.distance = distance
        cam.azimuth = az
        cam.elevation = el
        cameras.append(cam)

    return cameras


def capture_mujoco_scene(model_path: Union[str, Path], 
                        width: int = 800, 
                        height: int = 600,
                        camera_name: Optional[str] = None,
                        qpos: Optional[np.ndarray] = None,
                        auto_resize: bool = True) -> np.ndarray:
    """从MuJoCo场景中捕获截图并返回RGB图像数组。
    
    Args:
        model_path: MuJoCo模型文件路径 (.xml)
        width: 图像宽度 (默认: 800)
        height: 图像高度 (默认: 600) 
        camera_name: 相机名称，如果为None则使用默认相机
        qpos: 关节位置，如果为None则使用默认位置
        auto_resize: 是否自动调整尺寸以适应帧缓冲区限制 (默认: True)
        
    Returns:
        RGB图像数组 (height, width, 3)，数据类型为uint8
        
    Raises:
        ImportError: 如果MuJoCo未安装
        FileNotFoundError: 如果模型文件不存在
        RuntimeError: 如果渲染失败
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo未安装，请运行: pip install mujoco mujoco-python-viewer")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo模型文件不存在: {model_path}")
    
    renderer = None
    try:
        # 加载模型
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        # 检查帧缓冲区限制
        offwidth = getattr(model.vis.global_, 'offwidth', 640)
        offheight = getattr(model.vis.global_, 'offheight', 480)
        
        # 使用限制内的尺寸
        actual_width = min(width, offwidth)
        actual_height = min(height, offheight)
        
        if actual_width != width or actual_height != height:
            print(f"警告: 请求尺寸 {width}x{height} 超过帧缓冲区限制 {offwidth}x{offheight}")
            print(f"使用尺寸: {actual_width}x{actual_height}")
        
        # 创建渲染器
        renderer = mujoco.Renderer(model, height=actual_height, width=actual_width)
        
        # 渲染场景
        # 优先使用 XML 中的命名相机；否则使用自动生成的 fit-view 自由相机(默认 iso)
        camera_arg: Union[int, "mujoco.MjvCamera"]
        use_fit_view = False
        if camera_name:
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            except Exception:
                cam_id = -1
            if cam_id >= 0:
                camera_arg = cam_id
            else:
                use_fit_view = True
        else:
            use_fit_view = True

        if use_fit_view:
            fit_cams = _compute_fit_view_cameras(model, views=["iso"])  # 默认 iso
            camera_arg = fit_cams[0]

        renderer.update_scene(data, camera=camera_arg)
        rgb_image = renderer.render()
        
        # 增强图像亮度和对比度
        original_max = rgb_image.max()
        if original_max > 0:
            rgb_image = rgb_image.astype(np.float32)
            rgb_image = (rgb_image / original_max) * 255
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            print(f"图像增强: 原始范围 0-{original_max} -> 增强后范围 {rgb_image.min()}-{rgb_image.max()}")
        else:
            print("警告: 图像全黑，无法增强")
        
        # 如果需要，将图像调整到原始请求的尺寸
        if actual_width != width or actual_height != height:
            pil_image = Image.fromarray(rgb_image)
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            rgb_image = np.array(pil_image)
        
        return rgb_image
        
    except Exception as e:
        raise RuntimeError(f"MuJoCo场景渲染失败: {e}")
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


def capture_mujoco_scene_to_data_url(model_path: Union[str, Path],
                                   width: int = 800,
                                   height: int = 600, 
                                   camera_name: Optional[str] = None,
                                   qpos: Optional[np.ndarray] = None,
                                   auto_resize: bool = True,
                                   save_image: bool = False,
                                   save_path: Optional[str] = None,
                                   num_views: int = 1,
                                   views: Optional[List[str]] = None,
                                   tile_shape: Optional[Tuple[int, int]] = None,
                                   pad: int = 4) -> str:
    """从MuJoCo场景捕获图像，单/多视角统一入口，返回 data URL。
    
    - 当 `num_views>1` 或 `views` 的长度>1 时：生成多视角马赛克并返回其 data URL；
    - 否则：返回单张视角图的 data URL。
    
    Args:
        model_path: MuJoCo模型文件路径 (.xml)
        width: 子图宽度 (默认: 800)
        height: 子图高度 (默认: 600)
        camera_name: 单视角时可选固定相机名；否则使用自动 fit-view  
        qpos: 关节位置，如果为None则使用默认位置
        auto_resize: 是否自动调整尺寸以适应帧缓冲区限制 (默认: True)
        save_image: 是否保存图像到本地 (默认: False)
        save_path: 保存路径，如果为None则自动生成文件名
        num_views: 视角数量（默认1）；仅在 `views` 未指定时生效
        views: 指定视角名称列表（front/right/back/left/iso/top）；长度>1触发多视角
        tile_shape: 多视角马赛克网格 (rows, cols)，默认自动
        pad: 多视角马赛克间隔像素
        
    Returns:
        data URL字符串，可直接用于LLM API
        
    Raises:
        ImportError: 如果MuJoCo未安装
        FileNotFoundError: 如果模型文件不存在
        RuntimeError: 如果渲染失败
    """
    # 判断是否多视角
    view_count = 0
    if views is not None:
        view_count = len(views)
    else:
        view_count = num_views if isinstance(num_views, int) else 1

    if view_count and view_count > 1:
        # 多视角：生成马赛克
        mosaic = capture_mujoco_multiview_mosaic(
            model_path=model_path,
            width=width,
            height=height,
            num_views=num_views,
            views=views,
            distance_scale=1.6,
            auto_resize=auto_resize,
            save_image=save_image,
            save_path=save_path,
            tile_shape=tile_shape,
            pad=pad,
        )
        return _array_to_data_url(mosaic)

    # 单视角：沿用原逻辑
    rgb_image = capture_mujoco_scene(
        model_path=model_path,
        width=width, 
        height=height,
        camera_name=camera_name,
        qpos=qpos,
        auto_resize=auto_resize
    )
    
    if save_image:
        if save_path is None:
            model_name = Path(model_path).stem
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"mujoco_screenshot_{model_name}_{timestamp}.png"
        Image.fromarray(rgb_image).save(save_path)
        print(f"图像已保存到: {save_path}")
    
    return _array_to_data_url(rgb_image)


def list_mujoco_cameras(model_path: Union[str, Path]) -> list:
    """列出MuJoCo模型中的所有相机。
    
    Args:
        model_path: MuJoCo模型文件路径 (.xml)
        
    Returns:
        相机名称列表
        
    Raises:
        ImportError: 如果MuJoCo未安装
        FileNotFoundError: 如果模型文件不存在
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo未安装，请运行: pip install mujoco mujoco-python-viewer")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo模型文件不存在: {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        cameras = []
        for i in range(model.ncam):
            cameras.append(model.cam(i).name)
        return cameras
    except Exception as e:
        raise RuntimeError(f"读取MuJoCo模型失败: {e}")


def capture_mujoco_multiview_to_data_urls(model_path: Union[str, Path],
                                          width: int = 800,
                                          height: int = 600,
                                          num_views: int = 4,
                                          views: Optional[List[str]] = None,
                                          distance_scale: float = 1.6,
                                          auto_resize: bool = True,
                                          save_images: bool = False,
                                          save_prefix: Optional[str] = None) -> List[str]:
    """基于模型自动生成多视角(默认4)并执行 fit view 渲染，返回 data URL 列表。

    Args:
        model_path: MuJoCo XML 路径
        width: 目标宽度
        height: 目标高度
        num_views: 视角数量，若提供 views 列表则忽略此值
        views: 指定视角名称列表 ["front","right","back","left","iso","top"]
        distance_scale: 相机距离缩放系数，基于 model.stat.extent
        auto_resize: 若实际渲染尺寸受帧缓冲限制，是否回缩后再放大至目标尺寸
        save_images: 是否保存 PNG 文件
        save_prefix: 保存文件名前缀；默认基于模型名与时间戳

    Returns:
        data URL 字符串列表
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo未安装，请运行: pip install mujoco mujoco-python-viewer")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo模型文件不存在: {model_path}")

    renderer = None
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        offwidth = getattr(model.vis.global_, 'offwidth', 640)
        offheight = getattr(model.vis.global_, 'offheight', 480)
        actual_width = min(width, offwidth)
        actual_height = min(height, offheight)
        if (actual_width, actual_height) != (width, height):
            print(f"警告: 请求尺寸 {width}x{height} 超过帧缓冲区限制 {offwidth}x{offheight}")
            print(f"使用尺寸: {actual_width}x{actual_height}")

        renderer = mujoco.Renderer(model, height=actual_height, width=actual_width)

        if views is None:
            default_views = ["front", "right", "back", "left"]
            if isinstance(num_views, int) and num_views <= 0:
                num_views = 4
            views = default_views[:num_views] if num_views <= len(default_views) else default_views

        cameras = _compute_fit_view_cameras(model, views=views, distance_scale=distance_scale)

        data_urls: List[str] = []
        base_name = model_path.stem
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, (view_name, cam) in enumerate(zip(views, cameras)):
            renderer.update_scene(data, camera=cam)
            rgb_image = renderer.render()

            original_max = rgb_image.max()
            if original_max > 0:
                rgb_image = rgb_image.astype(np.float32)
                rgb_image = (rgb_image / original_max) * 255
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

            if (actual_width, actual_height) != (width, height) and auto_resize:
                pil_image = Image.fromarray(rgb_image)
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                rgb_image = np.array(pil_image)

            if save_images:
                prefix = save_prefix or f"mujoco_multiview_{base_name}_{timestamp}"
                out_path = f"{prefix}_{idx:02d}_{view_name}.png"
                Image.fromarray(rgb_image).save(out_path)
                print(f"图像已保存到: {out_path}")

            data_urls.append(_array_to_data_url(rgb_image))

        return data_urls
    except Exception as e:
        raise RuntimeError(f"MuJoCo多视角渲染失败: {e}")
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


def stitch_images_grid(image_arrays: List[np.ndarray],
                       tile_shape: Optional[Tuple[int, int]] = None,
                       pad: int = 4,
                       pad_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """将多张等尺寸 RGB 图像拼接为网格图。

    Args:
        image_arrays: 多张等尺寸(H, W, 3)图像
        tile_shape: (rows, cols)，不提供时尽量接近方阵布局
        pad: 单元之间与边界的像素间隔
        pad_color: 间隔颜色

    Returns:
        拼接后的 RGB 图像
    """
    if not image_arrays:
        raise ValueError("image_arrays 不能为空")
    h, w, c = image_arrays[0].shape
    if any(img.shape != (h, w, c) for img in image_arrays):
        raise ValueError("所有图像尺寸必须相同")

    n = len(image_arrays)
    if tile_shape is None:
        # 接近正方形布局
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = tile_shape

    grid_h = rows * h + (rows + 1) * pad
    grid_w = cols * w + (cols + 1) * pad
    canvas = np.full((grid_h, grid_w, 3), np.array(pad_color, dtype=np.uint8), dtype=np.uint8)

    for idx, img in enumerate(image_arrays):
        r = idx // cols
        cidx = idx % cols
        if r >= rows:
            break
        y0 = pad + r * (h + pad)
        x0 = pad + cidx * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w, :] = img

    return canvas


def capture_mujoco_multiview_mosaic(model_path: Union[str, Path],
                                    width: int = 800,
                                    height: int = 600,
                                    num_views: int = 4,
                                    views: Optional[List[str]] = None,
                                    distance_scale: float = 1.6,
                                    auto_resize: bool = True,
                                    save_image: bool = False,
                                    save_path: Optional[str] = None,
                                    tile_shape: Optional[Tuple[int, int]] = None,
                                    pad: int = 4) -> np.ndarray:
    """生成多视角并拼接为一张马赛克图，返回 RGB 数组。

    Args 参见 capture_mujoco_multiview_to_data_urls，新增：
        tile_shape: 指定网格(rows, cols)，默认自适应
        pad: 单元间间隔像素
    """
    urls = capture_mujoco_multiview_to_data_urls(
        model_path=model_path,
        width=width,
        height=height,
        num_views=num_views,
        views=views,
        distance_scale=distance_scale,
        auto_resize=auto_resize,
        save_images=False,
    )

    # 将 data URL 还原为数组
    imgs: List[np.ndarray] = []
    for u in urls:
        if not u.startswith("data:image"):
            raise ValueError("期望 data URL 格式")
        b64 = u.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        pil = Image.open(io.BytesIO(arr)).convert("RGB")
        imgs.append(np.array(pil))

    mosaic = stitch_images_grid(imgs, tile_shape=tile_shape, pad=pad)

    if save_image:
        if save_path is None:
            model_name = Path(model_path).stem
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"mujoco_multiview_mosaic_{model_name}_{timestamp}.png"
        Image.fromarray(mosaic).save(save_path)
        print(f"拼接图像已保存到: {save_path}")

    return mosaic


def get_mujoco_framebuffer_info(model_path: Union[str, Path]) -> dict:
    """获取MuJoCo模型的帧缓冲区信息。
    
    Args:
        model_path: MuJoCo模型文件路径 (.xml)
        
    Returns:
        包含帧缓冲区信息的字典，包括:
        - offwidth: 离屏宽度
        - offheight: 离屏高度
        - max_width: 最大支持宽度
        - max_height: 最大支持高度
        
    Raises:
        ImportError: 如果MuJoCo未安装
        FileNotFoundError: 如果模型文件不存在
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo未安装，请运行: pip install mujoco mujoco-python-viewer")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo模型文件不存在: {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        
        offwidth = getattr(model.vis.global_, 'offwidth', 640)
        offheight = getattr(model.vis.global_, 'offheight', 480)
        
        return {
            'offwidth': offwidth,
            'offheight': offheight,
            'max_width': offwidth,
            'max_height': offheight,
            'recommended_width': min(800, offwidth),
            'recommended_height': min(600, offheight)
        }
    except Exception as e:
        raise RuntimeError(f"读取MuJoCo模型失败: {e}")


__all__ = [
    "build_image_url",
    "capture_mujoco_scene",
    "capture_mujoco_scene_to_data_url",
    "capture_mujoco_multiview_to_data_urls",
    "capture_mujoco_multiview_mosaic",
    "stitch_images_grid",
    "list_mujoco_cameras",
    "get_mujoco_framebuffer_info",
]