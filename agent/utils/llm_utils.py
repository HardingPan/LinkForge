from typing import Optional, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from .image_utils import build_image_url

from .env_utils import load_environment, get_env

def build_llm(api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o", temperature: Optional[float] = 0.0):
    """Build a chat model instance with environment variable support.
    
    Args:
        api_key: Optional API key (overrides environment variables)
        base_url: Optional base URL (overrides environment variables) 
        model: Model name (default: gpt-4o)
        temperature: Temperature for the model (default: 0.0)
    
    Returns:
        Initialized chat model instance
    """
    # 先加载 .env
    load_environment()

    api_key = api_key or get_env("QWEN_API_KEY") or get_env("OPENAI_API_KEY")
    base_url = base_url or get_env("QWEN_BASE_URL") or get_env("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("未找到 API Key，请设置环境变量 QWEN_API_KEY 或 OPENAI_API_KEY")
    if not base_url:
        raise RuntimeError("未找到 Base URL，请设置环境变量 QWEN_BASE_URL 或 OPENAI_BASE_URL")

    # 使用 OpenAI 兼容接口 (OneAPI)；LangChain 将走 /v1/chat/completions
    # 确保完全确定性：temperature=0, top_p=1.0
    # 注意：同时设置顶层参数和model_kwargs以确保参数正确传递
    llm = init_chat_model(
        model=model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,  # 顶层参数
        timeout=120,  # 增加到2分钟
        max_retries=2,
    )
    
    # 使用 bind() 方法确保每次调用都使用确定的参数
    # 这样可以覆盖任何默认值或环境变量设置
    llm = llm.bind(
        temperature=temperature,
        top_p=1.0,  # 确保完全确定性（top_p=1.0 表示选择所有token，与temperature=0配合使用）
    )
    
    return llm

def describe_image(llm, 
                   image_input,
                   instruction: str = "请用中文详细描述这张图的内容、场景与关键要素。") -> str:
    """Describe an image using a vision-enabled chat model via LangChain.

    Args:
        llm: 语言模型实例
        image_input: 图像输入，支持多种格式：
            - numpy 数组 (BGR)
            - bytes (PNG 字节)
            - str (base64 字符串、data URL、http URL、本地文件路径)
            - Path (本地文件路径)
        instruction: 对图像的任务指令。
    Returns:
        模型返回的中文描述文本。
    """
    # 统一用工具函数适配输入
    image_url = build_image_url(image_input)

    # OpenAI 兼容多模态消息格式
    messages = [
        SystemMessage(content="你是一个专业的多模态视觉助手，请用准确、清晰的中文回答。"),
        HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        ),
    ]

    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


def describe_multiple_images(llm,
                            image_inputs: List,
                            instruction: str = "请用中文详细描述这些图像的内容、场景与关键要素，并分析它们之间的关系。") -> str:
    """Describe multiple images using a vision-enabled chat model via LangChain.

    Args:
        llm: 语言模型实例
        image_inputs: 图像输入列表，每个元素支持多种格式：
            - numpy 数组 (BGR)
            - bytes (PNG 字节)
            - str (base64 字符串、data URL、http URL、本地文件路径)
            - Path (本地文件路径)
        instruction: 对图像的任务指令。
    Returns:
        模型返回的中文描述文本。
    """
    # 统一用工具函数适配所有输入
    image_urls = [build_image_url(img) for img in image_inputs]

    # 构建消息内容：先文本，后所有图像
    content = [{"type": "text", "text": instruction}]
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    # OpenAI 兼容多模态消息格式
    messages = [
        SystemMessage(content="你是一个专业的多模态视觉助手，请用准确、清晰的中文回答。"),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


def build_multiview_instruction(base_instruction: str,
                                views: Optional[List[str]] = None) -> str:
    """根据视角数量自动增强提示词，例如 4/9 视角说明。
    
    Args:
        base_instruction: 基础任务指令
        views: 视角名称列表；若为 None 则不添加视角名，仅提示数量
    Returns:
        增强后的中文提示词
    """
    if views and len(views) > 0:
        view_count = len(views)
        view_list = "、".join(views)
        suffix = f"注意：这是一张由 {view_count} 个视角拼接的图像，视角包括：{view_list}。请分别参考各子图进行综合描述，注意区分不同视角的内容。"
    else:
        suffix = "注意：这是一张由多视角拼接的图像，请留意子图数量并综合分析。"
    return f"{base_instruction}\n\n{suffix}"


__all__ = ["build_llm", "describe_image", "describe_multiple_images", "build_multiview_instruction"]
