"""
用户提示解析工具
负责解析用户的自然语言提示，通过视觉比较找到匹配的部件并提取约束信息
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import xml.etree.ElementTree as ET

from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from .llm_utils import build_llm, describe_multiple_images
from .data_models import UserHintParsedResult
from .prompt_templates import build_user_hint_parsing_prompt


def get_parts_from_xml(xml_path: Path) -> List[str]:
    """从XML文件中提取所有part名称
    
    Args:
        xml_path: XML文件路径
        
    Returns:
        部件名称列表
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        parts = []
        
        # 查找所有body元素
        for body in root.findall(".//body"):
            body_name = body.get("name")
            if body_name:
                parts.append(body_name)
        
        return parts
    except Exception as e:
        print(f"  ⚠ 读取XML文件失败: {e}")
        return []


def parse_user_hint_with_visual_comparison(
    user_hint: str, 
    part_images: Dict[str, str], 
    overall_image_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """使用LLM通过视觉比较解析用户自然语言提示
    
    Args:
        user_hint: 用户的自然语言提示
        part_images: 部件名称到图片路径的字典，格式：{part_name: image_path}
        overall_image_path: 整体场景图片路径（可选）
        
    Returns:
        解析后的提示字典，格式：{part_name: {motion_type, sliding_direction, rotation_type, motion_range}}
    """
    try:
        llm = build_llm()
        parser = PydanticOutputParser(pydantic_object=UserHintParsedResult)
        
        # 构建所有部件的高亮图片列表
        part_image_list = []
        part_names_list = []
        for part_name, image_path in part_images.items():
            if Path(image_path).exists():
                part_image_list.append(image_path)
                part_names_list.append(part_name)
        
        if not part_image_list:
            print(f"  ⚠ 没有可用的部件高亮图片")
            return {}
        
        # 构建prompt
        parts_info = "\n".join([f"- {i+1}. {name}" for i, name in enumerate(part_names_list)])
        
        prompt = f"""请分析用户提示，并通过视觉比较所有部件的高亮图片，找到最符合描述的部件。

用户提示："{user_hint}"

可用的部件列表（按图片顺序）：
{parts_info}

任务：
1. **视觉匹配**：仔细查看所有部件的高亮图片，找到最符合用户描述的部件
   - 如果用户提到"按钮"、"按键"，找到看起来像按钮的部件
   - 如果用户提到"盖子"、"盖"，找到看起来像盖子的部件
   - 如果用户提到"门"，找到看起来像门的部件
   - 如果用户提到"抽屉"，找到看起来像抽屉的部件
2. **运动类型识别**：
   - "fixed": 固定部件，不运动
   - "sliding": 滑动运动（如抽屉、推拉门、按钮按下等）
   - "rotating": 旋转运动（如门、盖子、旋钮等）
3. **滑动方向**（仅当motion_type=sliding时）：
   - "x": 左右滑动
   - "y": 前后滑动
   - "z": 上下滑动（如按钮按下）
4. **旋转类型**（仅当motion_type=rotating时）：
   - "centerline": 绕中心线旋转
   - "edge": 绕边旋转
   - "custom_axis": 绕自定义轴旋转
5. **运动范围**（可选）：
   - 滑动：距离（米），如0.1表示±0.1米
   - 旋转：角度（度），如90表示±90度

**重要**：
- 必须从上述部件列表中选择一个确切的部件名称（part_name）
- 用户提示是绝对正确的，请严格按照用户提示解析
- 如果用户说"按下去"、"按下"、"推"等，通常是sliding运动，方向为z（上下）
- 如果用户说"旋转"、"转动"、"打开"（门）等，通常是rotating运动

请仔细比较所有图片，选择最符合描述的部件。"""
        
        full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
        
        # 如果提供了整体图片，一起分析
        if overall_image_path and Path(overall_image_path).exists():
            all_images = [overall_image_path] + part_image_list
            response_text = describe_multiple_images(llm, all_images, instruction=full_prompt)
        else:
            # 只使用部件图片
            if len(part_image_list) > 1:
                response_text = describe_multiple_images(llm, part_image_list, instruction=full_prompt)
            else:
                # 单个图片，使用普通调用
                messages = [
                    SystemMessage(content="你是一个专业的视觉分析助手，请通过比较图片准确识别部件。"),
                    HumanMessage(content=full_prompt)
                ]
                response = llm.invoke(messages)
                response_text = getattr(response, "content", str(response))
        
        # 解析LLM输出
        parsed_result = parser.parse(response_text)
        
        # 验证part_name是否在列表中
        if parsed_result.part_name not in part_names_list:
            print(f"  ⚠ LLM返回的部件名称 '{parsed_result.part_name}' 不在可用列表中，尝试模糊匹配...")
            # 模糊匹配
            for part_name in part_names_list:
                if parsed_result.part_name.lower() in part_name.lower() or part_name.lower() in parsed_result.part_name.lower():
                    parsed_result.part_name = part_name
                    print(f"  ✓ 模糊匹配到: {part_name}")
                    break
            else:
                print(f"  ✗ 无法匹配部件名称，使用第一个部件: {part_names_list[0]}")
                parsed_result.part_name = part_names_list[0]
        
        # 转换为字典格式
        hint_dict = {
            "motion_type": parsed_result.motion_type
        }
        
        if parsed_result.sliding_direction:
            hint_dict["sliding_direction"] = parsed_result.sliding_direction
        
        if parsed_result.rotation_type:
            hint_dict["rotation_type"] = parsed_result.rotation_type
        
        if parsed_result.motion_range is not None:
            hint_dict["motion_range"] = parsed_result.motion_range
        
        return {
            parsed_result.part_name: hint_dict
        }
    except Exception as e:
        print(f"  ✗ LLM解析失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_user_hints_interactive(
    part_images: Dict[str, str], 
    overall_image_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """交互式获取用户提示（自然语言输入，通过视觉比较自动解析）
    
    Args:
        part_images: 部件名称到图片路径的字典，格式：{part_name: image_path}
        overall_image_path: 整体场景图片路径（可选）
        
    Returns:
        解析后的提示字典，格式：{part_name: {motion_type, sliding_direction, rotation_type, motion_range}}
    """
    print("\n" + "=" * 80)
    print("💡 用户提示（可选）")
    print("=" * 80)
    print("您可以提供自然语言提示，系统将通过视觉比较所有部件图片自动解析。")
    print("例如：'马桶按钮是按下去的'、'盖子可以旋转打开'、'门是左右滑动的'")
    print("留空直接回车表示不使用提示\n")
    
    if part_images:
        print(f"可用的部件（已渲染高亮图片）: {', '.join(part_images.keys())}\n")
    
    hints = {}
    while True:
        hint_input = input("请输入自然语言提示（如：'按钮是按下去的' 或留空结束）: ").strip()
        if not hint_input:
            break
        
        print(f"  🔍 正在通过视觉比较解析: {hint_input}...")
        parsed_hints = parse_user_hint_with_visual_comparison(hint_input, part_images, overall_image_path)
        
        if parsed_hints:
            hints.update(parsed_hints)
            for part_name, hint in parsed_hints.items():
                print(f"  ✓ 解析结果: {part_name} -> {hint}")
        else:
            print(f"  ⚠ 解析失败，请重试")
    
    if hints:
        print(f"\n✓ 共解析 {len(hints)} 个用户提示")
    else:
        print("\n✓ 未使用用户提示")
    
    return hints


__all__ = [
    "get_parts_from_xml",
    "parse_user_hint_with_visual_comparison",
    "get_user_hints_interactive",
]


