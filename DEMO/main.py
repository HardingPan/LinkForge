"""
MJCF约束生成智能体Demo
演示如何将ConstraintReasoningAgent得到的运动约束结果转换为MJCF格式
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.mjcf_constraint_agent import MJCFConstraintAgent
from agent.constraint_reasoning_agent import ConstraintReasoningAgent
from agent.render_orchestrator import RenderOrchestrator
from agent.scene_awareness_agent import SceneAwarenessAgent
from agent.utils.data_models import MotionConstraintResult
from agent.utils.user_hint_parser import get_user_hints_interactive
import json


def main():
    """主函数：演示MJCF约束生成流程"""
    
    # 配置
    xml_path = project_root / "Examples" / "sv" / "马桶.xml"
    memory_path = project_root / "scene_memory"
    
    print("=" * 80)
    print("MJCF约束生成智能体 Demo")
    print("=" * 80)
    print(f"XML文件: {xml_path}")
    print(f"记忆路径: {memory_path}")
    print()
    
    # 1. 初始化智能体
    print("📦 初始化智能体...")
    memory_storage_path = str(memory_path)
    render_orchestrator = RenderOrchestrator(
        memory_storage_path=memory_storage_path
    )
    scene_agent = SceneAwarenessAgent(
        memory_storage_path=memory_storage_path
    )
    constraint_reasoning_agent = ConstraintReasoningAgent(
        memory_storage_path=memory_storage_path
    )
    mjcf_agent = MJCFConstraintAgent()
    print("✓ 智能体初始化完成\n")
    
    # 2. 运行渲染编排（会自动清空scene_memory）
    print("🎨 运行渲染编排...")
    render_result = render_orchestrator.orchestrate_rendering(
        xml_path=str(xml_path),
        max_workers=4,
        clear_memory=True  # 在开始前清空scene_memory
    )
    if render_result.get("success"):
        print(f"✓ 渲染完成\n")
    else:
        print(f"✗ 渲染失败: {render_result.get('message', '未知错误')}")
        return
    
    # 2.5. 获取用户提示（在渲染完成后，使用视觉比较）
    part_images = render_result.get("part_images", {})
    overall_image_path = render_result.get("overall_image_path")
    user_hints = get_user_hints_interactive(part_images, overall_image_path)
    
    # 更新智能体的用户提示
    scene_agent.user_hints = user_hints
    constraint_reasoning_agent.user_hints = user_hints
    print()
    
    # 3. 运行场景分析（获取part分析结果）
    print("🔍 运行场景分析...")
    scene_result = scene_agent.analyze_scene(
        xml_path=str(xml_path)
    )
    if not scene_result.get("success"):
        print(f"✗ 场景分析失败: {scene_result.get('message', '未知错误')}")
        return
    print(f"✓ 场景分析完成\n")
    
    # 4. 批量分析所有part（快速模式，只分析运动类型）
    print("📊 分析所有part的运动类型...")
    parts_result = scene_agent.analyze_all_parts_with_memory(
        max_workers=4
    )
    if not parts_result.get("success"):
        print(f"✗ Part分析失败: {parts_result.get('message', '未知错误')}")
        return
    
    motion_parts = parts_result["result"].motion_parts
    print(f"✓ Part分析完成，发现 {len(motion_parts)} 个运动部件: {motion_parts}\n")
    
    # 5. 运行约束推理（获取运动约束结果）
    print("🔧 运行约束推理...")
    # 处理所有运动部件
    if motion_parts:
        part_names = motion_parts  # 处理所有运动部件
        print(f"  将处理所有运动部件 ({len(part_names)}个): {part_names}")
    else:
        part_names = ["d0"]  # 如果没有分析结果，默认测试第一个门
        print(f"  未找到运动部件，将测试: {part_names}")
    
    # 并行处理约束推理
    constraint_results = []
    failed_parts = []
    
    print(f"  使用4个线程并行处理 {len(part_names)} 个部件")
    
    def process_single_part(part_name: str) -> tuple:
        """处理单个部件的约束推理（用于并行处理）"""
        try:
            # 获取part的分析结果
            part_analysis = scene_agent.get_part_analysis_by_name(part_name)
            if not part_analysis:
                return (part_name, None, "未找到分析结果")
            
            result_dict = constraint_reasoning_agent.reason_motion_constraint(
                part_name=part_name,
                part_analysis=part_analysis
            )
            
            if result_dict and result_dict.get("success"):
                result = result_dict.get("result")
                return (part_name, result, None)
            else:
                error_msg = result_dict.get("message", "未知错误") if result_dict else "返回None"
                return (part_name, None, error_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (part_name, None, f"异常: {str(e)}")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_part = {
            executor.submit(process_single_part, part_name): part_name
            for part_name in part_names
        }
        
        # 收集结果（按完成顺序）
        completed_count = 0
        for future in as_completed(future_to_part):
            part_name = future_to_part[future]
            completed_count += 1
            try:
                part_name_result, result, error = future.result()
                
                if error:
                    print(f"  [{completed_count}/{len(part_names)}] ✗ {part_name_result}: {error}")
                    failed_parts.append((part_name_result, error))
                elif result:
                    constraint_results.append(result)
                    print(f"  [{completed_count}/{len(part_names)}] ✓ {part_name_result}: {result.motion_type}")
            except Exception as e:
                print(f"  [{completed_count}/{len(part_names)}] ✗ {part_name}: 处理异常 - {str(e)}")
                failed_parts.append((part_name, f"异常: {str(e)}"))
    
    # 显示处理摘要
    print(f"\n📊 约束推理摘要:")
    print(f"  ✓ 成功: {len(constraint_results)} 个")
    print(f"  ✗ 失败: {len(failed_parts)} 个")
    if failed_parts:
        print(f"  失败的部件:")
        for part_name, error in failed_parts:
            print(f"    - {part_name}: {error}")
    
    if not constraint_results:
        print("\n❌ 没有获得任何约束结果，退出")
        return
    
    print(f"\n✓ 约束推理完成: {len(constraint_results)} 个结果")
    
    # 4. 生成MJCF约束
    print("\n🔧 生成MJCF约束...")
    
    # 输出文件保存到同一目录
    output_xml_path = xml_path.parent / "result.xml"
    
    generation_result = mjcf_agent.generate_constraints(
        xml_path=str(xml_path),
        constraint_results=constraint_results,
        output_path=str(output_xml_path),
        create_backup=True
    )
    
    if generation_result.success:
        print(f"\n✓ MJCF约束生成成功: {generation_result.xml_path}")
        
        # 5. 保存结果到JSON
        result_json_path = xml_path.parent / "mjcf_constraint_generation_result.json"
        result_data = {
            "success": generation_result.success,
            "message": generation_result.message,
            "xml_path": str(generation_result.xml_path) if generation_result.xml_path else None,
            "constraint_plans": [
                {
                    "part_name": plan.part_name,
                    "motion_type": plan.motion_type,
                    "rotation_type": plan.rotation_type,
                    "joint": {
                        "name": plan.joint.name,
                        "type": plan.joint.type.value,
                        "axis": plan.joint.axis,
                        "pos": plan.joint.pos,
                        "range": plan.joint.range,
                        "damping": plan.joint.damping,
                    } if plan.joint else None,
                    "sites_count": len(plan.sites),
                    "equality_constraints_count": len(plan.equality_constraints),
                    "confidence": plan.confidence,
                    "reasoning": plan.reasoning
                }
                for plan in generation_result.constraint_plans
            ],
            "modifications": generation_result.modifications
        }
        
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # 结果已保存（不打印）
        
    else:
        print(f"\n❌ MJCF约束生成失败: {generation_result.message}")
    
    print("\n" + "=" * 80)
    print("Demo 完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

