# 多模态家具智能体系统

这是一个基于LangChain和LangGraph框架的多模态家具智能体系统，能够自动分析家具场景，识别兴趣零件并生成约束建议。

## 🎯 核心功能

- **场景理解与记忆**：分析3x3渲染图像+XML，理解家具类型并存储到记忆系统
- **专家权重分配**：根据家具复杂度动态分配家具专家vs机构专家权重
- **逐mesh评估**：通过高亮单个mesh+半透明其他mesh的方式，让专家评估每个mesh是否为兴趣零件
- **多轮测试聚合**：通过多次评估提高识别稳定性
- **智能约束生成**：基于专家评估结果生成运动约束建议

## 🏗️ 系统架构

```
多模态家具智能体系统
├── 数据模型层 (data_models.py)
│   ├── 场景记忆 (SceneMemory)
│   ├── 专家评估 (ExpertEvaluation)
│   ├── 网格信息 (MeshInfo)
│   └── 分析结果 (SceneAnalysisResult)
├── 记忆系统层 (memory_system.py)
│   ├── 场景存储与检索
│   ├── 相似场景查找
│   └── 专家权重建议
├── 渲染工具层 (rendering_utils.py)
│   ├── 网格高亮渲染
│   ├── 多视角马赛克生成
│   └── 批量渲染处理
├── 专家智能体层 (expert_agents.py)
│   ├── 家具专家 (FurnitureExpert)
│   ├── 机构专家 (MechanismExpert)
│   └── 专家协调器 (ExpertCoordinator)
├── 主智能体层 (multimodal_agent.py)
│   ├── 多模态家具智能体
│   └── 智能体管理器
└── 工作流层 (langgraph_workflow.py)
    ├── LangGraph工作流
    └── 工作流管理器
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langgraph pydantic numpy pillow trimesh mujoco
```

### 2. 环境配置

创建 `.env` 文件：

```env
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=your_qwen_base_url
```

### 3. 运行测试

```bash
cd agent
python test_multimodal_agent.py
```

### 4. 分析单个场景

```bash
python run_multimodal_agent.py --xml_path Examples/wardrobe/obj.xml
```

### 5. 批量分析场景

```bash
python run_multimodal_agent.py --batch Examples/wardrobe/ Examples/crank_slider/
```

## 📊 使用示例

### 基本用法

```python
from agent.multimodal_agent import MultimodalFurnitureAgent

# 创建智能体
agent = MultimodalFurnitureAgent()

# 分析场景
result = agent.analyze_scene(
    xml_path="Examples/wardrobe/obj.xml",
    views=["front", "right", "back", "left", "iso"],
    per_view_size=(480, 360),
    save_results=True
)

# 查看结果
print(f"兴趣零件: {result.interest_parts}")
print(f"分析置信度: {result.analysis_confidence}")
```

### 使用工作流

```python
from agent.langgraph_workflow import WorkflowManager

# 创建工作流管理器
workflow_manager = WorkflowManager()

# 运行分析
result = workflow_manager.analyze_scene(
    xml_path="Examples/wardrobe/obj.xml",
    views=["front", "right", "back", "left"],
    per_view_size=(480, 360)
)

# 生成报告
report = workflow_manager.create_analysis_report(result)
print(report)
```

## 🔧 配置选项

### 渲染配置

```python
from agent.data_models import RenderingConfig

config = RenderingConfig(
    target_mesh_brightness=1.5,      # 目标网格亮度倍数
    target_mesh_saturation=1.3,      # 目标网格饱和度倍数
    other_mesh_alpha=0.3,            # 其他网格透明度
    other_mesh_desaturation=0.5,     # 其他网格去饱和度
    show_aabb_outline=True           # 显示AABB轮廓
)
```

### 专家权重

```python
from agent.data_models import ExpertWeights

# 简单家具：家具专家权重高
weights_simple = ExpertWeights(furniture_expert=0.8, mechanism_expert=0.2)

# 复杂机构：机构专家权重高
weights_complex = ExpertWeights(furniture_expert=0.3, mechanism_expert=0.7)
```

## 📈 工作流程

1. **场景理解**：分析3x3渲染图像，识别家具类型和复杂度
2. **记忆存储**：将场景信息存储到记忆系统
3. **网格分析**：解析XML文件，提取所有mesh信息
4. **逐mesh评估**：
   - 渲染高亮图像（目标mesh高亮，其他半透明）
   - 家具专家评估（功能重要性、运动潜力、结构完整性）
   - 机构专家评估（运动学特征、约束需求、机构复杂度）
   - 综合评分和兴趣等级判定
5. **结果聚合**：生成最终的兴趣零件列表和分析报告

## 🎨 可视化效果

- **高亮渲染**：目标mesh亮度1.5x，饱和度1.3x
- **半透明处理**：其他mesh透明度0.3，去饱和度0.5
- **AABB轮廓**：显示目标mesh的包围盒轮廓
- **多视角马赛克**：3x3网格布局，支持9个视角

## 📊 输出格式

### 分析结果

```json
{
  "scene_id": "wardrobe_001",
  "furniture_type": "wardrobe",
  "complexity_level": "simple",
  "scene_description": "四门衣柜，包含主体框架、四扇门板、隔板等",
  "expert_weights": {
    "furniture_expert": 0.8,
    "mechanism_expert": 0.2
  },
  "interest_parts": ["Plane001", "Plane002", "Plane003"],
  "analysis_confidence": 0.85,
  "mesh_evaluations": [...]
}
```

### 专家评估

```json
{
  "mesh_name": "Plane001",
  "furniture_evaluation": {
    "interest_level": "high",
    "confidence": 0.9,
    "reasoning": "门板部件，具有明显的旋转运动特征",
    "functional_importance": 0.9,
    "motion_potential": 0.8,
    "structural_integrity": 0.7
  },
  "mechanism_evaluation": {
    "interest_level": "medium",
    "confidence": 0.7,
    "reasoning": "需要添加铰链约束",
    "kinematic_features": 0.8,
    "constraint_requirements": 0.9,
    "mechanism_complexity": 0.3
  },
  "final_score": 0.85,
  "final_interest_level": "high",
  "is_interest_part": true
}
```

## 🔍 故障排除

### 常见问题

1. **MuJoCo渲染失败**
   - 检查XML文件路径是否正确
   - 确认mesh文件存在
   - 检查MuJoCo安装

2. **LLM调用失败**
   - 检查API密钥和Base URL
   - 确认网络连接
   - 检查模型可用性

3. **记忆系统错误**
   - 检查数据库文件权限
   - 确认SQLite安装
   - 清理损坏的数据库文件

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行测试
python test_multimodal_agent.py
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- LangChain 团队提供的优秀框架
- MuJoCo 物理仿真引擎
- 通义千问多模态大模型
