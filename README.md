# LinkForge

[English](README_EN.md) | 中文

LinkForge 是一个基于多模态大模型的 MJCF 约束生成系统，能够自动分析 3D 家具模型，识别运动部件，并生成 MuJoCo 物理仿真所需的运动约束。

## 🎬 演示效果

<table>
<tr>
<td width="50%">
  <img src="assets/demo.png" alt="Demo" style="width:100%; height:auto; max-height:400px; object-fit:contain;">
</td>
<td width="50%">
  <img src="assets/demo.gif" alt="Demo Animation" style="width:100%; height:auto; max-height:400px; object-fit:contain;">
</td>
</tr>
</table>

## 🎯 核心功能

- **场景渲染与编排**：自动渲染 3D 模型的多视角图像
- **场景感知分析**：识别家具类型和运动部件
- **约束推理**：基于视觉和几何信息推理运动约束
- **MJCF 约束生成**：自动生成 MuJoCo XML 格式的运动约束

## 🏗️ 系统架构

```
LinkForge
├── agent/                    # 核心智能体模块
│   ├── render_orchestrator.py      # 渲染编排智能体
│   ├── scene_awareness_agent.py   # 场景感知智能体
│   ├── constraint_reasoning_agent.py  # 约束推理智能体
│   ├── mjcf_constraint_agent.py   # MJCF约束生成智能体
│   ├── memory.py                  # 记忆系统
│   ├── tools/                     # 工具模块
│   └── utils/                     # 工具函数
├── DEMO/                    # 演示程序
│   └── main.py              # 主程序入口
├── docs/                    # 文档
└── Examples/                # 示例模型
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r agent/utils/requirements.txt
```

### 2. 环境配置

复制 `.env.example` 文件为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入你的实际 API 密钥和 Base URL。

支持两种配置方式：
- **通义千问 API**：设置 `QWEN_API_KEY` 和 `QWEN_BASE_URL`
- **OpenAI 兼容 API**：设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`

详细说明请参考 `.env.example` 文件中的注释。

### 3. 运行程序

```bash
python DEMO/main.py
```

程序将：
1. 加载指定的 XML 模型文件
2. 渲染多视角图像
3. 分析场景和运动部件
4. 推理运动约束
5. 生成 MJCF 格式的约束文件

## 📖 文档

详细的系统架构文档：

- [系统架构](docs/系统架构.md) - 完整的系统架构说明，包括各智能体的职责、数据流、记忆系统、数据模型等

## 🔧 配置

在 `DEMO/main.py` 中可以修改以下配置：

- `xml_path`: 要处理的 XML 模型文件路径
- `memory_path`: 记忆存储路径（默认为 `scene_memory`）
- `max_workers`: 并行处理线程数

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- LangChain 团队提供的优秀框架
- MuJoCo 物理仿真引擎
- 通义千问、OpenAI提供的多模态大模型
