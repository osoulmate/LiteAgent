# 🚀 LiteAgent

**LiteAgent** 是一个极简的、具备“自我进化”能力的智能体框架。它不仅能根据用户需求调用现有工具，还能在能力缺失时，通过 LLM 实时编写、验证并持久化存储新的 Python 工具函数。

---

## 🌟 核心特性 (Core Features)

* **极简架构 (Minimalist)**：仅需少量核心代码即可实现复杂的工具调用逻辑。
* **工具自生成 (Self-Evolution)**：当现有工具无法满足需求时，LiteAgent 会自动编写 Python 代码并生成新工具。
* **持久化存储 (Persistence)**：生成的工具会自动以 JSON 形式保存，下次遇到类似需求时无需重复生成，直接复用。
* **智能决策 (Smart Dispatcher)**：内置语义分析器，自动判断是该调用工具、生成新工具，还是直接回答。
* **国产模型优化 (Qwen Powered)**：深度适配通义千问（Qwen）系列模型，支持 `qwen-plus` 等高性能后端。
* **安全沙箱意识 (Security Minded)**：内置基础的代码扫描逻辑，防止生成并执行危险的系统级指令。

## 🛠️ 工作原理 (How It Works)

LiteAgent 的运行逻辑遵循以下循环：

1. **感知 (Perceive)**：分析用户输入，识别潜在的任务需求。
2. **检索 (Retrieve)**：在 `tools.json` 中匹配已有的工具哈希。
3. **创造 (Create)**：若无匹配项，调用 Qwen 生成符合规范的 Python 函数。
4. **验证 (Validate)**：通过静态检查与模拟执行，确保代码安全可用。
5. **进化 (Evolve)**：将新工具序列化保存，完成智能体能力的永久升级。

## 🚀 快速开始 (Quick Start)

### 1. 设置环境变量

```bash
export DASHSCOPE_API_KEY='你的通义千问API密匙'

```

### 2. 核心演示

```python
# 示例：让 LiteAgent 处理一个它原本不会的任务
question = "将 'Hello LiteAgent' 转换为 Base64 编码"
tool_type, tool_name, result = smart_agent.process_question(question)

# LiteAgent 会发现没有 Base64 工具，于是：
# 1. 自动编写 base64_converter 函数
# 2. 保存到 tools.json
# 3. 执行并返回结果

```

## ⚠️ 风险提示 (Disclaimer)

本项是一个**实验性**的极简智能体示例。由于其具备 `exec()` 动态执行代码的能力，请务必在受信任的环境中运行，并在生产环境部署前加强 `validate_tool` 中的安全过滤规则。

---

**LiteAgent**: *Small in size, Infinite in potential.*

---
