"""
AI生成并自修复 Python 摩斯密码生成工具示例：
1.可以实现根据用户需求描述自动生成一个Python函数，并通过自修复循环确保代码正确性。
2.同时用生成的函数实现根据输入的文字生成摩斯密码的功能。
"""
import os
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import dashscope
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# 设置千问API Key (从阿里云控制台获取)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")   # 确保变量名一致
#print(dashscope.api_key)
assert dashscope.api_key, "请先在环境变量里设置 DASHSCOPE_API_KEY"

class DynamicToolCreator:
    def __init__(self, llm):
        self.llm = llm
        self.available_tools = {}

    def create_tool_prompt(self):
        return PromptTemplate(
            template="""你是一个AI工具创建专家。根据用户需求，创建一个Python工具函数。

用户需求: {user_request}

请生成一个完整的Python函数，遵循以下要求：
1. 函数名应该描述其功能
2. 包含详细的docstring说明
3. 返回字符串结果
4. 不要使用危险操作（如文件删除、网络请求等）
5. 生成的代码不含诸如```这样的markdown风格的代码格式化字符

只返回纯净的Python代码，不要有其他解释：

""",
            input_variables=["user_request"]
        )

    def create_tool(self, user_request):
        """根据用户需求动态创建工具"""
        prompt = self.create_tool_prompt()
        chain = prompt | self.llm

        # 获取模型生成的代码
        code_response = chain.invoke({"user_request": user_request})
        full_code = f"{code_response}"

        print(f"生成的代码:\n{full_code}")

        try:
            # 创建新的命名空间来执行代码
            namespace = {}
            exec(full_code, namespace)

            # 获取刚定义的函数
            new_function = None
            for key, value in namespace.items():
                if callable(value) and key != '__builtins__':
                    new_function = value
                    break

            if new_function:
                tool_name = f"dynamic_tool_{len(self.available_tools) + 1}"
                self.available_tools[tool_name] = new_function
                return f"成功创建工具: {tool_name}", new_function
            else:
                return "无法从生成的代码中提取函数", None

        except Exception as e:
            return f"代码执行错误: {e}", None

    def use_dynamic_tool(self, tool_name, *args, **kwargs):
        """使用动态创建的工具"""
        if tool_name in self.available_tools:
            try:
                result = self.available_tools[tool_name](*args, **kwargs)
                return result
            except Exception as e:
                return f"工具执行错误: {e}"
        else:
            return f"工具 {tool_name} 不存在"
class MyLLM(LLM):
    """通义千问模型的LangChain包装器"""

    model_name: str = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from dashscope import Generation

        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            **kwargs
        )

        if response.status_code == 200:
            return response.output.text
        else:
            return f"Error: {response.code} - {response.message}"

# 定义工具函数
def search_web(query: str) -> str:
    """模拟网络搜索工具"""
    # 在实际应用中，这里可以接入真实的搜索API
    return f"根据搜索 '{query}'，我找到了以下信息：这是一个模拟的搜索结果，在实际应用中会返回真实数据。"

def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return f"无法计算表达式: {expression}"

# 创建工具列表
tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="当需要搜索最新的、实时的信息时使用此工具。输入应为搜索关键词。"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="当需要进行数学计算时使用此工具。输入应为数学表达式，例如 '2 + 3 * 4'。"
    )
]

# 初始化千问模型
llm = MyLLM(model_name="qwen-plus-2025-04-28", temperature=0.1)
tool_creator = DynamicToolCreator(llm)

# 让模型创建工具
user_need = "需要一个工具来将文本转换为摩斯密码"
creation_result, new_tool = tool_creator.create_tool(user_need)
print(creation_result)

if new_tool:
    # 使用新创建的工具
    result = tool_creator.use_dynamic_tool("dynamic_tool_1", "Hello World")
    print(f"工具执行结果: {result}")
'''
# 创建代理
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # 处理解析错误
)

# 测试不同的问题
test_questions = [
    "计算一下 15 * 24 + 38 等于多少？",
    "搜索一下今天的热点新闻",
    "请写一首关于春天的诗",  # 这个不需要工具
    "先计算 128 ÷ 8，然后告诉我结果"
]

for question in test_questions:
    print(f"\n{'='*50}")
    print(f"问题: {question}")
    print(f"{'='*50}")

    try:
        result = agent.run(question)
        print(f"答案: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
'''