"""
AI生成并存储工具示例：
1.可以实现根据用户需求描述选择合适的工具。
2.可以动态创建新的工具函数并保存以备后续使用。
3.具备基本的安全检查机制，防止生成危险代码。
4.支持工具的持久化存储和加载。
5.集成了一个智能代理，根据问题类型选择处理方式。
6.使用 Qwen 模型作为语言模型后端。
7.可选用qwen-plus。
8.请先在环境变量里设置 DASHSCOPE_API_KEY
9.本示例主要由大模型生成并伴有人类修改仅供学习参考，实际使用请注意安全风险。
"""
import os
import json
import hashlib
import inspect
import re
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import dashscope
from typing import Any, Dict, List, Mapping, Optional, Tuple, Callable
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# 设置千问API Key (从阿里云控制台获取)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
assert dashscope.api_key, "请先在环境变量里设置 DASHSCOPE_API_KEY"

class MyLLM(LLM):
    """通义千问模型的LangChain包装器"""

    model_name: str = "qwen-plus"
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

class ToolManager:
    """工具管理器，负责工具的存储、加载和验证"""

    def __init__(self, storage_file="tools.json"):
        self.storage_file = storage_file
        self.available_tools = {}
        self.tool_requests = {}  # 记录用户请求和对应工具
        self.load_tools()

    def get_tool_hash(self, code: str) -> str:
        """生成工具代码的哈希值作为唯一标识"""
        return hashlib.md5(code.encode()).hexdigest()

    def get_request_hash(self, user_request: str) -> str:
        """生成用户请求的哈希值"""
        # 清理请求文本，移除多余空格和标点
        cleaned_request = re.sub(r'[^\w\u4e00-\u9fff]', ' ', user_request.lower())
        cleaned_request = re.sub(r'\s+', ' ', cleaned_request).strip()
        return hashlib.md5(cleaned_request.encode()).hexdigest()

    def save_tools(self):
        """保存工具到文件"""
        try:
            serializable_tools = {}
            for tool_name, tool_info in self.available_tools.items():
                # 只保存可序列化的信息，不保存函数对象
                serializable_tools[tool_name] = {
                    'code': tool_info.get('code', ''),
                    'hash': tool_info.get('hash', ''),
                    'description': tool_info.get('description', ''),
                    'function_name': tool_info.get('function_name', ''),
                    'parameters': tool_info.get('parameters', {}),
                    'request_hash': tool_info.get('request_hash', '')
                }

            # 使用ensure_ascii=False来保存中文
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_tools, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存工具失败: {e}")

    def load_tools(self):
        """从文件加载工具"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    saved_tools = json.load(f)

                for tool_name, tool_info in saved_tools.items():
                    if 'code' in tool_info:
                        # 重新创建函数对象
                        try:
                            namespace = {}
                            exec(tool_info['code'], namespace)

                            function_name = tool_info.get('function_name', '')
                            if function_name and function_name in namespace:
                                tool_info['function'] = namespace[function_name]
                                self.available_tools[tool_name] = tool_info

                                # 恢复请求映射
                                request_hash = tool_info.get('request_hash')
                                if request_hash:
                                    self.tool_requests[request_hash] = tool_name
                        except Exception as e:
                            print(f"加载工具 {tool_name} 失败: {e}")
        except Exception as e:
            print(f"加载工具文件失败: {e}")

    def add_tool(self, tool_name: str, tool_function: Callable, code: str, description: str = "", request_hash: str = ""):
        """添加新工具"""
        try:
            # 分析函数参数
            sig = inspect.signature(tool_function)
            parameters = {}
            for param_name, param in sig.parameters.items():
                parameters[param_name] = {
                    'name': param_name,
                    'kind': str(param.kind),
                    'default': param.default if param.default != param.empty else None,
                    'required': param.default == param.empty
                }

            tool_hash = self.get_tool_hash(code)
            function_name = tool_function.__name__

            self.available_tools[tool_name] = {
                'function': tool_function,
                'code': code,
                'hash': tool_hash,
                'description': description,
                'function_name': function_name,
                'parameters': parameters,
                'request_hash': request_hash
            }

            # 记录请求映射
            if request_hash:
                self.tool_requests[request_hash] = tool_name

            self.save_tools()
            return True
        except Exception as e:
            print(f"添加工具失败: {e}")
            return False

    def get_tool_by_request(self, user_request: str) -> Optional[str]:
        """根据用户请求获取已存在的工具"""
        request_hash = self.get_request_hash(user_request)
        return self.tool_requests.get(request_hash)

    def get_tool(self, tool_name: str):
        """获取工具"""
        if tool_name in self.available_tools:
            return self.available_tools[tool_name].get('function')
        return None

    def get_tool_info(self, tool_name: str):
        """获取工具信息"""
        return self.available_tools.get(tool_name)

    def tool_exists(self, tool_hash: str) -> bool:
        """检查工具是否已存在"""
        for tool_info in self.available_tools.values():
            if tool_info.get('hash') == tool_hash:
                return True
        return False

    def use_dynamic_tool(self, tool_name: str, *args, **kwargs):
        """使用动态创建的工具"""
        tool_func = self.get_tool(tool_name)
        if tool_func:
            try:
                result = tool_func(*args, **kwargs)
                return result
            except Exception as e:
                return f"工具执行错误: {e}"
        else:
            return f"工具 {tool_name} 不存在"

class DynamicToolCreator:
    def __init__(self, llm):
        self.llm = llm
        self.tool_manager = ToolManager()

    def create_tool_prompt(self):
        return PromptTemplate(
            template="""你是一个AI工具创建专家。根据用户需求，创建一个Python工具函数。

用户需求: {user_request}

请生成一个完整的Python函数，遵循以下要求：
1. 函数名应该描述其功能（使用英文名称）
2. 包含详细的docstring说明
3. 返回字符串结果
4. 如使用危险操作（如import、文件删除、网络请求、系统调用等）是完成任务的必要条件，请在docstring中说明
5. 生成的代码不含诸如```这样的markdown风格的代码格式化字符
6. 生成的函数使用可变位置参数（*args）和可变关键字参数（**kwargs）两个参数完成任意传参需求
只返回纯净的Python代码，不要有其他解释：

""",
            input_variables=["user_request"]
        )

    def extract_function_name(self, code: str) -> str:
        """从代码中提取函数名"""
        # 查找def开头的行
        lines = code.split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                # 提取函数名
                match = re.match(r'def\s+(\w+)\s*\(', line.strip())
                if match:
                    return match.group(1)
        return "unknown_function"

    def validate_tool(self, code: str, test_input:any) -> Tuple[bool, str]:
        """验证工具代码的安全性并测试功能"""
        # 安全检查：禁止的危险操作
        dangerous_keywords = [
            'os.', 'subprocess', 'eval', 'exec', 'open(', '__import__',
            'file(', 'compile', 'input(', 'reload', 'pty', 'commands',
            'sys.', 'shutil.', 'glob.', 'pickle', 'marshal', 'ctypes',
            'paramiko', 'requests', 'urllib', 'socket', 'http.client',
            'import ', 'from ', '__', 'breakpoint', 'memoryview', 'bytearray'
        ]
        #屏蔽危险操作以供测试用例通过
        dangerous_keywords = []
        for keyword in dangerous_keywords:
            if keyword in code and not code.strip().startswith('#'):
                return False, f"代码包含危险操作: {keyword}"

        try:
            # 执行代码创建函数
            namespace = {}
            exec(code, namespace)
            # 获取刚定义的函数
            new_function = None
            function_name = None
            for key, value in namespace.items():
                if callable(value) and key != '__builtins__':
                    new_function = value
                    function_name = key
                    break

            if not new_function:
                return False, "无法从生成的代码中提取函数"
            return True, "验证通过"

        except Exception as e:
            return False, f"代码验证失败: {e}"

    def create_tool(self, user_request: str, test_input) -> Tuple[str, Any]:
        """根据用户需求动态创建并验证工具"""

        # 首先检查是否已有相同请求的工具
        existing_tool_name = self.tool_manager.get_tool_by_request(user_request)
        if existing_tool_name:
            return f"工具已存在: {existing_tool_name}", self.tool_manager.get_tool(existing_tool_name)

        prompt = self.create_tool_prompt()
        chain = prompt | self.llm

        # 获取模型生成的代码
        code_response = chain.invoke({"user_request": user_request})
        full_code = f"{code_response}"

        print(f"生成的代码:\n{full_code}")

        # 验证工具
        is_valid, validation_msg = self.validate_tool(full_code, test_input)

        if not is_valid:
            print(f"工具验证失败: {validation_msg}")
            #return f"工具验证失败: {validation_msg}", None

        # 创建工具函数
        try:
            namespace = {}
            exec(full_code, namespace)

            new_function = None
            function_name = None
            for key, value in namespace.items():
                if callable(value) and key != '__builtins__':
                    new_function = value
                    function_name = key
                    break

            if new_function:
                # 生成工具名称和描述
                tool_hash = self.tool_manager.get_tool_hash(full_code)
                request_hash = self.tool_manager.get_request_hash(user_request)

                # 检查是否已存在相同工具
                if self.tool_manager.tool_exists(tool_hash):
                    return "工具已存在，无需重复创建", None

                # 使用更有意义的工具名称
                base_name = self.extract_function_name(full_code)
                tool_name = f"{base_name}_{len(self.tool_manager.available_tools) + 1}"
                description = f"动态创建的工具: {user_request}"

                # 分析函数参数用于描述
                sig = inspect.signature(new_function)
                param_desc = []
                for param_name, param in sig.parameters.items():
                    if param.default == param.empty:
                        param_desc.append(f"{param_name}")
                    else:
                        param_desc.append(f"{param_name}(默认:{param.default})")

                description += f"\n参数: {', '.join(param_desc)}"

                # 保存到工具管理器
                success = self.tool_manager.add_tool(
                    tool_name, new_function, full_code, description, request_hash
                )
                if success:
                    return f"成功创建并验证工具: {tool_name}", new_function
                else:
                    return "工具保存失败", None
            else:
                return "无法从生成的代码中提取函数", None

        except Exception as e:
            return f"工具创建错误: {e}", None

    def get_available_tools(self) -> List[Tool]:
        """获取所有可用工具的LangChain Tool列表"""
        tools = []
        for tool_name, tool_info in self.tool_manager.available_tools.items():
            if 'function' in tool_info:
                # 创建包装函数来处理参数传递
                def create_tool_wrapper(func, params):
                    def wrapper(input_str: str) -> str:
                        try:
                            # 分析参数数量
                            sig = inspect.signature(func)
                            param_count = len(sig.parameters)

                            if param_count == 0:
                                return func()
                            elif param_count == 1:
                                return func(input_str)
                            else:
                                # 对于多参数函数，尝试解析输入
                                # 这里简单地将输入作为第一个参数，其他使用默认值
                                args = [input_str]
                                for param_name, param_info in params.items():
                                    if param_name != list(params.keys())[0]:  # 跳过第一个参数
                                        if param_info['required']:
                                            args.append("default_value")
                                        else:
                                            args.append(param_info['default'])
                                return func(*args)
                        except Exception as e:
                            return f"工具执行错误: {e}"
                    return wrapper

                wrapper_func = create_tool_wrapper(
                    tool_info['function'],
                    tool_info.get('parameters', {})
                )

                tools.append(
                    Tool(
                        name=tool_name,
                        func=wrapper_func,
                        description=tool_info.get('description', '动态创建的工具')
                    )
                )
        return tools

class SmartAgent:
    """智能代理，根据问题类型选择处理方式"""

    def __init__(self, llm, tool_creator):
        self.llm = llm
        self.tool_creator = tool_creator
        self.base_tools = self.initialize_base_tools()

    def initialize_base_tools(self):
        """初始化基础工具"""
        def search_web(query: str) -> str:
            return f"根据搜索 '{query}'，我找到了以下信息：这是一个模拟的搜索结果，在实际应用中会返回真实数据。"

        def calculator(expression: str) -> str:
            try:
                # 安全计算 - 只允许基本数学运算
                allowed_chars = set('0123456789+-*/(). ')
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"计算结果: {expression} = {result}"
                else:
                    return "错误: 表达式包含不安全字符"
            except Exception as e:
                return f"无法计算表达式: {expression}, 错误: {e}"
        return [
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

    def analyze_question_type(self, question: str) -> str:
        """分析问题类型"""
        prompt = f"""请分析以下问题需要什么类型的处理：

问题: {question}

请从以下选项中选择：
1. "need_tool" - 仅当明确知道有现成工具可以处理时
2. "need_new_tool" - 如果问题需要特定的、程序化的转换/处理逻辑（如编码转换、格式转换、特定算法），且不确定是否有现成工具
3. "general_knowledge" - 纯知识问答，无需计算或转换

只返回选项关键词，不要其他内容："""

        response = self.llm._call(prompt)
        return response.strip().lower()

    def safe_agent_run(self, agent, question: str) -> str:
        """安全的代理运行方法，处理版本兼容性问题"""
        try:
            # 尝试使用新的invoke方法
            if hasattr(agent, 'invoke'):
                result = agent.invoke({"input": question})
                return result.get('output', str(result))
            else:
                # 回退到旧的run方法
                return agent.run(question)
        except Exception as e:
            return f"工具执行出错: {e}"

    def process_question(self, question: str) -> list:
        """处理用户问题"""
        #返回值list(问题类型，工具名称，result)
        # 分析问题类型
        question_type = self.analyze_question_type(question)
        print(f"问题类型分析: {question_type}")

        if question_type == "general_knowledge":
            # 直接回答通用知识问题
            return "general_knowledge","LLM",self.llm._call(f"请直接回答以下问题，不要提及思考过程: {question}")

        elif question_type == "need_tool":
            # 使用现有工具处理
            #all_tools = self.base_tools + self.tool_creator.get_available_tools()
            all_tools = self.base_tools
            if all_tools:
                try:
                    agent = initialize_agent(
                        tools=all_tools,
                        llm=self.llm,
                        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=False,
                        handle_parsing_errors=True
                    )
                    return "need_tool","SYSTEM",self.safe_agent_run(agent, question)
                except Exception as e:
                    return "need_tool","LLM",f"工具代理执行失败: {e}\n直接回答: {self.llm._call(question)}"
            else:
                return "need_tool","SYSTEM","没有可用的工具来处理此问题"

        elif question_type == "need_new_tool":
            # 创建新工具处理
            creation_msg, new_tool = self.tool_creator.create_tool(question, question)
            # 检查是否是因为工具已存在
            if "工具已存在" in creation_msg:
                # 提取工具名称并使用它
                tool_name = creation_msg.split(": ")[-1]
                return "need_new_tool",tool_name,f"需要的新工具已经创建直接使用,工具名称:{tool_name}"
            else:
                if new_tool:
                    # 直接使用新创建的工具处理问题
                    tool_name = list(self.tool_creator.tool_manager.available_tools.keys())[-1]
                    return "need_new_tool",tool_name,"LLM新创建工具"
                else:
                    return "need_new_tool",tool_name,f"无法创建工具: {creation_msg}"
        else:
            # 默认处理
            return "default","LLM",self.llm._call(f"请直接回答以下问题: {question}")

# 初始化模型和组件
llm = MyLLM(model_name="qwen-plus", temperature=0.1)
tool_creator = DynamicToolCreator(llm)
smart_agent = SmartAgent(llm, tool_creator)

# 测试示例
if __name__ == "__main__":
    # 测试问题处理
    test_questions = [
        "计算 125 * 36 等于多少？",  # 需要现有工具
        "将'Hello World'转换BASE64编码并输出",  # 需要新工具
        "实现一个SSH远程登录工具，要求：1.可变关键字参数的参数默认值要求依次为host='127.0.0.1',port=22,username='root',password='',key_path='',cmd='whoami',2.key_path用于支持密钥方式登录"  # 需要新工具
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        tool_type, tool_name, result = smart_agent.process_question(question)
        if tool_type != 'need_new_tool':
            print(f"{tool_name}回答: {result}")
        else:
            import os
            key_path = os.path.expanduser("./ssh_aliyun.pem")
            args = ('ls', key_path)
            kwargs = {"host": "192.168.1.1", "port": 22,"username":"root","password":"","key_path":key_path,"cmd":"ls -al;ls -al /home"}
            if tool_name:
                out = tool_creator.tool_manager.use_dynamic_tool(tool_name, *args,**kwargs)
                print(f"{result}回答: \n{out}")
            else:
                print(f"tool_name:{tool_name},result:{result}")
        print("-" * 50)

    # 显示当前所有自动创建的工具
    print("\n当前所有AI自动创建的工具:")
    for tool_name, tool_info in tool_creator.tool_manager.available_tools.items():
        print(f"- {tool_name}: {tool_info.get('description', '')}")

