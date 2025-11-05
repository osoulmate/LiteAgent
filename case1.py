# case_self_verify.py
import os
import json
import re
import time
from typing import Any, Dict, List, Optional
from langchain.agents import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from litellm import completion

assert os.environ['DEEPSEEK_API_KEY'], "请先在环境变量里设置 DEEPSEEK_API_KEY"

class MyLLM(LLM):
    """LangChain 包装 DeepSeek 模型（官方 deepseek 包）"""
    model_name: str = "deepseek/deepseek-reasoner"   # 或 deepseek-coder
    temperature: float = 0
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        resp = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        return resp.choices[0].message.content.strip()


# ---------------- 动态工具工厂（自修复版） ----------------
class SelfFixedToolCreator:
    def __init__(self, llm: LLM, max_loop: int = 3):
        self.llm = llm
        self.registry = {}
        self.max_loop = max_loop

    # ---- 1. 初始生成 ----
    def _generate_v1(self, user_request: str) -> str:
        tmpl = PromptTemplate(
            template="""你是 Python 工具专家。按需求生成一个安全、独立的函数。

需求: {user_request}

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
        return (tmpl | self.llm).invoke({"user_request": user_request}).strip()

    # ---- 2. 生成单测 ----
    def _generate_tests(self, func_code: str) -> List[dict]:
        tmpl = PromptTemplate(
            template="""给定函数，生成 3 组单元测试（输入/期望输出 JSON 列表）。

函数:
{func_code}

返回格式:
[
  {{"input": "参数", "expected": "期望输出"}},
  ...
只返回 JSON，勿多余解释。
""",
            input_variables=["func_code"]
        )
        txt = (tmpl | self.llm).invoke({"func_code": func_code}).strip()
        try:
            return json.loads(txt)
        except Exception:
            return []

    # ---- 3. 静态安全过滤 ----
    def _is_safe(self, code: str) -> bool:
        blk = ["import", "os.", "sys.", "eval", "exec", "open", "socket", "request", "subprocess"]
        return not any(re.search(rf"\b{w}\b", code) for w in blk)

    # ---- 4. 执行+单测 ----
    def _validate(self, func_code: str, tests: List[dict]) -> Optional[str]:
        ns = {"__builtins__": {}}
        try:
            exec(func_code, ns)
        except Exception as e:
            return f"语法错误: {e}"
        func = next((v for k, v in ns.items() if callable(v) and not k.startswith("_")), None)
        if not func:
            return "找不到函数"
        for t in tests:
            try:
                if func(t["input"]) != t["expected"]:
                    return f'测试失败: in="{t["input"]}" exp="{t["expected"]}"'
            except Exception as e:
                return f"运行错误: {e}"
        return None

    # ---- 5. 自修复循环 ----
    def _loop_fix(self, user_request: str, func_code: str, tests: List[dict]) -> Optional[str]:
        for i in range(self.max_loop):
            err = self._validate(func_code, tests)
            if err is None:
                print(f"第{i+1}轮生成的代码通过测试验证...")
                return func_code  # 成功
            else:
                print(f"第{i+1}轮生成的代码验证出错...\n{err}")
            # 构造修复 prompt
            fix_tmpl = PromptTemplate(
                template="""你之前写的 Python 函数运行失败，请基于错误信息输出**完整修正后的代码**（非 diff）。

需求: {user_request}
当前代码:
{func_code}
错误/测试失败: {error}
要求：
1. 函数名不变
2. 包含详细的docstring说明
3. 返回字符串结果
4. 不要使用危险操作（如文件删除、网络请求等）
5. 生成的代码不含诸如```这样的markdown风格的代码格式化字符

修正后代码:
""",
                input_variables=["user_request", "func_code", "error"]
            )
            func_code = (fix_tmpl | self.llm).invoke({
                "user_request": user_request,
                "func_code": func_code,
                "error": err
            }).strip()
            if not self._is_safe(func_code):
                continue
        return None  # 修复失败

    # ---- 6. 对外接口 ----
    def create_tool(self, user_request: str):
        print("[1] 生成 V1 代码...")
        v1 = self._generate_v1(user_request)
        print(f"{v1}")
        if not self._is_safe(v1):
            return "未通过安全过滤", None
        tests = self._generate_tests(v1)
        if not tests:
            return "测试用例生成失败", None
        else:
            print("测试用例生成结果...")
            print(tests)

        print("[2] 开始自修复循环...")
        final_code = self._loop_fix(user_request, v1, tests)
        print(f"生成的代码:\n{final_code}")
        if final_code is None:
            return "自修复失败", None

        # 注册
        ns = {"__builtins__": {}}
        exec(final_code, ns)
        func = next((v for k, v in ns.items() if callable(v) and not k.startswith("_")), None)
        name = f"dynamic_tool_{len(self.registry)+1}"
        self.registry[name] = func
        return f"工具 `{name}` 创建并自修复完成！", func

    # ---- 7. 调用 ----
    def invoke(self, name: str, *a, **kw):
        if name not in self.registry:
            return f"工具 {name} 不存在"
        try:
            return self.registry[name](*a, **kw)
        except Exception as e:
            return f"执行出错: {e}"


# ----------------- 演示 -----------------
if __name__ == "__main__":
    llm = MyLLM(model_name="deepseek/deepseek-reasoner", temperature=0.1)
    factory = SelfFixedToolCreator(llm, max_loop=3)

    need = "将英文文本转为摩斯密码，只返回摩斯字符串"
    msg, tool = factory.create_tool(need)
    print(msg)
    if tool:
        print("测试样例:", factory.invoke("dynamic_tool_1", "SOS"))