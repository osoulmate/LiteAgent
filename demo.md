```
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, full_write_guard
from RestrictedPython.PrintCollector import PrintCollector

def safe_execute_code(code: str, func_name: str, *args, **kwargs):
    """
    在受限环境中执行 AI 生成的代码
    """
    # 1. 准备受限的全局变量
    # 我们只给它最基础的内置函数，禁用了 __import__, open, eval 等
    local_vars = {}
    byte_code = compile_restricted(code, filename='<inline_tool>', mode='exec')
    
    # 2. 这里的 _print_ 是为了捕获代码里的 print 语句（可选）
    # _write_ 则是为了限制对属性的写入
    restricted_globals = {
        '__builtins__': safe_builtins,
        '_print_': PrintCollector,
        '_write_': full_write_guard,
        'math': __import__('math'), # 如果需要，可以手动注入安全的库
    }

    try:
        # 执行定义函数的代码
        exec(byte_code, restricted_globals, local_vars)
        
        # 获取生成的函数对象
        func = local_vars.get(func_name)
        if not func:
            return "错误：未能找到指定的函数名"

        # 3. 执行函数并返回结果
        result = func(*args, **kwargs)
        return result

    except Exception as e:
        return f"沙箱执行安全拦截或错误: {str(e)}"
``
