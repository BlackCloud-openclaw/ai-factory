def get_tool_info():
    return {
        "name": "calculator",
        "description": "提供加、减、乘、除四则运算",
        "module_path": "",
        "function_name": "calculate",
        "parameters": {
            "operation": {"type": "string", "enum": ["add", "sub", "mul", "div"], "description": "运算类型"},
            "a": {"type": "number", "description": "第一个数"},
            "b": {"type": "number", "description": "第二个数"}
        }
    }

def calculate(operation: str, a: float, b: float) -> float:
    """
    执行指定的四则运算。

    参数:
    operation (str): 运算类型，支持 "add", "sub", "mul", "div"。
    a (float): 第一个数。
    b (float): 第二个数。

    返回:
    float: 运算结果。

    异常:
    ValueError: 当运算类型不在支持范围内时抛出。
    ZeroDivisionError: 当执行除法运算且第二个数为零时抛出。
    """
    if operation == "add":
        return a + b
    elif operation == "sub":
        return a - b
    elif operation == "mul":
        return a * b
    elif operation == "div":
        if b == 0:
            raise ZeroDivisionError("除数不能为零")
        return a / b
    else:
        raise ValueError("不支持的运算类型")