import json

def get_tool_info():
    return {
        "name": "json_field_extractor",
        "description": "从JSON字符串中提取指定字段的值",
        "module_path": "",
        "function_name": "extract_field",
        "parameters": {
            "json_string": {"type": "string", "description": "JSON格式的字符串"},
            "field_name": {"type": "string", "description": "要提取的字段名"}
        }
    }

def extract_field(json_string: str, field_name: str):
    """
    从JSON字符串中提取指定字段的值。

    参数:
    json_string (str): JSON格式的字符串。
    field_name (str): 要提取的字段名。

    返回:
    该字段的值，可以是任何JSON支持的数据类型。

    异常:
    ValueError: 如果输入的字符串不是有效的JSON格式。
    KeyError: 如果指定的字段名在JSON中不存在。
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        raise ValueError("输入的字符串不是有效的JSON格式")
    
    try:
        return data[field_name]
    except KeyError:
        raise KeyError(f"指定的字段名 '{field_name}' 在JSON中不存在")