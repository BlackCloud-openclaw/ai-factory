import pathlib

def get_tool_info():
    return {
        "name": "text_file_reader",
        "description": "读取指定路径的文本文件并返回其内容",
        "module_path": "",
        "function_name": "read_text_file",
        "parameters": {
            "file_path": {"type": "string", "description": "文本文件的路径"}
        }
    }

def read_text_file(file_path: str) -> str:
    """
    读取指定路径的文本文件并返回其内容。

    参数:
    file_path (str): 文本文件的路径

    返回:
    str: 文件的内容

    异常:
    FileNotFoundError: 如果文件路径不存在
    ValueError: 如果文件路径不是一个有效的文本文件
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件路径 {file_path} 不存在")
    if not path.is_file() or path.suffix != '.txt':
        raise ValueError(f"文件路径 {file_path} 不是一个有效的文本文件")
    
    with path.open('r', encoding='utf-8') as file:
        content = file.read()
    
    return content