import urllib.request

def get_tool_info():
    return {
        "name": "http_get_request",
        "description": "发送HTTP GET请求并返回响应文本",
        "module_path": "",
        "function_name": "send_get_request",
        "parameters": {
            "url": {"type": "string", "description": "请求的URL地址"}
        }
    }

def send_get_request(url: str) -> str:
    """
    发送HTTP GET请求到指定的URL，并返回响应的文本内容。

    参数:
    url (str): 请求的URL地址。

    返回:
    str: 响应的文本内容。

    异常:
    ValueError: 如果URL格式不正确。
    urllib.error.URLError: 如果请求过程中发生错误。
    """
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except ValueError as e:
        raise ValueError(f"URL格式不正确: {e}")
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"请求过程中发生错误: {e}")