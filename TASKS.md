# AI Factory Coder Agent 完善任务清单

## 任务1：创建重试装饰器
文件：src/common/retry.py

实现 smart_retry 装饰器：
- 支持 max_retries 参数（默认3）
- 支持 backoff_factor 参数（指数退避：1秒、2秒、4秒）
- 支持 fallback_model 参数（重试时切换模型）
- 记录每次重试的日志

## 任务2：创建 LLM 连接池
文件：src/execution/llm_pool.py

实现 LLMPool 类：
- max_concurrent=4（最多4个并发）
- max_queue_size=20（队列最大20）
- timeout=120（超时120秒）
- execute() 方法：Semaphore控制并发
- get_stats() 方法：返回状态统计

## 任务3：创建工具注册表
文件：src/execution/tools_registry.py

实现 ToolsRegistry 类：
- 工具保存目录：/tmp/ai_factory/tools/
- register_tool()：注册工具
- get_tool()：获取工具
- list_tools()：列出工具
- load_tool_module()：动态加载工具模块

## 任务4：增强 Coder Agent
文件：src/agents/coder.py

添加功能：
- PRIMARY_MODEL = "Qwen3.6-35B-A3B-UD-Q5_K_M"
- FALLBACK_MODEL = "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"
- generate_with_fallback()：双模型切换
- validate_code()：代码验证
- generate_tool_for_agent()：为其他Agent生成工具
- 集成 LLMPool 和 ToolsRegistry

## 任务5：更新配置文件
文件：src/config/settings.py

添加配置项：
- llm_max_concurrent: int = 4
- llm_timeout: int = 120
- tools_dir: str = "/tmp/ai_factory/tools"

## 执行要求
- 每个文件完成后，保持原有功能不变
- 添加完整的类型注解
- 添加 docstring 注释
- 保持与现有代码风格一致
