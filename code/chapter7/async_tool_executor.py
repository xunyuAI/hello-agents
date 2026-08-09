# async_tool_executor.py
"""7.5.4 异步工具执行：用线程池把同步工具包装成可 await，并支持并行 gather。"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional

from hello_agents import ToolRegistry


def _result_to_text(result: Any) -> str:
    """兼容 ToolResponse / str，统一成文本。"""
    if result is None:
        return ""
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    return str(result)


class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具"""
        loop = asyncio.get_running_loop()

        def _execute():
            return _result_to_text(self.registry.execute_tool(tool_name, input_data))

        result = await loop.run_in_executor(self.executor, _execute)
        return result

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[str]:
        """并行执行多个工具"""
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        # 创建异步任务
        async_tasks = []
        for task in tasks:
            tool_name = task["tool_name"]
            input_data = task["input_data"]
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append(async_task)

        # 等待所有任务完成
        results = await asyncio.gather(*async_tasks)

        print(f"✅ 所有工具任务执行完成")
        return list(results)

    def shutdown(self, wait: bool = True) -> None:
        """显式关闭线程池（比依赖 __del__ 更可靠）。"""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=wait)
            self.executor = None

    def __del__(self):
        """清理资源"""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=False)


# 使用示例（与教材一致；真正注册工具后见 test_async_tool_executor.py）
async def test_parallel_execution():
    """测试并行工具执行"""
    from hello_agents import ToolRegistry

    registry = ToolRegistry()
    # 假设已经注册了搜索和计算工具

    executor = AsyncToolExecutor(registry)

    # 定义并行任务
    tasks = [
        {"tool_name": "search", "input_data": "Python编程"},
        {"tool_name": "search", "input_data": "机器学习"},
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
    ]

    # 并行执行
    results = await executor.execute_tools_parallel(tasks)

    for i, result in enumerate(results):
        preview = result[:100] if isinstance(result, str) else str(result)[:100]
        print(f"任务 {i+1} 结果: {preview}...")

    executor.shutdown()


if __name__ == "__main__":
    # 未注册工具时只会得到「未找到工具」类错误信息；完整演示请运行测试文件
    asyncio.run(test_parallel_execution())
