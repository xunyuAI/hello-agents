# test_async_tool_executor.py
"""
7.5.4（2）异步工具执行测试

验证：
- 单工具 async 包装
- 多任务并行 gather（I/O 等待可重叠）
- 与串行对比的耗时优势
- 无真实 API 依赖（本地 sleep / 计算 mock）
"""

from __future__ import annotations

import asyncio
import time

from hello_agents import ToolRegistry

from async_tool_executor import AsyncToolExecutor
from my_calculator_tool import my_calculate


def _mock_search(query: str) -> str:
    """模拟「网络搜索」：阻塞 sleep，模拟 I/O 等待。"""
    time.sleep(0.2)
    return f"搜索结果:{query}"


def create_async_demo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        name="search",
        description="模拟慢搜索（sleep 模拟网络）",
        func=_mock_search,
    )
    registry.register_function(
        name="my_calculator",
        description="数学计算",
        func=my_calculate,
    )
    return registry


async def test_single_tool_async():
    print("=" * 60)
    print("测试 1: 单工具异步执行")
    print("=" * 60)

    registry = create_async_demo_registry()
    executor = AsyncToolExecutor(registry, max_workers=2)

    result = await executor.execute_tool_async("my_calculator", "2 + 2")
    print(f"结果: {result}")
    assert result.strip() in ("4", "4.0")

    executor.shutdown()
    print("✅ 单工具异步包装正常\n")


async def test_parallel_faster_than_serial():
    """4 次慢搜索：并行应明显快于串行（I/O 重叠）。"""
    print("=" * 60)
    print("测试 2: 并行 vs 串行耗时")
    print("=" * 60)

    registry = create_async_demo_registry()
    tasks = [
        {"tool_name": "search", "input_data": "Python编程"},
        {"tool_name": "search", "input_data": "机器学习"},
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
    ]

    # 串行
    t0 = time.perf_counter()
    serial_results = []
    for task in tasks:
        raw = registry.execute_tool(task["tool_name"], task["input_data"])
        text = getattr(raw, "text", None) or str(raw)
        serial_results.append(str(text))
    serial_elapsed = time.perf_counter() - t0

    # 并行
    executor = AsyncToolExecutor(registry, max_workers=4)
    t1 = time.perf_counter()
    parallel_results = await executor.execute_tools_parallel(tasks)
    parallel_elapsed = time.perf_counter() - t1
    executor.shutdown()

    print(f"串行耗时: {serial_elapsed:.3f}s")
    print(f"并行耗时: {parallel_elapsed:.3f}s")
    for i, (s, p) in enumerate(zip(serial_results, parallel_results), 1):
        print(f"  任务{i}: serial={s[:40]!r} | parallel={p[:40]!r}")

    assert len(parallel_results) == 4
    assert "Python编程" in parallel_results[0]
    assert parallel_results[2].strip() in ("4", "4.0")
    assert parallel_results[3].strip() in ("4", "4.0")
    # 两次 0.2s sleep 串行 ≈0.4+，并行应明显更短
    assert parallel_elapsed < serial_elapsed * 0.85
    print("✅ 独立 I/O 任务并行可提速\n")


async def test_textbook_parallel_api_shape():
    """与教材 test_parallel_execution 相同的任务结构"""
    print("=" * 60)
    print("测试 3: 教材并行任务结构")
    print("=" * 60)

    registry = create_async_demo_registry()
    executor = AsyncToolExecutor(registry)

    tasks = [
        {"tool_name": "search", "input_data": "Python编程"},
        {"tool_name": "search", "input_data": "机器学习"},
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
    ]

    results = await executor.execute_tools_parallel(tasks)
    for i, result in enumerate(results):
        print(f"任务 {i+1} 结果: {result[:100]}...")

    executor.shutdown()
    assert len(results) == 4
    print("✅ 教材 API 形状可运行\n")


async def test_dependent_steps_should_not_parallelize():
    """
    有数据依赖时不应并行：先 A 再 B（B 依赖 A 的输出）。
    这里用「链思路」对比说明：并行 gather 无法表达依赖。
    """
    print("=" * 60)
    print("测试 4: 有依赖时并行不适用（概念验证）")
    print("=" * 60)

    registry = ToolRegistry()

    def step_a(x: str) -> str:
        time.sleep(0.05)
        return f"parsed:{x}"

    def step_b(x: str) -> str:
        # 依赖 A 的输出格式
        assert x.startswith("parsed:"), "B 必须吃到 A 的输出"
        return f"done:{x}"

    registry.register_function(name="step_a", description="A", func=step_a)
    registry.register_function(name="step_b", description="B", func=step_b)

    executor = AsyncToolExecutor(registry)
    # 正确做法：先 await A，再用结果调 B（顺序）
    mid = await executor.execute_tool_async("step_a", "query")
    final = await executor.execute_tool_async("step_b", mid)
    executor.shutdown()

    print(f"A→B 顺序结果: {final}")
    assert final == "done:parsed:query"
    print("✅ 有依赖用顺序（工具链）；无依赖才 gather 并行\n")


async def main():
    await test_single_tool_async()
    await test_parallel_faster_than_serial()
    await test_textbook_parallel_api_shape()
    await test_dependent_steps_should_not_parallelize()
    print("🎉 全部异步工具测试通过")


if __name__ == "__main__":
    asyncio.run(main())
