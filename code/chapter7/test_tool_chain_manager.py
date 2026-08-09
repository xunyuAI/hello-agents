# test_tool_chain_manager.py
"""
7.5.4（1）工具链测试

不依赖真实搜索 API：用本地 mock 工具验证
- context / 模板变量替换
- 步骤顺序与 output_key 传递
- ToolChainManager 注册与调度
- 三步可运行链（对应习题「≥3 工具」）
"""

from __future__ import annotations

import re
import time

from hello_agents import ToolRegistry

from my_calculator_tool import my_calculate
from tool_chain_manager import (
    ToolChain,
    ToolChainManager,
    create_demo_three_step_chain,
    create_research_chain,
)


def _mock_search(query: str) -> str:
    """模拟搜索：把查询包装成带表达式的「检索结果」。"""
    # 故意让结果像自然语言，方便下一步 extract_expr 抽取
    known = {
        "机器学习算法": "检索摘要：常见复杂度记号；示例表达式 sqrt(16) + 2 * 3",
        "圆面积": "半径为 3 时，面积表达式为 pi * 3 * 3（此处用 3.14*3*3 近似）",
    }
    if query in known:
        return known[query]
    return f"检索摘要：关于「{query}」的结果；示例表达式 2 + 3"


def _extract_expr(text: str) -> str:
    """从搜索结果里抽取「像算式」的片段，供 AST 计算器使用。"""
    # 优先匹配含 sqrt(...) 的完整式；否则匹配多操作数算术式
    patterns = [
        r"sqrt\(\d+\)(?:\s*[+\-*/]\s*(?:sqrt\(\d+\)|\d+(?:\.\d+)?))+",
        r"\d+(?:\.\d+)?(?:\s*[+\-*/]\s*(?:sqrt\(\d+\)|\d+(?:\.\d+)?))+",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(re.findall(pat, text))
    if candidates:
        # 取最长匹配，避免只抽到 "2 * 3" 而丢掉前面的 sqrt(16)
        return max(candidates, key=len).strip()
    stripped = text.strip()
    if re.fullmatch(r"[0-9.\s+\-*/()asqrtpi]+", stripped, flags=re.I):
        return stripped
    return "2 + 2"


def create_mock_registry() -> ToolRegistry:
    """注册 search / extract_expr / my_calculator，无需外部 API。"""
    registry = ToolRegistry()
    registry.register_function(
        name="search",
        description="模拟搜索工具（教学用，不访问外网）",
        func=_mock_search,
    )
    registry.register_function(
        name="extract_expr",
        description="从文本中抽取数学表达式",
        func=_extract_expr,
    )
    registry.register_function(
        name="my_calculator",
        description="安全数学计算（AST）",
        func=my_calculate,
    )
    return registry


def test_add_step_and_default_output_key():
    """步骤结构与默认 output_key"""
    print("=" * 60)
    print("测试 1: add_step 与默认 output_key")
    print("=" * 60)

    chain = ToolChain("demo", "结构检查")
    # 第 1 步：不传 output_key → 默认生成 step_0_result
    chain.add_step("search", "{input}")
    # 第 2 步：显式传 output_key="calc"；模板引用上一步默认键
    chain.add_step("my_calculator", "{step_0_result}", output_key="calc")

    # assert：条件为真就继续；为假立刻抛 AssertionError，测试失败。
    # 本组断言只检查「步骤怎么存进 steps」，不跑工具——盯的是数据结构/约定。
    assert len(chain.steps) == 2  # add_step 真的加了两步，没有漏加/多加
    # 没传 output_key 时，默认按索引生成 step_{n}_result（n 从 0 起）
    assert chain.steps[0]["output_key"] == "step_0_result"
    # 显式传了 output_key="calc" 时，用你给的名字，不用默认名
    assert chain.steps[1]["output_key"] == "calc"
    # 保证后续模板写 {step_0_result}/{calc} 时，键名规则和实现一致
    print("✅ 步骤字段与默认键名正确\n")


def test_missing_template_variable():
    """模板缺变量时应友好失败"""
    print("=" * 60)
    print("测试 2: 模板变量缺失")
    print("=" * 60)

    registry = create_mock_registry()
    chain = ToolChain("bad", "缺变量")
    # 模板引用 context 里不存在的键；执行时 context 通常只有 input（及后续结果键）
    chain.add_step("search", "{not_exist}")

    result = chain.execute(registry, "hello")
    # 本组断言盯的是「异常路径的返回内容」，不是搜索是否成功：
    # 1) format(**context) 缺键时，链应被 catch，友好失败而不是硬崩
    # 2) 返回约定错误文案（含「模板变量」「未找到」），方便上层/调试识别
    assert "模板变量" in result and "未找到" in result
    print(f"结果: {result}")
    print("✅ 缺变量时正确返回错误信息\n")


def test_manager_register_and_list():
    """ToolChainManager 注册 / 列表 / 不存在链"""
    print("=" * 60)
    print("测试 3: ToolChainManager")
    print("=" * 60)

    registry = create_mock_registry()
    manager = ToolChainManager(registry)
    chain = create_demo_three_step_chain()
    manager.register_chain(chain)

    assert "search_extract_calculate" in manager.list_chains()
    missing = manager.execute_chain("no_such_chain", "x")
    assert "不存在" in missing
    print(f"已注册链: {manager.list_chains()}")
    print(f"不存在链: {missing}")
    print("✅ Manager 行为正确\n")


def test_three_step_chain_end_to_end():
    """三步链端到端：search → extract_expr → my_calculator"""
    print("=" * 60)
    print("测试 4: 三步可运行链（教学主路径）")
    print("=" * 60)

    registry = create_mock_registry()
    manager = ToolChainManager(registry)
    manager.register_chain(create_demo_three_step_chain())

    result = manager.execute_chain("search_extract_calculate", "机器学习算法")
    # mock 搜索里含 sqrt(16) + 2 * 3 → 4 + 6 = 10
    print(f"最终结果: {result}")
    assert result.strip() == "10.0" or result.strip() == "10"
    print("✅ 三步链结果正确（期望 10）\n")


def test_textbook_research_chain_structure():
    """教材 create_research_chain 的结构（search → my_calculator）"""
    print("=" * 60)
    print("测试 5: 教材 research_and_calculate 结构")
    print("=" * 60)

    chain = create_research_chain()
    assert chain.name == "research_and_calculate"
    assert [s["tool_name"] for s in chain.steps] == ["search", "my_calculator"]
    assert chain.steps[0]["input_template"] == "{input}"
    assert chain.steps[0]["output_key"] == "search_result"
    assert "{search_result}" in chain.steps[1]["input_template"]
    print("✅ 与教材步骤定义一致")
    print("说明: 教材第 2 步模板含中文说明，需「能理解自然语言」的计算工具；")
    print("      AST 计算器请用测试 4 的 extract 中间步，或把模板改成 {search_result}。\n")


def test_two_step_with_pure_expr_template():
    """两步链但用纯表达式模板，证明 context 传递本身没问题"""
    print("=" * 60)
    print("测试 6: 两步链 + 纯表达式模板")
    print("=" * 60)

    registry = create_mock_registry()

    def search_returns_expr(q: str) -> str:
        return "sqrt(16) + 2 * 3"

    registry.register_function(
        name="search",
        description="直接返回表达式",
        func=search_returns_expr,
    )

    chain = ToolChain("expr_chain", "搜索即表达式")
    chain.add_step("search", "{input}", "search_result")
    chain.add_step("my_calculator", "{search_result}", "calculation_result")

    result = chain.execute(registry, "任意查询")
    print(f"最终结果: {result}")
    assert result.strip() in ("10", "10.0")
    print("✅ context 传递与顺序执行正确\n")


def test_chain_is_sequential_not_parallel():
    """工具链是串行的：总耗时应接近各步之和（对比异步节）"""
    print("=" * 60)
    print("测试 7: 串行耗时特征（对比异步并行）")
    print("=" * 60)

    registry = ToolRegistry()

    def slow_a(x: str) -> str:
        time.sleep(0.15)
        return f"A:{x}"

    def slow_b(x: str) -> str:
        time.sleep(0.15)
        return f"B:{x}"

    registry.register_function(name="slow_a", description="慢工具A", func=slow_a)
    registry.register_function(name="slow_b", description="慢工具B", func=slow_b)

    chain = ToolChain("seq", "串行")
    chain.add_step("slow_a", "{input}", "a")
    chain.add_step("slow_b", "{a}", "b")

    t0 = time.perf_counter()
    result = chain.execute(registry, "x")
    elapsed = time.perf_counter() - t0

    print(f"结果: {result}, 耗时: {elapsed:.3f}s")
    assert result == "B:A:x"
    assert elapsed >= 0.28, "串行应接近两段 sleep 之和"
    print("✅ 串行依赖：有数据依赖时必须用链，不能盲目并行\n")


if __name__ == "__main__":
    test_add_step_and_default_output_key()
    test_missing_template_variable()
    test_manager_register_and_list()
    test_three_step_chain_end_to_end()
    test_textbook_research_chain_structure()
    test_two_step_with_pure_expr_template()
    test_chain_is_sequential_not_parallel()
    print("🎉 全部工具链测试通过")
