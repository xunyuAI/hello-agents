"""
完善版数学计算工具（基于第7章自定义工具示例扩展）

相对教材简易版的改进：
1. 补齐一元正负号（如 -3、+5），否则复合表达式易失败
2. 增加幂运算 **、取模 %、整除 //
3. 扩充常用数学函数与常数（sin/cos/log/e 等）
4. 明确错误信息（除零、未知函数、语法错误），避免裸 except
5. 自带与教材一致的测试用例，可直接运行验证
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from hello_agents import ToolRegistry

# 二元运算符：AST 节点类型 -> 实际计算函数
BINARY_OPERATORS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,  # +
    ast.Sub: operator.sub,  # -
    ast.Mult: operator.mul,  # *
    ast.Div: operator.truediv,  # /
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,  # %
    ast.Pow: operator.pow,  # **
}

# 一元运算符
UNARY_OPERATORS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,  # +x
    ast.USub: operator.neg,  # -x
}

# 可调用函数（出现在 sqrt(16) 这类调用中）
MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,  # 默认自然对数；log(x, base) 也支持
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

# 常数（出现在表达式里的裸名字，如 pi、e）
MATH_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def my_calculate(expression: str) -> str:
    """安全求值数学表达式，返回字符串结果或错误说明。"""
    if not isinstance(expression, str) or not expression.strip():
        return "计算表达式不能为空"

    expression = expression.strip()

    try:
        node = ast.parse(expression, mode="eval")
        result = _eval_node(node.body)
        # 整数结果尽量显示为整型字符串，避免 5.0 造成困扰（除法仍保留浮点）
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "计算失败：除数不能为 0"
    except SyntaxError as exc:
        return f"计算失败：表达式语法错误（{exc.msg}）"
    except ValueError as exc:
        return f"计算失败：{exc}"
    except TypeError as exc:
        return f"计算失败：参数类型或个数不正确（{exc}）"
    except Exception as exc:  # 兜底，便于排查未知问题
        return f"计算失败：{exc}"


def _eval_node(node: ast.AST) -> Any:
    """递归计算 AST 节点（仅允许白名单运算，避免任意代码执行）。"""
    # 数字字面量：2、3.14
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    # 二元运算：a + b、a * b ...
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op = BINARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op(left, right)

    # 一元运算：-x、+x
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op = UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(operand)

    # 函数调用：sqrt(16)、sin(pi/2)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持简单函数名调用，如 sqrt(16)")
        func_name = node.func.id
        func = MATH_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"不支持的函数: {func_name}")
        if node.keywords:
            raise ValueError("不支持关键字参数，请使用位置参数")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)

    # 名字引用：pi、e
    if isinstance(node, ast.Name):
        if node.id in MATH_CONSTANTS:
            return MATH_CONSTANTS[node.id]
        if node.id in MATH_FUNCTIONS:
            raise ValueError(f"'{node.id}' 是函数，请写成 {node.id}(...)")
        raise ValueError(f"未定义的变量: {node.id}")

    raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


def create_calculator_registry() -> ToolRegistry:
    """创建包含完善版计算器的工具注册表。"""
    registry = ToolRegistry()
    registry.register_function(
        name="my_calculator",
        description=(
            "数学计算工具，支持 + - * / // % **、一元正负号，"
            "以及 sqrt/sin/cos/tan/log/exp/abs/round 等函数和 pi/e 常数。"
            "示例：2+3、sqrt(16)、sqrt(16)+2*3、sin(pi/2)"
        ),
        func=my_calculate,
    )
    return registry


def _run_doc_test_cases() -> None:
    """用教材中的测试用例验证核心功能。"""
    registry = create_calculator_registry()

    # 教材 test_calculator_tool 中的用例 + 集成测试中的复合表达式
    test_cases = [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("5 * 6", "30"),
        ("15 / 3", "5"),
        ("sqrt(16)", "4"),
        ("sqrt(16) + 2 * 3", "10"),  # 教材 SimpleAgent 集成场景
    ]

    # 完善版额外能力（文档提到可自行扩展）
    extra_cases = [
        ("2 ** 3", "8"),
        ("-5 + 3", "-2"),
        ("17 % 5", "2"),
        ("17 // 5", "3"),
        ("abs(-7)", "7"),
        ("round(3.14159, 2)", "3.14"),
        ("sin(pi / 2)", "1"),
        ("log(e)", "1"),
        ("", "计算表达式不能为空"),
        ("1 / 0", "计算失败：除数不能为 0"),
        ("unknown(1)", "计算失败：不支持的函数: unknown"),
    ]

    print("🧪 教材测试用例\n")
    _assert_cases(registry, test_cases)

    print("✨ 完善版扩展用例\n")
    _assert_cases(registry, extra_cases)

    print("✅ 全部用例通过")


def _assert_cases(registry: ToolRegistry, cases: list[tuple[str, str]]) -> None:
    for i, (expression, expected) in enumerate(cases, 1):
        result = registry.execute_tool("my_calculator", expression)
        # ToolRegistry.execute_tool 返回 ToolResponse，优先取 text
        result_text = getattr(result, "text", None) or str(result)
        result_text = str(result_text).strip()
        ok = result_text == expected or expected in result_text
        status = "PASS" if ok else "FAIL"
        shown = expression if expression else "(空字符串)"
        print(f"测试 {i}: {shown}")
        print(f"  期望: {expected}")
        print(f"  实际: {result_text}")
        print(f"  [{status}]\n")
        if not ok:
            raise AssertionError(f"用例失败: {shown!r} -> {result_text!r}, 期望 {expected!r}")


if __name__ == "__main__":
    _run_doc_test_cases()
