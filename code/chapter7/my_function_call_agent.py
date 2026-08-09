# my_function_call_agent.py
"""
框架化 FunctionCallAgent 实现（对应第 7.4.5 节）

对比 MySimpleAgent（用 Prompt 约束 `[TOOL_CALL:...]`）：
本实现走 OpenAI 原生 Function Calling —— 模型返回结构化 tool_calls，
框架侧负责 schema 构建、参数解析、类型转换与多轮工具循环。

说明：hello-agents ≥1.0 已将同类能力并入官方 SimpleAgent；
本文件按教材拆解关键步骤，便于学习与对照测试。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from hello_agents import HelloAgentsLLM, Config, Message, ToolRegistry
from hello_agents.core.agent import Agent


class MyFunctionCallAgent(Agent):
    """基于原生 Function Calling 的 Agent（教学版）"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_tool_iterations: int = 5,
        tool_choice: Union[str, dict] = "auto",
    ):
        super().__init__(name, llm, system_prompt, config, tool_registry=tool_registry)
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations
        self.tool_choice = tool_choice
        print(f"✅ {name} 初始化完成（Function Calling），最大工具轮次: {max_tool_iterations}")

    # ------------------------------------------------------------------
    # 教材核心能力 1：工具 description → OpenAI function calling schema
    # ------------------------------------------------------------------
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """通过工具的 description / 参数定义，构建 OpenAI tools schema。"""
        if not self.tool_registry:
            return []

        schemas: List[Dict[str, Any]] = []

        # 1) Tool 对象（完整参数定义）
        for tool in self.tool_registry.get_all_tools():
            # 优先使用工具自带的 to_openai_schema（见教材 7.5.1）
            if hasattr(tool, "to_openai_schema"):
                schemas.append(tool.to_openai_schema())
                continue

            properties: Dict[str, Any] = {}
            required: List[str] = []
            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": self._map_parameter_type(param.type),
                    "description": param.description or "",
                }
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # 2) 函数工具（register_function 注册，简化为单 input 字符串）
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本",
                                }
                            },
                            "required": ["input"],
                        },
                    },
                }
            )

        return schemas

    @staticmethod
    def _map_parameter_type(param_type: str) -> str:
        """将工具参数类型映射为 JSON Schema 允许的类型。"""
        normalized = (param_type or "").lower()
        if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
            return normalized
        return "string"

    # ------------------------------------------------------------------
    # 教材核心能力 2：从 OpenAI 响应中提取文本
    # ------------------------------------------------------------------
    def _extract_message_content(self, message: Any) -> str:
        """从 OpenAI message 对象中提取纯文本内容。"""
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if content is None:
            return ""
        return str(content)

    # ------------------------------------------------------------------
    # 教材核心能力 3：解析模型返回的 JSON 字符串参数
    # ------------------------------------------------------------------
    def _parse_function_call_arguments(self, arguments: Union[str, dict, None]) -> Dict[str, Any]:
        """解析 function call 的 arguments（通常是 JSON 字符串）。"""
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return {}

        text = arguments.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            # 容错：模型偶发返回非 JSON，当作单字符串输入
            return {"input": text}

    # ------------------------------------------------------------------
    # 教材核心能力 4：按工具定义转换参数类型
    # ------------------------------------------------------------------
    def _convert_parameter_types(
        self, tool_name: str, param_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """根据工具参数定义，把模型给出的字符串等值转成正确 Python 类型。"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: Dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    # ------------------------------------------------------------------
    # 教材核心能力：封装底层 chat.completions.create(..., tools=...)
    # ------------------------------------------------------------------
    def _invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, dict],
        **kwargs,
    ):
        """
        调用底层客户端执行函数调用。

        教材写法直接访问 self.llm._client；当前 HelloAgentsLLM 已提供
        invoke_with_tools，内部同样是对 OpenAI 兼容 API 的再封装。
        """
        # 优先走官方封装（兼容更多 provider adapter）
        if hasattr(self.llm, "invoke_with_tools"):
            return self.llm.invoke_with_tools(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        # 回退：与教材一致，直接打原生客户端
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", getattr(self.llm, "temperature", None))
        max_tokens = getattr(self.llm, "max_tokens", None)
        if max_tokens is not None:
            client_kwargs.setdefault("max_tokens", max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行一次工具调用，返回字符串结果。"""
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_args = self._convert_parameter_types(tool_name, arguments)
                # 新版工具可能返回 ToolResponse；旧版返回 str
                result = tool.run(typed_args)
                if hasattr(result, "text"):
                    return str(result.text)
                return str(result)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                if not input_text and len(arguments) == 1:
                    input_text = next(iter(arguments.values()))
                result = self.tool_registry.execute_tool(tool_name, str(input_text))
                if hasattr(result, "text"):
                    return str(result.text)
                return str(result)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        return f"❌ 错误：未找到工具 '{tool_name}'"

    # ------------------------------------------------------------------
    # 主循环：原生 Function Calling 多轮
    # ------------------------------------------------------------------
    def run(self, input_text: str, **kwargs) -> str:
        """运行 FunctionCallAgent。"""
        print(f"\n🤖 {self.name} 开始处理: {input_text}")

        messages: List[Dict[str, Any]] = []
        system_prompt = self.system_prompt or (
            "你是一个智能助手。需要时请通过函数调用使用工具，再基于工具结果回答用户。"
        )
        messages.append({"role": "system", "content": system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schemas()
        print(f"🛠️ 已构建 {len(tool_schemas)} 个工具 schema")
        for schema in tool_schemas:
            print(f"   - {schema['function']['name']}: {schema['function']['description'][:60]}")

        final_response = ""
        current_iteration = 0

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1
            print(f"\n--- Function Calling 第 {current_iteration} 轮 ---")

            response = self._invoke_with_tools(
                messages=messages,
                tools=tool_schemas,
                tool_choice=self.tool_choice,
                **kwargs,
            )

            response_message = response.choices[0].message
            tool_calls = getattr(response_message, "tool_calls", None) or []

            if not tool_calls:
                final_response = self._extract_message_content(response_message) or (
                    "抱歉，我无法回答这个问题。"
                )
                print(f"💬 最终文本回复（无工具调用）")
                break

            print(f"🔧 模型请求调用 {len(tool_calls)} 个工具")

            # 把带 tool_calls 的 assistant 消息写回对话（原生协议要求）
            messages.append(
                {
                    "role": "assistant",
                    "content": self._extract_message_content(response_message) or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                print(f"   → {tool_name}({arguments})")

                result = self._execute_tool_call(tool_name, arguments)
                print(f"   ← {result[:200]}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        if current_iteration >= self.max_tool_iterations and not final_response:
            # 强制收尾：不再传 tools，让模型基于已有 tool 结果作答
            llm_response = self.llm.invoke(messages, **kwargs)
            final_response = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )
            print("⚠️ 达到最大工具轮次，使用收尾回答")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 完成")
        return final_response
