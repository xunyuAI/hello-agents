# plugin_system.py
"""
习题 6.3：插件系统

原则：第三方只依赖稳定 PluginAPI（HelloAgentsApp），不 patch 框架核心源码。

架构：
┌─────────────────────────────────────────┐
│              HelloAgents Core           │
│  Agent ABC · ToolRegistry · LLM · Config│
└───────────────┬─────────────────────────┘
                │ PluginAPI / hooks
        ┌───────┴────────┐
        │  PluginManager │  discover → load → register → unload
        └───────┬────────┘
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 AgentPlugin  ToolPlugin  ProviderPlugin
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from hello_agents import ToolRegistry
from hello_agents.core.agent import Agent


class HelloAgentsApp:
    """
    插件可调用的稳定扩展面（PluginAPI）。

    - register_agent_type：注册新的 Agent 子类（必须继承 Agent）
    - register_tool / register_function：挂到统一 ToolRegistry
    - register_provider：注册 LLM 工厂
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.agent_types: Dict[str, Type[Agent]] = {}
        self.tool_registry = tool_registry or ToolRegistry()
        self.providers: Dict[str, Callable[..., Any]] = {}
        self.metadata: Dict[str, Any] = {}

    def register_agent_type(self, name: str, cls: Type[Agent]) -> None:
        if not inspect.isclass(cls) or not issubclass(cls, Agent):
            raise TypeError(f"{cls} 必须是 Agent 的子类，避免重复造轮子")
        self.agent_types[name] = cls
        print(f"🔌 已注册 Agent 类型: {name} -> {cls.__name__}")

    def register_tool(self, tool) -> None:
        self.tool_registry.register_tool(tool)
        print(f"🔌 已注册工具: {getattr(tool, 'name', tool)}")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
    ) -> None:
        self.tool_registry.register_function(name=name, description=description, func=func)
        print(f"🔌 已注册函数工具: {name}")

    def register_provider(self, name: str, factory: Callable[..., Any]) -> None:
        self.providers[name] = factory
        print(f"🔌 已注册 Provider: {name}")

    def create_agent(self, type_name: str, **kwargs) -> Agent:
        if type_name not in self.agent_types:
            raise KeyError(f"未知 Agent 类型: {type_name}，已注册: {list(self.agent_types)}")
        cls = self.agent_types[type_name]
        # 默认注入 app 级 tool_registry（可被 kwargs 覆盖）
        kwargs.setdefault("tool_registry", self.tool_registry)
        return cls(**kwargs)

    def create_llm(self, provider: str, **kwargs):
        if provider not in self.providers:
            raise KeyError(f"未知 Provider: {provider}，已注册: {list(self.providers)}")
        return self.providers[provider](**kwargs)


class Plugin(ABC):
    """插件基类：setup / teardown 生命周期。"""

    name: str = "unnamed-plugin"
    version: str = "0.1.0"

    @abstractmethod
    def setup(self, app: HelloAgentsApp) -> None:
        """向 app 注册 Agent / Tool / Provider。"""

    def teardown(self) -> None:
        """卸载时清理资源（默认无操作）。"""


class PluginManager:
    """discover → load → register → unload。"""

    def __init__(self, app: Optional[HelloAgentsApp] = None):
        self.app = app or HelloAgentsApp()
        self.plugins: Dict[str, Plugin] = {}

    def load_plugin(self, plugin: Plugin) -> None:
        if plugin.name in self.plugins:
            raise ValueError(f"插件已加载: {plugin.name}")
        plugin.setup(self.app)
        self.plugins[plugin.name] = plugin
        print(f"✅ 插件已加载: {plugin.name}@{plugin.version}")

    def unload_plugin(self, name: str) -> None:
        plugin = self.plugins.pop(name, None)
        if plugin is None:
            raise KeyError(f"插件未加载: {name}")
        plugin.teardown()
        print(f"🧹 插件已卸载: {name}")

    def load_from_path(self, dotted: str) -> Plugin:
        """
        显式加载：'package.module:ClassName' 或 'package.module'（取首个 Plugin 子类）。
        """
        if ":" in dotted:
            module_path, class_name = dotted.split(":", 1)
        else:
            module_path, class_name = dotted, None

        module = importlib.import_module(module_path)
        if class_name:
            obj = getattr(module, class_name)
            plugin = obj() if inspect.isclass(obj) else obj
        else:
            plugin = self._find_plugin_instance(module)

        if not isinstance(plugin, Plugin):
            raise TypeError(f"{dotted} 不是 Plugin 实例/子类")
        self.load_plugin(plugin)
        return plugin

    def discover(self, plugins_dir: Union[str, Path]) -> List[Plugin]:
        """约定目录发现：扫描 *.py，加载其中的 Plugin 子类。"""
        root = Path(plugins_dir)
        loaded: List[Plugin] = []
        if not root.exists():
            return loaded

        import sys

        sys.path.insert(0, str(root.parent.resolve()))
        package_hint = root.name

        for py_file in sorted(root.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"{package_hint}.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"⚠️ 跳过 {py_file.name}: {e}")
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj is not Plugin and obj.__module__ == module.__name__:
                    plugin = obj()
                    if plugin.name not in self.plugins:
                        self.load_plugin(plugin)
                        loaded.append(plugin)
        return loaded

    @staticmethod
    def _find_plugin_instance(module) -> Plugin:
        for _, obj in inspect.getmembers(module):
            if isinstance(obj, Plugin):
                return obj
            if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin:
                return obj()
        raise ValueError(f"模块 {module.__name__} 中未找到 Plugin")
