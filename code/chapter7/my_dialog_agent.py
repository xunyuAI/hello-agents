# my_dialog_agent.py
"""
习题 6.2：带多轮对话管理的 Agent

继承 MySimpleAgent：
- 复用 run / 工具调用 / 历史 Message 体系
- 每轮结束后自动写入 Conversation.DialogTree Checkpoint
- branch / rollback 只重置父类 _history，不重复维护第二份消息列表
"""

from __future__ import annotations

from typing import List, Optional

from hello_agents import HelloAgentsLLM, Config, Message

from conversation import Conversation, ConversationStore, Checkpoint
from my_simple_agent import MySimpleAgent


class DialogAgent(MySimpleAgent):
    """在 MySimpleAgent 上挂载 Conversation / DialogTree。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry=None,
        enable_tool_calling: bool = False,
        conversation: Optional[Conversation] = None,
        store: Optional[ConversationStore] = None,
        auto_checkpoint: bool = True,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
        )
        self.conversation = conversation or Conversation(title=f"{name}-会话")
        self.store = store
        self.auto_checkpoint = auto_checkpoint

        # 若会话已有当前分支，同步到 Agent 历史
        if self.conversation.dialog_tree.current_id:
            self._sync_history_from_messages(self.conversation.messages)

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """跑完父类逻辑后，自动打 Checkpoint。"""
        reply = super().run(
            input_text,
            max_tool_iterations=max_tool_iterations,
            **kwargs,
        )
        if self.auto_checkpoint:
            self.checkpoint(label=f"after:{input_text[:24]}")
        if self.store:
            self.store.save(self.conversation)
        return reply

    def checkpoint(self, label: Optional[str] = None) -> Checkpoint:
        """把当前 Agent._history 快照进对话树。"""
        return self.conversation.snapshot(list(self.get_history()), label=label)

    def rollback(self, checkpoint_id: str) -> List[Message]:
        """回溯到某 Checkpoint，并重置 Agent 历史。"""
        messages = self.conversation.dialog_tree.rollback_to(checkpoint_id)
        self._sync_history_from_messages(messages)
        if self.store:
            self.store.save(self.conversation)
        return messages

    def branch(self, checkpoint_id: str) -> List[Message]:
        """从某 Checkpoint 开新分支（语义上等同于先回溯，再继续对话）。"""
        messages = self.conversation.dialog_tree.branch_from(checkpoint_id)
        self._sync_history_from_messages(messages)
        if self.store:
            self.store.save(self.conversation)
        return messages

    def branch_from_message(self, message_id: str) -> List[Message]:
        """按 Message.metadata['message_id'] 找到节点再分支。"""
        cp = self.conversation.dialog_tree.find_by_message_id(message_id)
        if cp is None:
            raise KeyError(f"未找到包含 message_id={message_id} 的 Checkpoint")
        return self.branch(cp.id)

    def current_checkpoint_id(self) -> Optional[str]:
        return self.conversation.dialog_tree.current_id

    def list_checkpoints(self) -> List[Checkpoint]:
        return list(self.conversation.dialog_tree.checkpoints.values())

    def _sync_history_from_messages(self, messages: List[Message]) -> None:
        """
        用 Conversation 快照覆盖 Agent._history。
        走基类 setter，保证 HistoryManager / token 计数一致。
        """
        self._history = list(messages)
