# conversation.py
"""
习题 6.2：多轮对话管理（分支 / 回溯）

与 Message 的集成原则：
- 叶子节点仍是 hello_agents.Message
- Checkpoint 保存「到该步为止」的 Message 快照（深拷贝）
- 分支 / 回溯 = 把 Agent._history 重置为某 Checkpoint 的拷贝
- Agent 不直接操作树结构，只消费「当前分支的 Message 列表」
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hello_agents import Message


def _copy_messages(messages: List[Message]) -> List[Message]:
    """深拷贝 Message 列表，避免分支间互相污染。"""
    copied: List[Message] = []
    for msg in messages:
        meta = deepcopy(msg.metadata) if msg.metadata else {}
        copied.append(
            Message(
                msg.content,
                msg.role,
                timestamp=msg.timestamp,
                metadata=meta,
            )
        )
    return copied


def _ensure_message_id(msg: Message) -> str:
    """在 metadata 中写入 message_id，便于按消息定位分支点。"""
    if msg.metadata is None:
        msg.metadata = {}
    mid = msg.metadata.get("message_id")
    if not mid:
        mid = uuid.uuid4().hex[:10]
        msg.metadata["message_id"] = mid
    return mid


@dataclass
class Checkpoint:
    """对话树上的一个节点：保存到该步为止的 Message 快照。"""

    id: str
    messages: List[Message]
    parent_id: Optional[str] = None
    label: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        return cls(
            id=data["id"],
            messages=messages,
            parent_id=data.get("parent_id"),
            label=data.get("label"),
            created_at=created or datetime.now(),
        )


class DialogTree:
    """
    对话树：每个 Checkpoint 是一条分支上的快照。

    回溯 = 把 current 指针移到历史节点
    分支 = 从某节点继续 append，新 Checkpoint 的 parent 指向该节点
    """

    def __init__(self):
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.current_id: Optional[str] = None
        self.root_id: Optional[str] = None

    def create_checkpoint(
        self,
        messages: List[Message],
        label: Optional[str] = None,
    ) -> Checkpoint:
        for msg in messages:
            _ensure_message_id(msg)

        cp = Checkpoint(
            id=uuid.uuid4().hex[:8],
            messages=_copy_messages(messages),
            parent_id=self.current_id,
            label=label,
        )
        self.checkpoints[cp.id] = cp
        self.current_id = cp.id
        if self.root_id is None:
            self.root_id = cp.id
        return cp

    def get(self, checkpoint_id: str) -> Checkpoint:
        if checkpoint_id not in self.checkpoints:
            raise KeyError(f"Checkpoint 不存在: {checkpoint_id}")
        return self.checkpoints[checkpoint_id]

    def rollback_to(self, checkpoint_id: str) -> List[Message]:
        """回溯到指定节点，返回该节点 Message 快照的拷贝。"""
        cp = self.get(checkpoint_id)
        self.current_id = checkpoint_id
        return _copy_messages(cp.messages)

    def branch_from(self, checkpoint_id: str) -> List[Message]:
        """
        从指定节点开新分支：先把 current 指过去，
        之后下一次 create_checkpoint 会以该节点为 parent。
        """
        return self.rollback_to(checkpoint_id)

    def find_by_message_id(self, message_id: str) -> Optional[Checkpoint]:
        """
        按 Message.metadata['message_id'] 定位分支点。

        后续 Checkpoint 会复制前缀历史，因此同一 message_id 可能出现在多个节点；
        取「消息数最少」的快照，即该消息刚进入历史时的节点。
        """
        matches: List[Checkpoint] = []
        for cp in self.checkpoints.values():
            for msg in cp.messages:
                mid = (msg.metadata or {}).get("message_id")
                if mid == message_id:
                    matches.append(cp)
                    break
        if not matches:
            return None
        return min(matches, key=lambda c: (len(c.messages), c.created_at))

    def children_of(self, checkpoint_id: str) -> List[Checkpoint]:
        return [c for c in self.checkpoints.values() if c.parent_id == checkpoint_id]

    def path_to_current(self) -> List[Checkpoint]:
        """从 root 到 current 的路径。"""
        path: List[Checkpoint] = []
        cursor = self.current_id
        while cursor:
            cp = self.checkpoints[cursor]
            path.append(cp)
            cursor = cp.parent_id
        path.reverse()
        return path


@dataclass
class Conversation:
    """一次会话：消息列表 + 对话树。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=datetime.now)
    title: str = ""
    dialog_tree: DialogTree = field(default_factory=DialogTree)

    @property
    def messages(self) -> List[Message]:
        """当前分支上的消息（来自 current checkpoint；若无则空）。"""
        if not self.dialog_tree.current_id:
            return []
        return self.dialog_tree.get(self.dialog_tree.current_id).messages

    def snapshot(self, messages: List[Message], label: Optional[str] = None) -> Checkpoint:
        return self.dialog_tree.create_checkpoint(messages, label=label)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "title": self.title,
            "current_id": self.dialog_tree.current_id,
            "root_id": self.dialog_tree.root_id,
            "checkpoints": {
                cid: cp.to_dict() for cid, cp in self.dialog_tree.checkpoints.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        conv = cls(
            id=data["id"],
            created_at=created or datetime.now(),
            title=data.get("title", ""),
        )
        tree = conv.dialog_tree
        for cid, cp_data in data.get("checkpoints", {}).items():
            tree.checkpoints[cid] = Checkpoint.from_dict(cp_data)
        tree.current_id = data.get("current_id")
        tree.root_id = data.get("root_id")
        return conv


class ConversationStore:
    """对话存储：默认内存；也可落盘 JSON。"""

    def __init__(self, storage_dir: Optional[str] = None):
        self._memory: Dict[str, Conversation] = {}
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, conversation: Conversation) -> str:
        self._memory[conversation.id] = conversation
        if self.storage_dir:
            path = self.storage_dir / f"{conversation.id}.json"
            path.write_text(
                json.dumps(conversation.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return str(path)
        return conversation.id

    def load(self, conversation_id: str) -> Conversation:
        if conversation_id in self._memory:
            return self._memory[conversation_id]
        if self.storage_dir:
            path = self.storage_dir / f"{conversation_id}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                conv = Conversation.from_dict(data)
                self._memory[conv.id] = conv
                return conv
        raise KeyError(f"Conversation 不存在: {conversation_id}")

    def list_ids(self) -> List[str]:
        ids = set(self._memory.keys())
        if self.storage_dir:
            ids.update(p.stem for p in self.storage_dir.glob("*.json"))
        return sorted(ids)
