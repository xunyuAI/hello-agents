# test_dialog_agent.py
"""
习题 6.2：多轮对话管理测试（真实 LLM）

验证：
- DialogAgent 继承 MySimpleAgent，run 后自动 Checkpoint
- rollback / branch 重置 Agent._history（历史内容来自真实模型回复）
- ConversationStore 磁盘读写
- 与 Message.metadata['message_id'] 集成
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, Message

from conversation import ConversationStore, DialogTree
from my_dialog_agent import DialogAgent

_ENV_DIR = Path(__file__).resolve().parent
load_dotenv(_ENV_DIR / ".env")

llm = HelloAgentsLLM()


def _make_agent(name: str, store=None) -> DialogAgent:
    return DialogAgent(
        name=name,
        llm=llm,
        system_prompt="你是简洁助手。每轮只用一两句话回答，不要列大纲。",
        enable_tool_calling=False,
        store=store,
    )


def test_auto_checkpoint_after_run():
    agent = _make_agent("对话助手")

    q1, q2 = "用一句话介绍你自己", "再给我一个学习建议"
    print(f"\n=== 第1轮: {q1} ===")
    r1 = agent.run(q1)
    print(f"回复1: {r1}")
    cp1 = agent.current_checkpoint_id()

    print(f"\n=== 第2轮: {q2} ===")
    r2 = agent.run(q2)
    print(f"回复2: {r2}")
    cp2 = agent.current_checkpoint_id()

    assert isinstance(r1, str) and r1.strip()
    assert isinstance(r2, str) and r2.strip()
    assert cp1 is not None and cp2 is not None and cp1 != cp2
    assert len(agent.get_history()) == 4  # 2 轮 × (user+assistant)
    assert len(agent.list_checkpoints()) == 2
    assert agent.get_history()[1].content == r1
    assert agent.get_history()[3].content == r2


def test_rollback_restores_history():
    agent = _make_agent("回溯助手")

    print("\n=== 回溯场景：A → B → rollback → C ===")
    r_a = agent.run("把数字 7 记在心里，只回复：已记住7")
    print(f"A回复: {r_a}")
    cp_a = agent.current_checkpoint_id()
    hist_a = [m.content for m in agent.get_history()]

    r_b = agent.run("现在把刚才的数字加3，只回复结果数字")
    print(f"B回复: {r_b}")
    assert len(agent.get_history()) == 4

    restored = agent.rollback(cp_a)
    assert len(restored) == 2
    assert len(agent.get_history()) == 2
    assert [m.content for m in agent.get_history()] == hist_a
    print(f"已回溯到 checkpoint={cp_a}，历史轮数恢复为 1")

    r_c = agent.run("忘记加法，改成用一个成语形容坚持")
    print(f"C回复: {r_c}")
    tree = agent.conversation.dialog_tree
    children = tree.children_of(cp_a)
    assert len(children) >= 1
    assert tree.get(agent.current_checkpoint_id()).parent_id == cp_a
    assert isinstance(r_c, str) and r_c.strip()


def test_branch_from_message_id():
    agent = _make_agent("消息分支")

    print("\n=== 按 message_id 分支 ===")
    r1 = agent.run("用一个词形容大海")
    print(f"答1: {r1}")
    mid = agent.get_history()[0].metadata["message_id"]

    r2 = agent.run("再用一个词形容高山")
    print(f"答2: {r2}")
    assert len(agent.get_history()) == 4

    agent.branch_from_message(mid)
    # 最短含该 message_id 的 checkpoint = 第一轮结束后的快照
    assert len(agent.get_history()) == 2
    assert agent.get_history()[1].content == r1
    print(f"已按 message_id={mid} 分支，恢复到第一轮历史")


def test_conversation_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(storage_dir=tmp)
        agent = _make_agent("持久化助手", store=store)

        print("\n=== 会话持久化 ===")
        reply = agent.run("用一句话说明什么是对话历史")
        print(f"回复: {reply}")
        cid = agent.conversation.id

        loaded = store.load(cid)
        assert loaded.id == cid
        assert loaded.dialog_tree.current_id is not None
        assert Path(tmp, f"{cid}.json").exists()

        agent2 = DialogAgent(
            name="恢复助手",
            llm=llm,
            system_prompt="简洁回答。",
            enable_tool_calling=False,
            conversation=loaded,
            store=store,
        )
        assert len(agent2.get_history()) == 2
        assert agent2.get_history()[1].content == reply
        print(f"已从磁盘恢复会话 {cid}，assistant 内容与原回复一致")


def test_dialog_tree_unit():
    """纯数据结构单测：不调用 LLM，只验证树指针与拷贝语义。"""
    tree = DialogTree()
    cp1 = tree.create_checkpoint(
        [Message("u1", "user"), Message("a1", "assistant")],
        label="t1",
    )
    cp2 = tree.create_checkpoint(
        [
            Message("u1", "user"),
            Message("a1", "assistant"),
            Message("u2", "user"),
        ],
        label="t2",
    )
    assert cp2.parent_id == cp1.id
    msgs = tree.branch_from(cp1.id)
    assert len(msgs) == 2
    cp3 = tree.create_checkpoint(msgs + [Message("u3", "user")], label="branch")
    assert cp3.parent_id == cp1.id
    assert len(tree.children_of(cp1.id)) == 2


if __name__ == "__main__":
    test_auto_checkpoint_after_run()
    test_rollback_restores_history()
    test_branch_from_message_id()
    test_conversation_store_roundtrip()
    test_dialog_tree_unit()
    print("\n✅ test_dialog_agent 全部通过（真实 LLM）")
