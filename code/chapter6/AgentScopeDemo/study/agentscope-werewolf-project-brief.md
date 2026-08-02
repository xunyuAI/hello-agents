# AgentScope 三国狼人杀 · 项目任务书

> 版本：v1  
> 日期：2026-08-02  
> 服务目标：把「看懂 Demo」升级为「能改一层而不拆整盘」

---

## 背景

三国狼人杀 Demo 已经具备完整昼夜流程。学习重点不是重写游戏，而是练习：

- 在正确的层做最小改动  
- 保持 MsgHub / Pipeline / 结构化输出的边界清晰  

---

## 目标

证明你已掌握：

1. 游戏控制层与交互层的职责划分  
2. 用结构化输出约束新动作  
3. （进阶）用 MsgHub 参与者子集做信息隔离  

---

## 交付物

### 最小交付物（三选一即可）

**A. 配置向：启用 8 人局猎人**

- 调用 `setup_game(player_count=8)`  
- 确认猎人出局时能进入 `hunter_phase`  
- 交付：改动点说明 + 一段含「猎人开枪」或「放弃开枪」的日志  

**B. 结构化向：给白天投票增加「弃票」显式字段**

- 扩展 `get_vote_model_cn`（例如 `abstain: bool` 或允许特殊字面量）  
- 同步修改 `day_phase` 计票逻辑  
- 交付：模型定义 diff 说明 + 一局中出现弃票的处理结果  

**C. 提示词向：强化某一三国人物话风**

- 只改 `prompt_cn.py` 或 `game_roles.CHARACTER_TRAITS`  
- 对比改前/改后同一角色两段发言风格差异  
- 交付：改动前后发言摘录各 1 段  

### 进阶交付物（学有余力）

**D. 新神职：守护者（最小可用）**

- `game_roles` 里已有「守护者」描述；在 `main_cn` 增加 `guard_phase`  
- 夜晚在狼人击杀后、女巫前：守护者选择守护目标；若命中击杀目标则该击杀无效  
- 用新的 `GuardActionModelCN` + 必要时 MsgHub/单人 `await agent`  
- 8 或 9 人配置里启用守护者  

**E. 迁移提纲（可不写完整代码）**

设计「三人代码评审投票」：

- 谁用 MsgHub 广播评论  
- 谁用 fanout 收集 Approve/Request Changes  
- 用哪个 Pydantic 模型  

交付：一页 Markdown 设计说明。

---

## 约束

- **时间**：最小交付物 ≤ 2 小时；进阶 D ≤ 半天  
- **改动范围**：优先单文件或「模型 + 一处 phase」；禁止顺手大重构  
- **依赖**：沿用 `agentscope==1.0.2` 与 DashScope，不新增框架  
- **密钥**：只用环境变量 / `.env`，禁止把 key 写进笔记或提交  

---

## 任务拆分（以进阶 D 为例）

1. 在 `structured_output_cn.py` 增加守护模型（目标必须是存活玩家）  
2. 在 `ThreeKingdomsWerewolfGame.__init__` 增加 `self.guard` 列表与状态  
3. `setup_game` 分配守护者到 `self.guard`  
4. 实现 `guard_phase(killed_player) -> protected_name`  
5. 在 `run_game` 夜流程接入，并与女巫救人逻辑协调（先写清优先级再写代码）  
6. 跑一局验证：守护命中时死亡公告符合预期  

---

## 验收标准

### 自己如何判断完成

- [ ] 能指出改动落在控制层 / 交互层 / 角色层的哪一层  
- [ ] 游戏能从 `setup` 跑到至少完整的一夜一天  
- [ ] 无密钥泄露；异常时有合理降级（跳过技能/弃票），不直接 traceback 崩掉整局  

### 别人如何判断完成

- 阅读你的 10 行说明，能在 2 分钟内找到改动入口  
- 对照日志能看到新行为被触发至少一次  

---

## 变式挑战

- 把「投票结果」也做成结构化主持人播报（扩展 `GameModerator`）  
- 为 `analyze_speech_pattern`（`utils_cn.py`）接一条白天讨论后的旁路分析（可选，非必须）  
- 对比 AutoGenDemo：用表格写「若用 AutoGen 实现私密狼人频道，难点在哪」  

---

## 完成记录（学习时填写）

- 选择的交付物：A / B / C / D / E  
- 完成日期：  
- 遇到的坑：  
- 复测命令：`python main_cn.py`
