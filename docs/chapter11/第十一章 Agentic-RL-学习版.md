# 第十一章 Agentic-RL

前十章我们搭好了智能体的「身手」：范式、工具、记忆、上下文、通信协议。但当你把 Agent 丢进更难的任务时，很快会发现——**它好像只会照着示例演，不会自己变强**。

这一章要补上的，是智能体走向「可训练、可自我改进」的关键拼图：

- **Agentic RL**：把大模型当成可学习的策略，放进多步交互循环里，用奖励信号优化长期任务表现。
- **SFT（监督微调）**：先学会「怎么写答案」——格式、分步推理、可读的 Final Answer。
- **GRPO（群组相对策略优化）**：再学会「怎么想得更好」——同一题试多条路径，奖励更好的那条。

你可以把这一章理解成：**给 Agent 装上「先模仿、再试错」的训练引擎**。

---

## 读这一章之前，先建立心智模型

### 一个生活比喻

| 人类学习 | Agent / LLM 对应 | 本章组件 |
|---------|------------------|---------|
| 老师带着做例题 | 监督学习 / SFT | `SFTTrainerWrapper` |
| 做完题对答案，错了改 | 强化学习奖励 | `AccuracyReward` 等 |
| 同一题试三种解法，选更好的 | 组内相对比较 | GRPO 的 `num_generations` |
| 笔记本上记公式模板 | 输出格式 | Step / Final Answer |
| 高考前模拟考 | 评估 | `action="evaluate"` |

LLM 预训练像「读完整个图书馆」——会续写，但未必会考试。SFT 像「刷例题学答题格式」。RL 像「模考后按分数调整策略」，甚至可能发现题解本里没有的捷径。

### 几个关键词，先抓牢

- **Agentic RL**：多步、可交互、优化累积回报；不是只把对话回答润色得更讨喜。
- **PBRFT**：Preference-Based Reinforcement Fine-Tuning，偏单轮偏好优化（经典 RLHF 思路）。
- **SFT**：Supervised Fine-Tuning，用 (prompt, completion) 教模型「标准写法」。
- **GRPO**：Group Relative Policy Optimization，用同组答案的相对好坏更新策略，不必再训一个 Value Model。
- **LoRA**：低秩适配，只训很少参数，省显存。
- **奖励函数**：告诉训练「什么叫好」。坏奖励会让模型「钻空子」（reward hacking）。
- **RLTrainingTool**：HelloAgents 统一入口，四种 action：`train` / `load_dataset` / `create_reward` / `evaluate`。

### 建议怎么读

1. **先建立「为何需要 RL」**（11.1）：弄清 PBRFT vs Agentic RL，再往下看公式。
2. **数据与奖励是地基**（11.2）：搞懂 SFT/RL 两种格式差在哪；三种奖励为何「答错就不给长度/步骤分」。
3. **必须先 SFT 再 GRPO**（11.3–11.4）：这是本章最硬的工程铁律。
4. **评估不是附属品**（11.5）：只看平均准确率会被骗；要分层、看格式、看错误类型。
5. **工程闭环**（11.6）：Pipeline、超参、分布式、部署——跑通比追求 SOTA 更重要。
6. **习题当复盘**：章末 5 题偏开放，按主题对齐到 11.2–11.4（原文个别节号与现结构不完全一致）。

### 本章在 HelloAgents 里的位置

```
第十章：智能体通信协议（与外部世界握手）
    ↓
第十一章：Agentic RL（用奖励训练智能体能力）  ← 你在这里
    ↓
后续：更复杂的工具学习、多智能体、长程任务……
```

技术选型：**TRL**（Hugging Face 强化学习库）+ **Qwen3-0.6B**（小模型、可在普通 GPU 上练流程）。

安装：

```bash
pip install "hello-agents[rl]==0.2.5"
# 或源码：pip install -e ".[rl]"
```

---

## 11.1 从 LLM 训练到 Agentic RL

### 11.1.1 从强化学习到 Agentic RL

强化学习（RL）解决的是**序贯决策**：智能体与环境交互，在试错中最大化长期收益。把框架套到「数学解题 Agent」上：

- **智能体**：LLM 推理系统  
- **环境**：题目 + 答案校验器  
- **状态**：题干 + 已写下的推理  
- **行动**：下一步推理或最终答案  
- **奖励**：对了 +1，错了 0  

**监督学习的三个局限**：数据质量封顶、缺乏探索、难精确优化多步中间过程。RL 允许模型自己生成多条候选路径，按正确性强化更好的路径——这就是 Agentic RL 的核心思想。

### 11.1.2 LLM 训练全景图

一个强大的对话模型通常经历：

1. **预训练**：海量文本上做下一词预测（因果语言建模）。学到语法、知识、基础推理，但不会「听话」。  
2. **后训练**：  
   - **SFT**：学指令与对话格式  
   - **奖励建模（RM）**：学 chosen > rejected  
   - **RL 微调**：用 PPO 等最大化奖励，并用 KL 项防止漂得太远  

**RLHF** 靠人标偏好，贵；**RLAIF** 用强 AI 打分，成本低，效果可接近甚至超过 RLHF。

公式不必死记，抓住含义即可：

- 预训练 / SFT：最大化「正确 token」的概率  
- RM：让更好的回答分数更高  
- PPO：高奖励 + 别离参考模型太远  

### 11.1.3 Agentic RL 的核心理念

| 维度 | PBRFT（单轮偏好） | Agentic RL |
|------|-------------------|------------|
| 时间跨度 | T=1 | T ≫ 1 |
| 状态 | 仅 prompt | prompt + 历史观察 |
| 行动 | 生成整段文本 | 文本 + 工具 + 环境操作 |
| 奖励 | 单步质量分 | 稀疏/密集/混合的累积回报 |
| 目标思维 | 更好的单个回答 | 完成复杂任务 |

**六大能力**：推理、工具使用、记忆、规划、自我改进、感知。本章数学案例主要练「推理」；工具学习在习题中延伸。

#### 再看一个「多步奖励」故事（加深直觉）

用户说：「帮我分析这个 GitHub 仓库的代码质量。」

在 **PBRFT** 世界里：模型一次性吐出一篇分析文章，然后用「写得好不好」打一个分——中间没有真正读仓库的过程。

在 **Agentic RL** 世界里，轨迹可能是：

1. 调用 GitHub API 拉文件树 → 成功，+0.1  
2. 读取关键源文件 → 成功，+0.1  
3. 做静态/风格分析 → 合理，+0.2  
4. 生成最终报告 → 质量高，+0.6  

总回报 ≈ 1.0。注意：第 1～3 步看起来像「绕路」，但对最终目标是必要的。这正是「优化长期任务」而不是「优化单段文采」。

#### MDP 五元组（不用怕数学符号）

- \(S\) 状态：现在知道什么（题干、工具返回、历史思考）  
- \(A\) 行动：下一步说什么 / 调什么工具  
- \(P\) 转移：行动后环境怎么变（例如工具返回新观察）  
- \(R\) 奖励：这一步或最终给多少分  
- \(\gamma\) 折扣：越远的奖励打多少折（0～1）  

Agentic 目标可以写成：

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t=0}^{T}\gamma^t r(s_t,a_t)\Big]
\]

读作：在策略 \(\pi_\theta\) 采出的整条轨迹上，把逐步奖励加起来（可打折），求期望并最大化。



### 11.1.4 HelloAgents 的四层架构

自下而上：

1. **数据集层**：`GSM8KDataset`、`create_sft_dataset()`、`create_rl_dataset()`  
2. **奖励函数层**：`MathRewardFunction`、`AccuracyReward`、`LengthPenaltyReward`、`StepReward`  
3. **训练器层**：`SFTTrainerWrapper`、`GRPOTrainerWrapper`（含 LoRA）  
4. **统一接口层**：`RLTrainingTool`  

### 11.1.5 快速上手

```python
from hello_agents.tools import RLTrainingTool
import json

rl_tool = RLTrainingTool()

# 小样本 SFT 冒烟测试
sft_result = json.loads(rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_test_sft",
    "max_samples": 10,
    "num_epochs": 1,
    "batch_size": 2,
    "use_lora": True,
}))
```

**新手提示**：10 条样本、1 个 epoch 的准确率低是正常的。GRPO 时注意 `batch_size` 能被 `num_generations`（默认常为 8）整除。复制书中代码时，把全角逗号 `，` 改成半角 `,`。

---

## 11.2 数据集与奖励函数

数据集定义「学什么任务」，奖励定义「什么叫好」。

### 11.2.1 GSM8K：为什么特别适合 RL

GSM8K（Grade School Math 8K）：约 **7473** 训练题、**1319** 测试题，小学应用题，通常 2–8 步推理。

优点：

1. 有唯一（或可自动判定的）答案 → 可自动给奖励  
2. 天然多步推理  
3. 学到的推理习惯可迁移  

答案文本里常见标记：中间计算 `<<48/2=24>>`，最终答案 `#### 72`。

#### 三种格式（一定要分清）

| 格式 | 关键字段 | 用途 |
|------|----------|------|
| 原始 | question + answer（含步骤） | 人类阅读 |
| SFT | prompt + completion（完整解题） | 模仿格式与推理 |
| RL | prompt + ground_truth（**只有答案**） | 自主探索推理 |

**为什么 RL 不给过程？** 若把完整题解塞进监督目标，模型倾向于背题；只给答案，才逼它自己生成推理，再用奖励筛选好坏。

```python
# 查看格式
rl_tool.run({"action": "load_dataset", "format": "sft", "max_samples": 5})
rl_tool.run({"action": "load_dataset", "format": "rl", "max_samples": 5})
```

### 11.2.2 奖励函数设计

目标仍是最大化累积回报。数学题可简化为：根据预测答案与标准答案打分。

#### （1）AccuracyReward

答对 1，答错 0。需要稳健的**答案抽取**（Final Answer / #### / 末位数字）和数值容差。简单，但稀疏——差一点也是 0。

#### （2）LengthPenaltyReward

$$
r = r_{acc} - \alpha \cdot \max(0, l - l_{target})
$$

默认 α≈0.001。**只在答对时惩罚过长**，否则模型会用「又短又错」刷分。

#### （3）StepReward

$$
r = r_{acc} + \beta \cdot s
$$

默认 β≈0.1。检测 Step 标记等。**只在答对时加步骤分**，防假步骤刷分。

#### 组合

常见：准确率+长度；准确率+步骤；三者平衡。GRPO 完整训练可用：

```python
"reward_type": "combined",
"reward_config": {
  "components": [
    {"type": "accuracy", "weight": 1.0},
    {"type": "length_penalty", "weight": 0.5, "target_length": 200},
    {"type": "step", "weight": 0.3, "step_bonus": 0.1},
  ]
}
```

`create_reward` 往往先配单一类型；组合多在训练配置里完成。

### 11.2.3 自定义数据集与奖励

1. 准备含 `question` / `answer` 的原始表  
2. `format_math_dataset(..., format_type="sft"|"rl")`  
3. `register_dataset` + `register_reward_function`（建议同名）  
4. 再 `train`

自定义奖励签名大致为：`def fn(completions: List[str], **kwargs) -> List[float]`，`kwargs` 里常有 `ground_truth`。

**易错点**：答案抽不出来 → 奖励全 0 → 看起来「训了等于没训」。先用几条样本打印抽取结果。

#### 最小自定义数据集示例

```python
from datasets import Dataset
from hello_agents.rl import format_math_dataset
from hello_agents.tools import RLTrainingTool

raw = Dataset.from_list([
    {
        "question": "A baker makes 12 cakes and sells 5. How many left?",
        "answer": "12 - 5 = <<12-5=7>>7\n#### 7",
    },
    # ... 更多样本
])

sft_ds = format_math_dataset(raw, format_type="sft", model_name="Qwen/Qwen3-0.6B")
rl_ds = format_math_dataset(raw, format_type="rl", model_name="Qwen/Qwen3-0.6B")

rl = RLTrainingTool()
rl.register_dataset("bakery_math", rl_ds)

def bakery_reward(completions, **kwargs):
    gts = kwargs.get("ground_truth", [])
    rewards = []
    for c, gt in zip(completions, gts):
        # 极简：字符串包含正确答案则 1，否则 0（正式应做稳健抽取）
        rewards.append(1.0 if str(gt) in c else 0.0)
    return rewards

rl.register_reward_function("bakery_math", bakery_reward)
```

**字段再强调一次**：

- SFT：`prompt` + `completion`（可选 `text`）  
- RL：`question` + `prompt` + `ground_truth` + `full_answer`  

注册名与奖励名保持一致，后期调用不容易乱。



---

## 11.3 SFT 训练

### 11.3.1 为什么需要 SFT

预训练模型输出自由散漫，经常抽不出标准答案 → RL 奖励失效。SFT 负责：

1. 学会对话模板与 **Step / Final Answer** 结构  
2. 模仿基础推理模式  
3. 提供可评估的初始策略  
4. **缩小**后续 RL 的探索空间  

**铁律：先 SFT，再 GRPO。**

### 11.3.2 LoRA：参数高效微调

冻结大矩阵 \(W\)，只训练低秩 \(B,A\)（秩 \(r\) 很小）：\(\Delta W = BA\)。显存和可训练参数量大幅下降，效果通常略逊全量微调，但对教学与单卡场景极友好。

常用：`r=8~16`，`lora_alpha ≈ 2r`，目标模块 `q_proj/k_proj/v_proj/o_proj`。

#### 新手怎么选 r / alpha？

1. **先小后大**：r=8 冒烟 → 正式可试 16。  
2. **alpha ≈ 2r**：r=8→alpha=16；r=16→alpha=32。  
3. **先只挂注意力投影**（q/k/v/o），不够再考虑 MLP。  
4. **显存仍爆**：降 batch，而不是先把 r 砍到 1（信号太弱）。  

#### 「SFT 前后对比」你应该看到什么

预训练输出常见问题：冗长、没 Final Answer、推理东拉西扯、无法自动抽答案。  
SFT 后期望：出现 `Step 1` / `Step 2` / `Final Answer: …`，答案可被脚本提取。  

如果 SFT 后仍然抽不出答案，**不要进入 GRPO**，先查：

- 对话模板是否用了模型对应的 special tokens（如 Qwen 的 `<|im_start|>`）  
- completion 是否真的教了格式  
- `max_new_tokens` 是否截断  



### 11.3.3–11.3.4 实战与评估

```python
rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/sft_full",
    "max_samples": None,      # 全量 7473
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
})
```

书中量级参考（Qwen3-0.6B / GSM8K）：SFT 后约 **40–50%**；再经 RL 约 **60–70%**（以你环境实测为准）。

故障速查：

- **OOM**：降 batch、开 LoRA、梯度累积、混合精度  
- **过拟合**：增 weight_decay、减 epoch、加数据  
- **loss 不降**：查模板格式、提 lr、加 epoch  

---

## 11.4 GRPO 训练

SFT 会「模仿」；GRPO 让模型在奖励下「探索更好推理」。

### 11.4.1 从 PPO 到 GRPO

PPO 经典但重：需要 Value Model，常要维护多个模型，LLM 场景下不稳、吃显存。

**GRPO** 核心：

- 每题生成一组答案  
- 用 \(r_i - \bar{r}_{group}\) 当「相对优势」  
- **不需要 Value Model**  
- 用 KL 项把策略拴在参考模型（通常是 SFT）附近  

相对奖励小例子：`[1,1,0,0.8]`，均值 `0.7` → 相对 `[+0.3,+0.3,-0.7,+0.1]`。

#### 用一道题把 GRPO「走」一遍

问题：`What is 48 + 24?`（示意）  
`num_generations=4`，模型吐出 4 个回答，奖励分别为：

| 样本 | 内容摘要 | r |
|------|----------|---|
| A | 正确，步骤清晰 | 1.0 |
| B | 正确，但啰嗦 | 0.9 |
| C | 算成 70 | 0.0 |
| D | 正确，步骤一般 | 0.8 |

均值 \(\bar r=0.675\)。相对奖励约：

- A: +0.325 → 强化  
- B: +0.225 → 强化（略弱）  
- C: −0.675 → 抑制  
- D: +0.125 → 轻度强化  

再加上 KL：别离 SFT 参考策略太远，避免「为了刷分把语言能力搞崩」。

#### 监控仪表盘上盯三件事

1. **Average reward**：整体是否在爬升（允许震荡）  
2. **KL**：理想大致落在 0.01～0.1 量级（书中经验区间）；突然 >0.5 要警惕  
3. **人工抽查**：随机看 10 条，格式是否仍可读  

可用 TensorBoard 或 Weights & Biases（`use_wandb` / `WANDB_PROJECT`）。



### 11.4.2 实战参数

```python
rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "./models/sft_full",  # 强烈建议从 SFT 起步
    "output_dir": "./models/grpo_full",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5,       # 通常小于 SFT
    "num_generations": 4,        # 典型 4–8
    "max_new_tokens": 512,
    "temperature": 0.8,
    "kl_coef": 0.05,             # 建议 0.05–0.1
    "clip_range": 0.2,
    "use_lora": True,
    "reward_type": "combined",
})
```

### 11.4.3 训练循环在干什么

1. **采样**：每题生成 G 个答案  
2. **打分**：算每个 \(r_i\)  
3. **相对化**：减组内均值  
4. **更新**：强化相对更好的样本，KL 防漂  
5. **监控**：reward↑、KL 在合理区间、准确率↑、抽查格式  

异常：奖励不升 → 查 lr / kl / 奖励 / SFT 基线；KL 爆炸 → 降 lr、增 kl_coef；准确率升但格式崩 → combined 奖励或加大 kl。

---

## 11.5 模型评估与分析

### 11.5.1 指标体系

三类：**准确性**（Accuracy、Accuracy@K、数值误差）、**效率**（长度、步数、时延）、**质量**（格式正确率、连贯性等）。

只看平均 Accuracy 不够：简单题拉高分数时，难题可能一塌糊涂。

### 11.5.2 评估实战

```python
rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/grpo_full",
    "max_samples": 200,
    "use_lora": True,
    "metrics": ["accuracy", "accuracy_at_k", "average_length",
                "average_steps", "format_correctness"],
    "k": 3,
    "return_details": True,
})
```

建议对比：**Base / SFT / GRPO** 三档。

#### Accuracy@K 为什么有用？

单次 greedy 解码可能「运气不好」。同一题采样 K 次，只要有一次对，就算命中——这反映模型「潜力」和探索多样性。若 Accuracy 低但 Accuracy@K 高，说明「会，但不稳」；两者都低，才是「还不会」。

#### 错误分析模板（可直接复制）

对 `return_details=True` 的失败样本，贴标签：

```text
[ ] 计算错误：推理结构对，算术错
[ ] 推理错误：步骤逻辑断或跳步
[ ] 理解错误：题意读偏
[ ] 格式错误：有想法但抽不出答案
难度：1-2步 / 3-4步 / 5+步
```

然后统计占比，决定下一步是补 SFT、改奖励，还是加工具。



### 11.5.3–11.5.4 错误分析与改进

错误常分：计算 / 推理 / 理解 / 格式。按步数分层（1–2 / 3–4 / 5+）看准确率。

闭环：**评估 → 分析 → 定改进（补 SFT / 改奖励 / 调超参）→ 再训**。

- 格式差 → 先补 SFT，别急着堆 RL  
- 计算错多 → 考虑工具或更细的过程奖励  
- 只会简单题 → 别宣称「已学会推理」  

---

## 11.6 完整训练流程实战

### 11.6.1 端到端六阶段

可用 `AgenticRLPipeline`（config.json 驱动）串联：

1. 数据准备（`load_dataset`，质量检查）  
2. SFT 训练  
3. SFT 评估（可设准确率门槛，如 0.40）  
4. GRPO 训练（输入 SFT 路径）  
5. GRPO 评估  
6. 保存结果 / 进入部署  

**建议**：先用 100–1000 条样本把流水线跑通，再上全量。

### 11.6.2 超参数调优

方法：网格搜索 / 随机搜索 / 贝叶斯（Optuna）。优先拧：`learning_rate`、`lora_rank`、`kl_coef`。

### 11.6.3 分布式训练

- 2–8 卡：常先 DDP  
- 更大模型：DeepSpeed ZeRO-2/3  
- 工具：`accelerate config` / `accelerate launch`  
- 记住：**总 batch** = 单卡 batch × GPU 数 × 梯度累积；换总 batch 时学习率常按 \(\sqrt{B_{new}/B_{base}}\) 粗调  

### 11.6.4 生产部署

1. LoRA `merge_and_unload` 合并进基座  
2. 可选 8-bit 量化推理  
3. FastAPI 暴露 `POST /generate`，`uvicorn` 启动  

未 merge 就上线，路径与依赖容易乱。

---

## 11.8 本章小结

1. **Agentic RL ≠ 单轮 RLHF**：多步状态、工具行动、累积回报。  
2. **六大能力**是目标清单；本章用数学推理把「训练闭环」跑通。  
3. **预训练（现成）→ SFT 打底 → GRPO 拔高 → 评估改进**：SFT 与 RL 缺一不可。  

进阶路径：RL/LLM 基础 → 跑通 HelloAgents 示例 → 读 TRL、自定义数据与奖励 → 工具学习 / 多智能体 / 论文。

---

## 习题导读（对应原书习题）

> 部分无标准答案，重在综合理解与动手。

1. **MDP / 任务映射**：解释为何 Agentic 状态含历史观察；把「代码调试助手」映射到 S/A/R/P；设计一个监督难优化中间步、但 RL 可用延迟奖励的任务。  
2. **SFT vs GRPO**：LoRA 原理与选型；GRPO 相对 PPO 的简化；扩展多轮数据、增强与可视化。  
3. **奖励设计**：部分正确 / 过程分 / 冗长惩罚；为代码助手、客服、游戏 AI 设计奖励；举例 reward hacking 与防御。  
4. **GSM8K 深化**：数据集特点；扩展到更难数学；泛化评估；在线学习的质量 / 遗忘 / 安全。  
5. **工具学习**：何时选工具的训练与奖励；分层 RL；多工具课程学习。  

---

## 自测清单

- [ ] 能用白话区分 PBRFT 与 Agentic RL  
- [ ] 能说出后训练三步与四层架构  
- [ ] 能列出 `RLTrainingTool` 四个 action  
- [ ] 能对比 SFT / RL 数据字段  
- [ ] 能写出三种奖励并解释「仅正确时塑形」  
- [ ] 能解释为何必须先 SFT  
- [ ] 能口述 GRPO 相对奖励与 KL 的作用  
- [ ] 能列出评估指标与 Pipeline 六阶段  
- [ ] 知道 merge → 量化 → API 部署路径  

---



---

## 附录 A · 动手实验：从冒烟到完整链路（推荐照做）

下面按「最小可运行 → 再放大」组织。没有 GPU 也能先读懂；有 GPU 再逐步放开 `max_samples`。

### A.1 环境检查清单

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
pip show hello-agents | head -5
```

若安装失败，优先确认：

1. 使用项目/虚拟环境的 pip，而不是系统 Python（macOS Homebrew 有 PEP 668 限制）。  
2. 额外依赖装的是 `hello-agents[rl]`，不是只装了基础包。  
3. Hugging Face 能拉到 `Qwen/Qwen3-0.6B`（可先手动 `huggingface-cli download`）。

### A.2 三分钟看懂数据长什么样

```python
from hello_agents.tools import RLTrainingTool
import json

rl = RLTrainingTool()

for fmt in ("sft", "rl"):
    info = json.loads(rl.run({
        "action": "load_dataset",
        "format": fmt,
        "max_samples": 2,
    }))
    print("=" * 40, fmt)
    print("size:", info.get("dataset_size"))
    print("keys:", info.get("sample_keys"))
    # 若返回里带有样本预览字段，打印出来对照 prompt/completion/ground_truth
```

**你要亲眼确认的两件事**：

1. SFT 样本的 `completion` 里有清晰步骤和最终答案。  
2. RL 样本的 `ground_truth` **只有答案数字（或短字符串）**，没有整段题解。

### A.3 先训 10 条 SFT（冒烟）

```python
from hello_agents.tools import RLTrainingTool
import json

rl = RLTrainingTool()
out = json.loads(rl.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/ch11_smoke_sft",
    "max_samples": 10,
    "num_epochs": 1,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 16,
}))
print(out)
```

**成功标准（冒烟级）**：目录里出现适配器权重/配置；日志里 loss 有数值且没有立刻崩溃。准确率可以很低——这阶段不考核分数。

### A.4 再训 5～20 条 GRPO（冒烟）

```python
out = json.loads(rl.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "./models/ch11_smoke_sft",
    "output_dir": "./models/ch11_smoke_grpo",
    "max_samples": 8,
    "num_epochs": 1,
    "batch_size": 4,          # 必须能被 num_generations 整除
    "num_generations": 4,
    "learning_rate": 1e-5,
    "kl_coef": 0.05,
    "use_lora": True,
    "reward_type": "accuracy",
}))
print(out)
```

若报错与整除有关：把 `batch_size` 改成 `num_generations` 的倍数（或反过来减小 generations）。

### A.5 评估时看什么

```python
ev = json.loads(rl.run({
    "action": "evaluate",
    "model_path": "./models/ch11_smoke_grpo",
    "max_samples": 20,
    "use_lora": True,
    "return_details": True,
}))
print("accuracy:", ev.get("accuracy"))
print("average_reward:", ev.get("average_reward"))
```

新手常见误读：冒烟集准确率 0～20% 就惊慌。先确认 **能否抽出答案、格式是否开始出现 Step/Final Answer**，再谈分数。

---

## 附录 B · 奖励黑客（Reward Hacking）小案例

奖励函数写得不好时，模型会「钻空子」拿高分但不完成真正目标。

| 糟糕设计 | 模型可能的作弊 | 防御 |
|----------|----------------|------|
| 只要输出里出现数字就给分 | 乱写数字 | 与 ground_truth 严格比对 |
| 步骤越多分越高，且答错也加分 | 灌水假步骤 | **仅答对时**给步骤分 |
| 越短越好，答错也减罚 | 输出极短错误答案 | **仅答对时**施加长度惩罚 |
| 只看关键词「因此答案是」 | 套话但算错 | 抽取后再数值/字符串匹配 |
| 多目标权重失衡 | 某一项刷爆 | 调权重 + 人工抽查 |

记住一句话：**先保证「对错」主信号，再用长度/步骤做塑形。**

---

## 附录 C · 超参速查卡（可贴在显示器边）

### SFT

| 参数 | 起步建议 | 备注 |
|------|----------|------|
| learning_rate | 5e-5（LoRA 可试 1e-4） | 过大发散，过小不学 |
| num_epochs | 1（冒烟）/ 3（正式） | 注意过拟合 |
| batch_size | 2～8 | OOM 就降 |
| lora_rank | 8～16 | 更大更强也更吃资源 |
| lora_alpha | 约 2×rank | 常见搭配 |
| warmup_ratio | 0.1 | 稳一点 |
| weight_decay | 0.01 | 抑过拟合 |

### GRPO

| 参数 | 起步建议 | 备注 |
|------|----------|------|
| learning_rate | 1e-5～5e-5 | 通常 **小于** SFT |
| num_generations | 4～8 | 越大信号越好、越贵 |
| temperature | 0.7～1.0 | 太低探索不足 |
| kl_coef | 0.05～0.1 | KL 爆炸就加大 |
| clip_range | 0.2 | 类似 PPO ε |
| max_new_tokens | 256～512 | 太短截断，太长浪费 |
| batch_size | generations 的倍数 | 硬约束 |

### 期望量级（书中参考，非保证）

- Qwen3-0.6B + GSM8K：SFT 后约 40–50%；GRPO 后再到约 60–70%。  
- 你的机器、样本数、随机种子不同，数字会漂——用相对提升判断「流程是否有效」。

---

## 附录 D · 端到端 Pipeline 配置骨架

```json
{
  "model": {"base_model": "Qwen/Qwen3-0.6B"},
  "data": {"max_samples": 1000},
  "sft": {
    "output_dir": "./models/sft_model",
    "num_epochs": 3,
    "batch_size": 8
  },
  "grpo": {
    "output_dir": "./models/grpo_model",
    "num_epochs": 3,
    "batch_size": 4
  },
  "eval": {
    "max_samples": 200,
    "sft_accuracy_threshold": 0.40
  },
  "monitoring": {
    "use_wandb": false,
    "use_tensorboard": true,
    "wandb_project": "agentic-rl-pipeline"
  }
}
```

六阶段对应关系：

1. `stage1_prepare_data` → `load_dataset`  
2. `stage2_sft_training` → `algorithm=sft`  
3. `stage3_sft_evaluation` → 不达标则停，回头修  
4. `stage4_grpo_training` → 输入 SFT 路径  
5. `stage5_grpo_evaluation` → 对比与错误分析  
6. `stage6_save_results` → `training_results.json` / 部署  

**门槛思维**：SFT 准确率远低于阈值时强行 GRPO，等于在烂地基上盖楼。

---

## 附录 E · 部署最小路径

```python
# 1) merge LoRA（示意）
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base, "./models/grpo_full")
merged = model.merge_and_unload()
merged.save_pretrained("./models/merged_model")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tok.save_pretrained("./models/merged_model")
```

```python
# 2) FastAPI 骨架（示意）
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    text: str
    max_tokens: int = 256

class Answer(BaseModel):
    text: str
    confidence: float = 0.0

@app.post("/generate", response_model=Answer)
def generate(q: Question):
    # 这里接 merged 模型的 generate
    return Answer(text="...", confidence=0.0)

# uvicorn api:app --host 0.0.0.0 --port 8000
```

可选：`load_in_8bit=True` + `device_map="auto"` 降低推理显存。

---

## 附录 F · 原书图表速查

| 编号 | 内容 |
|------|------|
| 图 11.1 | LLM 训练全景 |
| 图 11.2 | 六大核心能力 |
| 图 11.3 | HelloAgents 四层架构 |
| 图 11.4 | GSM8K 三种格式转换 |
| 图 11.5 | 奖励函数设计 |
| 图 11.6 | SFT 在流程中的桥梁作用 |
| 图 11.7 | PPO vs GRPO 流程 |
| 图 11.8 | 评估→分析→改进闭环 |
| 图 11.9 | 端到端六阶段 |
| 表 11.1–11.9 | PBRFT对比、GSM8K统计、格式、奖励、LoRA、GRPO、指标、调参、显存 |

原书配图 URL 前缀：

`https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/11-figures/`

---

## 附录 G · 常见报错对照

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| CUDA OOM | batch / generations / 序列过长 | 降三者之一，开 LoRA |
| 找不到模块 `hello_agents.rl` | 未装 `[rl]` 额外依赖 | `pip install "hello-agents[rl]==0.2.5"` |
| batch 与 generations 冲突 | 不整除 | 调整倍数关系 |
| 奖励长期为 0 | 抽不出答案 / 无 SFT 格式 | 先修 SFT 与抽取 |
| KL 突然很大 | lr 过大或奖励过激 | 降 lr、增 kl_coef |
| 准确率升但胡言乱语 | 偏离参考模型 | 加大 KL，检查 combined 奖励 |
| 分布式 hang | NCCL / IP / rank | `NCCL_DEBUG=INFO`，核对 accelerate 配置 |
| 全角逗号 SyntaxError | 从中文文档复制代码 | 全局替换 `，` → `,` |

---

## 附录 H · 一页纸复习稿

```
Agentic RL = LLM 当策略 + 多步交互 + 累积奖励

后训练：SFT（格式/基线）→（可选 RM）→ RL（GRPO）
铁律：没有 SFT，RL 奖励常是噪声

数据：SFT 要 completion；RL 只要 ground_truth
奖励：Accuracy 为主；Length/Step 仅在答对时塑形
LoRA：训 BA，冻 W；部署前可 merge

GRPO：同题多样本 → r - mean(r) → 更新 + KL
评估：Accuracy + @K + 格式 + 难度分层 + 错误类型
工程：小样本冒烟 → 门槛评估 → 全量 → merge → API
```


## 与互动教程的配合

浏览器打开：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter11/interactive-tutorial/index.html"
```

建议：**先互动页建立地图与自测 → 再回本学习版精读公式与代码 → 最后对照原书 `第十一章 Agentic-RL.md` 查图表与参考文献**。
