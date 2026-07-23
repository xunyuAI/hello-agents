import collections

# 示例语料库，与上方案例讲解中的语料库保持一致
# 合并两句：'datawhale agent learns' + 'datawhale agent works'
corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()  # -> ['datawhale', 'agent', 'learns', 'datawhale', 'agent', 'works']
total_tokens = len(tokens)  # 6

# --- 第一步：计算 P(datawhale) ---
# 句首词没有“前文”，用一元概率：词频 / 总词数
count_datawhale = tokens.count('datawhale')  # 2
p_datawhale = count_datawhale / total_tokens  # 2/6 ≈ 0.333
print(f"第一步: P(datawhale) = {count_datawhale}/{total_tokens} = {p_datawhale:.3f}")

# --- 第二步：计算 P(agent|datawhale) ---
# 【为什么要这一步？】
# Bigram (N=2) 马尔可夫假设：当前词只依赖前一个词。
# 估算句子时，第二个词 agent 的概率写成条件概率：
#   P(agent | datawhale) = Count(datawhale, agent) / Count(datawhale)
# 含义：在语料里，凡是出现过 datawhale 的地方，后面紧跟 agent 的比例是多少。
#
# 【本例手工对照】
# 词对 (datawhale, agent) 出现 2 次；(datawhale) 出现 2 次 → 2/2 = 1.0
# 即：语料中每次见到 datawhale，后面一定是 agent。

# zip(tokens, tokens[1:])：把序列与“右移一位”的序列两两配对，得到相邻词对
#   tokens      = [datawhale, agent, learns, datawhale, agent, works]
#   tokens[1:]  = [agent,     learns, datawhale, agent, works]
#   zip 结果    = (datawhale,agent), (agent,learns), (learns,datawhale),
#                 (datawhale,agent), (agent,works)
# 注意：zip 在较短一侧结束，最后一对到 works 为止，共 5 个 bigram（总词数-1）
bigrams = zip(tokens, tokens[1:])

# Counter 统计每个二元组出现次数；键是元组 (前词, 后词)，值是次数
# 例：{('datawhale','agent'): 2, ('agent','learns'): 1, ('agent','works'): 1, ...}
bigram_counts = collections.Counter(bigrams)

# 用元组当键查询词对频次；找不到时 Counter 返回 0（普通 dict 会 KeyError）
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]  # 2

# 分母复用第一步的 count_datawhale（= Count(前词)），不是总词数
p_agent_given_datawhale = count_datawhale_agent / count_datawhale  # 2/2 = 1.0
print(f"第二步: P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.3f}")

# --- 第三步：计算 P(learns|agent) ---
# 【为什么要这一步？】
# 继续按 Bigram 链式展开第三个词：
#   P(learns | agent) = Count(agent, learns) / Count(agent)
# 含义：在语料里，凡是出现过 agent 的地方，后面紧跟 learns 的比例是多少。
#
# 【本例手工对照】
# 词对 (agent, learns) 出现 1 次；(agent) 出现 2 次（后面分别接 learns / works）
# → 1/2 = 0.5，说明 agent 后面有一半概率是 learns，一半是 works。

# 分子：相邻词对 (agent, learns) 的次数（复用上面已统计好的 bigram_counts）
count_agent_learns = bigram_counts[('agent', 'learns')]  # 1

# 分母：前词 agent 的一元词频（注意：不是 Count(agent, *) 的 bigram 之和在数学上等价于
# 非句尾出现次数；本迷你语料里 agent 都不在句尾，所以用 tokens.count 即可）
count_agent = tokens.count('agent')  # 2

p_learns_given_agent = count_agent_learns / count_agent  # 1/2 = 0.5
print(f"第三步: P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.3f}")

# --- 最后：将概率连乘 ---
# P(datawhale agent learns) ≈ P(datawhale) · P(agent|datawhale) · P(learns|agent)
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(f"最后: P('datawhale agent learns') ≈ {p_datawhale:.3f} * {p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = {p_sentence:.3f}")
