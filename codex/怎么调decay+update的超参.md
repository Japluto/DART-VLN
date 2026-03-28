我会优先改这三个，按顺序来：

1. `dynamic_memory_decay_lambda`
2. `dynamic_memory_base_gate`
3. `dynamic_memory_match_radius`

原因是这三个最直接决定“memory 会不会真的动起来”，而且最容易看出效果。

**第一优先：`dynamic_memory_decay_lambda`**
这是我最建议先动的。

原因：
- 它只影响读出侧，不改写 memory 本体
- 风险比 `update` 小
- 更适合先验证“soft forgetting 本身有没有帮助”
- 如果有收益，通常更稳，不容易把行为改坏

怎么改：
- 默认是 `0.12`
- 我建议先试：
  - `0.05`
  - `0.08`
  - `0.12`
  - `0.18`

怎么判断：
- 如果和 `off` 几乎没差别，说明 decay 太弱
- 如果 `SR` 涨了但 `SPL` 掉很多，说明 decay 太强
- 一般目标是找到“SR 小涨，SPL 基本不掉”的点

**第二优先：`dynamic_memory_base_gate`**
这是 update 侧最核心、也最直观的强度参数。

原因：
- 它决定“默认写入有多猛”
- 比起先动 `novelty_weight`，它更像总旋钮
- 很适合先判断 update 机制是不是整体太保守/太激进

怎么改：
- 默认是 `0.15`
- 我建议先试：
  - `0.08`
  - `0.12`
  - `0.15`
  - `0.22`

怎么判断：
- `SR` 不动：可能太小
- `SPL` 明显掉：可能太大
- 如果一调大就不稳，说明 write 机制本身应该更保守

**第三优先：`dynamic_memory_match_radius`**
这是 update 侧另一个非常关键的参数。

原因：
- 它控制“新 patch 是融合进旧 memory，还是新开 slot”
- 这个参数往往决定你这个 heuristic 到底是在“去冗余”，还是在“保细节”
- 对 memory 结构影响很大

怎么改：
- 默认是 `0.75`
- 我建议先试：
  - `0.5`
  - `0.75`
  - `1.0`

怎么判断：
- 太大：容易错误合并，可能损伤区分度
- 太小：memory 太碎，dynamic update 形同虚设

**我暂时不建议最先改的**
- `novelty_weight`
- `age_weight`
- `repeat_weight`
- `min_gate`
- `max_gate`
- `min_mem_weight`

不是它们不重要，而是它们更像“细调参数”。

我的建议是先用前面三个把系统调到一个合理区间，再动这些细项。

**最省算力的优先顺序**
如果你现在就想开始跑，我建议：

1. 先只跑 `DECAY_ONLY`，扫 `decay_lambda`
2. 然后只跑 `UPDATE_ONLY`，扫 `base_gate`
3. 再扫 `match_radius`
4. 最后才碰 `novelty_weight`

**一句话版本**
- 想先稳一点：先改 `decay_lambda`
- 想先看 update 有没有戏：先改 `base_gate`
- 想决定 memory 合并尺度：再改 `match_radius`

如果你愿意，我可以直接给你一组“只改这 3 个参数”的最小 6 条实验命令。















可以，而且这类超参我建议按“先分治、再联调”的方式来调，不要一上来全开一起扫。

**总原则**
你这个改动本质上是 inference-time heuristic，不是训练出来的，所以调参目标应该很明确：

- 先看 `SR` 能不能稳步上升
- 同时盯住 `SPL` 不要掉太多
- 再看 `nav_error / oracle_sr / nDTW`
- 优先在 `val_unseen` 上选参数，`val_seen` 只用来判断是不是过拟合启发式

我会把它拆成两阶段。

**第一阶段：分开调**
先固定一边，只调另一边。

1. 先调 `UPDATE_ONLY`
目标：确认“写入策略”本身有没有收益。

重点参数：
- `dynamic_memory_match_radius`
- `dynamic_memory_base_gate`
- `dynamic_memory_novelty_weight`
- `dynamic_memory_age_weight`
- `dynamic_memory_repeat_weight`
- `dynamic_memory_min_gate`
- `dynamic_memory_max_gate`

我的经验判断：
- `match_radius` 决定“合并不合并”，这是最敏感的
- `base_gate` 决定整体更新强度
- `novelty_weight` 通常是最该优先调高/调低的
- `age_weight` 影响“旧信息多久后允许刷新”
- `repeat_weight` 影响“反复看到的东西还要不要继续写”

建议顺序：
1. 先固定除了 `match_radius` 外的其他值，只扫 `match_radius`
2. 再固定 `match_radius`，扫 `base_gate`
3. 再扫 `novelty_weight`
4. 最后再小范围补 `age_weight / repeat_weight`

一个比较稳的初始搜索区间可以是：
- `match_radius`: `0.4 / 0.6 / 0.75 / 0.9 / 1.2`
- `base_gate`: `0.05 / 0.10 / 0.15 / 0.25`
- `novelty_weight`: `0.15 / 0.30 / 0.45 / 0.60`
- `age_weight`: `0.0 / 0.1 / 0.2 / 0.3`
- `repeat_weight`: `0.0 / 0.1 / 0.2 / 0.3`

直觉上：
- `SR` 上不去：通常是更新不够，优先加 `base_gate` 或 `novelty_weight`
- `SPL` 掉很多：通常是 memory 被刷得太快，优先减 `base_gate`、减 `match_radius`、减 `age_weight`
- `val_seen` 好、`val_unseen` 差：通常是启发式太激进或太依赖重复 pattern，优先减 `repeat_weight` 或减 `base_gate`

2. 再调 `DECAY_ONLY`
目标：确认“遗忘读出”能不能带来更稳的收益。

重点参数：
- `dynamic_memory_decay_lambda`
- `dynamic_memory_min_mem_weight`
- `dynamic_memory_max_mem_weight`

建议顺序：
1. 先扫 `decay_lambda`
2. 再扫 `min_mem_weight`

建议区间：
- `decay_lambda`: `0.02 / 0.05 / 0.08 / 0.12 / 0.18`
- `min_mem_weight`: `0.2 / 0.35 / 0.5 / 0.7`
- `max_mem_weight`: 一般先固定 `1.0`

直觉上：
- 如果 decay 太强，`SPL` 和 `oracle_sr` 会一起掉
- 如果 decay 太弱，结果会很接近 `OFF`
- 我会优先找“比 OFF 稍好、而且波动最小”的 decay，而不是最激进的 decay

**第二阶段：再做 FULL 联调**
等你分别找到：
- 一个最稳的 `UPDATE_ONLY`
- 一个最稳的 `DECAY_ONLY`

再组合成 `FULL`，这时候只做小幅微调，不要重新大扫。

联调时优先看这几个冲突：
- `update` 已经很激进时，`decay_lambda` 往往要更小
- `match_radius` 偏大时，`decay` 不能再太强
- `novelty_weight` 偏高时，`repeat_weight` 通常也要一起稍微提高一点，不然 memory 会太活跃

**我建议的实际调参流程**
按单卡研究最省时间的方式来：

1. `OFF` 跑一组基线
2. `UPDATE_ONLY` 先扫 `match_radius`
3. 固定最好 `match_radius` 后扫 `base_gate`
4. 再扫 `novelty_weight`
5. 得到一个 `UPDATE_ONLY` 最优点
6. 回到 `OFF`，单独调 `DECAY_ONLY`
7. 先扫 `decay_lambda`
8. 再扫 `min_mem_weight`
9. 得到一个 `DECAY_ONLY` 最优点
10. 组合成 `FULL`
11. 在 `FULL` 上只微调 `decay_lambda` 和 `base_gate`

**我最推荐先试的几组**
如果你不想一开始就大范围扫，我会先试这几组：

- 保守 `UPDATE_ONLY`
```bash
DYNAMIC_MEMORY_MODE=update_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_match_radius 0.6 --dynamic_memory_base_gate 0.10 --dynamic_memory_novelty_weight 0.30 --dynamic_memory_age_weight 0.10 --dynamic_memory_repeat_weight 0.15" \
bash scripts/run_r2r.sh test
```

- 稍激进 `UPDATE_ONLY`
```bash
DYNAMIC_MEMORY_MODE=update_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_match_radius 0.9 --dynamic_memory_base_gate 0.15 --dynamic_memory_novelty_weight 0.45 --dynamic_memory_age_weight 0.20 --dynamic_memory_repeat_weight 0.20" \
bash scripts/run_r2r.sh test
```

- 保守 `DECAY_ONLY`
```bash
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.05 --dynamic_memory_min_mem_weight 0.50" \
bash scripts/run_r2r.sh test
```

- 稍激进 `DECAY_ONLY`
```bash
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35" \
bash scripts/run_r2r.sh test
```

- 保守 `FULL`
```bash
DYNAMIC_MEMORY_MODE=full \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_match_radius 0.75 --dynamic_memory_base_gate 0.12 --dynamic_memory_novelty_weight 0.35 --dynamic_memory_age_weight 0.15 --dynamic_memory_repeat_weight 0.15 --dynamic_memory_decay_lambda 0.05 --dynamic_memory_min_mem_weight 0.50" \
bash scripts/run_r2r.sh test
```

**怎么看结果最有价值**
不要只看一个数字。建议每次至少记录：

- `SR`
- `SPL`
- `nav_error`
- `oracle_sr`
- `nDTW`

我会优先保留这种参数：
- `SR` 有提升
- `SPL` 基本持平或小降
- `nav_error` 没恶化
- `val_seen` 和 `val_unseen` 都不是极端割裂

**一句话判断每个参数**
- `match_radius`：决定 memory slot 合并尺度
- `base_gate`：决定默认写入有多猛
- `novelty_weight`：决定新信息是否容易覆盖旧信息
- `age_weight`：决定旧 memory 是否更容易被刷新
- `repeat_weight`：决定重复观察是否被抑制
- `decay_lambda`：决定旧 memory 在读出时衰减多快
- `min_mem_weight`：决定最旧/最无价值 memory 最低还能保留多少影响

如果你愿意，我下一步可以直接帮你做一版“最省算力的调参计划表”，比如先给出 `R2R` 上 8 组最值得跑的实验命令。

————
这些都不是训练参数，也不是新模型参数，而是你这次加进去的 **inference-time heuristic 超参数**，也就是“推理时 memory 读写规则的手工控制量”。

可以分成 3 类看：

**1. 模式开关**
- `DYNAMIC_MEMORY_MODE`
作用：决定跑哪种 ablation
可选：
- `off`：全关，原始行为
- `update_only`：只开动态写入
- `decay_only`：只开软遗忘
- `full`：写入和遗忘都开

**2. update 写入参数**
这些在 `env.py` 里生效，控制“新观测来了以后，怎么更新旧 memory”。

- `dynamic_memory_match_radius`
作用：新 patch 和旧 memory 多近才算同一个 slot
大了更容易合并，小了更容易新建

- `dynamic_memory_base_gate`
作用：默认更新强度
越大表示更容易用新信息改写旧 memory

- `dynamic_memory_novelty_weight`
作用：新旧特征越不一样，更新得越猛
这是“鼓励新信息写入”的核心参数

- `dynamic_memory_age_weight`
作用：memory 越久没更新，越容易被刷新

- `dynamic_memory_repeat_weight`
作用：一个 memory 被反复看见很多次后，抑制继续强更新

- `dynamic_memory_min_gate`
作用：更新门最小值，下限保护

- `dynamic_memory_max_gate`
作用：更新门最大值，上限保护

**3. decay 遗忘参数**
这些主要在 `vilmodel.py` 读出时生效，控制“旧 memory 在读取时被降权多少”。

- `dynamic_memory_decay_lambda`
作用：衰减速度
越大表示旧 memory 忘得越快

- `dynamic_memory_min_mem_weight`
作用：最老/最低价值 memory 的最低保留权重
越小表示忘得更狠

- `dynamic_memory_max_mem_weight`
作用：memory 权重上限
通常保持 `1.0`

**一句话总结**
- `match_radius`：合不合并
- `base_gate`：默认写多猛
- `novelty_weight`：新信息值不值得写
- `age_weight`：旧信息该不该刷新
- `repeat_weight`：重复信息要不要抑制
- `decay_lambda`：旧 memory 忘多快
- `min_mem_weight`：最差 memory 至少还保留多少

如果你愿意，把你刚跑的几组参数和结果贴给我，我可以直接帮你判断每组是在测“更强更新”还是“更强遗忘”。