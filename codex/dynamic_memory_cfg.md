# Dynamic Memory Config

这份文档说明当前 `dynamic memory` 的实现方式，以及三个离散数据集上的四组实际运行命令：

- `OFF`
- `UPDATE_ONLY`
- `DECAY_ONLY`
- `FULL`

默认工作目录：

```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src
```

## 遗忘机制是怎么实现的

当前实现是纯规则、纯推理时生效的轻量机制，不增加新可学习参数，不需要重训练。

### Part A: 动态写入

实现文件：

- `/home/japluto/VLN/GridMM_ff/map_nav_src/r2r/env.py`
- `/home/japluto/VLN/GridMM_ff/map_nav_src/reverie/env.py`
- `/home/japluto/VLN/GridMM_ff/map_nav_src/rxr/env.py`

核心逻辑：

- 新 patch 到来时，先用世界坐标尝试匹配一个已有 memory slot。
- 如果没有匹配到，就新建一个 slot。
- 如果匹配到了，就计算一个启发式更新门 `update_gate`。

用到的信号：

- `novelty = 1 - cosine_similarity(old_feat, new_feat)`
- `age = current_step - last_update_step`
- `repeat_norm = visit_count / (visit_count + 1)`

更新门：

```text
update_gate =
  clamp(
    BASE_GATE
    + NOVELTY_WEIGHT * novelty
    + AGE_WEIGHT * age_norm
    - REPEAT_WEIGHT * repeat_norm,
    MIN_GATE,
    MAX_GATE
  )
```

写入方式：

```text
mem_feat = normalize((1 - update_gate) * old_feat + update_gate * new_feat)
```

同时维护元数据：

- `last_update_step`
- `visit_count`
- `novelty_ema`

### Part B: 软遗忘

实现文件：

- `/home/japluto/VLN/GridMM_ff/map_nav_src/models/vilmodel.py`

核心逻辑：

- 不做硬删除。
- 在 `grid cell` 内部聚合 patch 时，为每个 patch 计算一个 `memory_weight`。
- 这个权重越小，这个 patch 在 cell 聚合里贡献越小。

权重形式：

```text
memory_weight =
  clamp(
    exp(-DECAY_LAMBDA * age)
    * (1 - REPEAT_WEIGHT * repeat_norm)
    * (0.5 + 0.5 * novelty_ema),
    MIN_MEM_WEIGHT,
    MAX_MEM_WEIGHT
  )
```

然后把它加到 cell 内部 patch 聚合的 softmax logits 上：

```text
cell_logits = text_match_logits + log(memory_weight)
```

这意味着：

- 很旧、重复很多次、且没什么新意的 patch 会被软降权。
- 最近更新、较新颖的 patch 会更容易在 cell 聚合中保留影响。

## 参数和开关

定义文件：

- `/home/japluto/VLN/GridMM_ff/map_nav_src/r2r/parser.py`
- `/home/japluto/VLN/GridMM_ff/map_nav_src/reverie/parser.py`
- `/home/japluto/VLN/GridMM_ff/map_nav_src/rxr/parser.py`

主要参数：

- `--dynamic_memory_enabled`
- `--dynamic_memory_mode {off,update_only,decay_only,full}`
- `--dynamic_memory_base_gate`
- `--dynamic_memory_novelty_weight`
- `--dynamic_memory_age_weight`
- `--dynamic_memory_repeat_weight`
- `--dynamic_memory_min_gate`
- `--dynamic_memory_max_gate`
- `--dynamic_memory_decay_enabled`
- `--dynamic_memory_decay_lambda`
- `--dynamic_memory_min_mem_weight`
- `--dynamic_memory_max_mem_weight`
- `--dynamic_memory_match_radius`

当前默认值：

```text
BASE_GATE=0.15
NOVELTY_WEIGHT=0.35
AGE_WEIGHT=0.20
REPEAT_WEIGHT=0.15
MIN_GATE=0.05
MAX_GATE=0.85
DECAY_LAMBDA=0.12
MIN_MEM_WEIGHT=0.35
MAX_MEM_WEIGHT=1.0
MATCH_RADIUS=0.75
```

## 模式解释

- `off`
  完全关闭，保持原始行为。
- `update_only`
  只启用动态写入，不启用软遗忘。
- `decay_only`
  不改 memory 写入，只在读出时做软遗忘。
- `full`
  同时启用动态写入和软遗忘。

---

## R2R 命令

checkpoint 默认使用：

```text
/home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best
```

### R2R OFF

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_nav.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset r2r \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/dm_off \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --batch_size 4 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best \
  --dynamic_memory_mode off
```

### R2R UPDATE_ONLY

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_nav.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset r2r \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/dm_update_only \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --batch_size 4 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best \
  --dynamic_memory_enabled \
  --dynamic_memory_mode update_only
```

### R2R DECAY_ONLY

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_nav.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset r2r \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/dm_decay_only \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --batch_size 4 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best \
  --dynamic_memory_enabled \
  --dynamic_memory_mode decay_only \
  --dynamic_memory_decay_enabled
```

### R2R FULL

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_nav.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset r2r \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/dm_full \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --batch_size 4 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best \
  --dynamic_memory_enabled \
  --dynamic_memory_mode full \
  --dynamic_memory_decay_enabled
```

---

## REVERIE 命令

checkpoint 默认使用：

```text
/home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best
```

### REVERIE OFF

```bash
python3 main_nav_obj.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset reverie \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/REVERIE/exprs_map/eval/dm_off \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --multi_endpoints \
  --dagger_sample sample \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --max_objects 20 \
  --batch_size 1 \
  --features vitbase \
  --obj_features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --obj_feat_size 768 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best \
  --dynamic_memory_mode off
```

### REVERIE UPDATE_ONLY

```bash
python3 main_nav_obj.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset reverie \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/REVERIE/exprs_map/eval/dm_update_only \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --multi_endpoints \
  --dagger_sample sample \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --max_objects 20 \
  --batch_size 1 \
  --features vitbase \
  --obj_features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --obj_feat_size 768 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best \
  --dynamic_memory_enabled \
  --dynamic_memory_mode update_only
```

### REVERIE DECAY_ONLY

```bash
python3 main_nav_obj.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset reverie \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/REVERIE/exprs_map/eval/dm_decay_only \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --multi_endpoints \
  --dagger_sample sample \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --max_objects 20 \
  --batch_size 1 \
  --features vitbase \
  --obj_features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --obj_feat_size 768 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best \
  --dynamic_memory_enabled \
  --dynamic_memory_mode decay_only \
  --dynamic_memory_decay_enabled
```

### REVERIE FULL

```bash
python3 main_nav_obj.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset reverie \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/REVERIE/exprs_map/eval/dm_full \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --multi_endpoints \
  --dagger_sample sample \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 15 \
  --max_instr_len 200 \
  --max_objects 20 \
  --batch_size 1 \
  --features vitbase \
  --obj_features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --obj_feat_size 768 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best \
  --dynamic_memory_enabled \
  --dynamic_memory_mode full \
  --dynamic_memory_decay_enabled
```

---

## RxR 命令

说明：

- 当前仓库里还没有确认好的离散 `RxR eval` checkpoint。
- 所以下面给的是实际命令模板。
- 你需要把 `YOUR_RXR_CHECKPOINT` 替换成真实可用的 checkpoint。

### RxR OFF

```bash
python3 main_rxr.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset rxr \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/RXR/exprs_map/eval/dm_off \
  --world_size 1 \
  --seed 0 \
  --tokenizer xlm \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 20 \
  --max_instr_len 250 \
  --batch_size 1 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file YOUR_RXR_CHECKPOINT \
  --dynamic_memory_mode off
```

### RxR UPDATE_ONLY

```bash
python3 main_rxr.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset rxr \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/RXR/exprs_map/eval/dm_update_only \
  --world_size 1 \
  --seed 0 \
  --tokenizer xlm \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 20 \
  --max_instr_len 250 \
  --batch_size 1 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file YOUR_RXR_CHECKPOINT \
  --dynamic_memory_enabled \
  --dynamic_memory_mode update_only
```

### RxR DECAY_ONLY

```bash
python3 main_rxr.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset rxr \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/RXR/exprs_map/eval/dm_decay_only \
  --world_size 1 \
  --seed 0 \
  --tokenizer xlm \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 20 \
  --max_instr_len 250 \
  --batch_size 1 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file YOUR_RXR_CHECKPOINT \
  --dynamic_memory_enabled \
  --dynamic_memory_mode decay_only \
  --dynamic_memory_decay_enabled
```

### RxR FULL

```bash
python3 main_rxr.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset rxr \
  --output_dir /home/japluto/VLN/GridMM_ff/datasets/RXR/exprs_map/eval/dm_full \
  --world_size 1 \
  --seed 0 \
  --tokenizer xlm \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --expert_policy spl \
  --train_alg dagger \
  --num_l_layers 9 \
  --num_x_layers 4 \
  --num_pano_layers 2 \
  --max_action_len 20 \
  --max_instr_len 250 \
  --batch_size 1 \
  --features vitbase \
  --image_feat_size 768 \
  --angle_feat_size 4 \
  --ml_weight 0.2 \
  --feat_dropout 0.4 \
  --dropout 0.5 \
  --gamma 0. \
  --test --submit \
  --resume_file YOUR_RXR_CHECKPOINT \
  --dynamic_memory_enabled \
  --dynamic_memory_mode full \
  --dynamic_memory_decay_enabled
```

---

## 建议的 ablation 顺序

建议顺序：

1. `OFF`
2. `DECAY_ONLY`
3. `UPDATE_ONLY`
4. `FULL`

原因：

- `DECAY_ONLY` 最稳，最不容易破坏原本模型行为。
- `UPDATE_ONLY` 会改 memory 本身，收益可能更大，但也更激进。
- `FULL` 适合最后做组合验证。
