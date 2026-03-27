# Discrete Eval Commands

下面这些命令默认在新的工作目录 [`GridMM_ff`](/home/japluto/VLN/GridMM_ff) 内使用。

## 环境准备

```bash
conda activate gridmm
cd /home/japluto/VLN/GridMM_ff/map_nav_src
```

## R2R Eval

推荐直接走脚本：

```bash
bash scripts/run_r2r.sh test
```

这条命令默认会使用：

- 数据根目录：`/home/japluto/VLN/GridMM_ff/datasets`
- checkpoint：`/home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best`
- 输出目录：`/home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/Grid_Map-dagger-vitbase-single-gpu-seed.0`

如果你想显式指定 checkpoint：

```bash
RESUME_FILE=/home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best \
bash scripts/run_r2r.sh test
```

## REVERIE Eval

当前推荐直接用主入口命令：

```bash
python main_nav_obj.py \
  --root_dir /home/japluto/VLN/GridMM_ff/datasets \
  --dataset reverie \
  --output_dir /home/japluto/VLN/GridMM_ff/default/reverie_eval \
  --world_size 1 \
  --seed 0 \
  --tokenizer bert \
  --enc_full_graph \
  --graph_sprels \
  --fusion dynamic \
  --multi_endpoints \
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
  --test \
  --resume_file /home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best
```

## 输出位置

R2R:

- 日志：`/home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/Grid_Map-dagger-vitbase-single-gpu-seed.0/logs`
- 预测：`/home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/Grid_Map-dagger-vitbase-single-gpu-seed.0/preds`

REVERIE:

- 日志：`/home/japluto/VLN/GridMM_ff/default/reverie_eval/logs`
- 预测：`/home/japluto/VLN/GridMM_ff/default/reverie_eval/preds`

## 备注

- `R2R` 不建议直接裸跑 `python main_nav.py ...`，这套代码默认会走分布式初始化，直接起容易报 `rank/local_rank` 相关错误。
- 跑完如果担心显存残留，可以先看：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
