You are working inside my repository `gridmm_ff`, which is a modified GridMM codebase for discrete VLN experiments.

Your task is to implement a **minimal, inference-time dynamic memory update + forgetting mechanism** for the Grid Memory Map, with the following hard constraints:

## High-level goal
Improve SR (and ideally not hurt SPL too much) on discrete VLN benchmarks by making the grid memory update rule smarter, without training a new model.

The idea is:
- the grid memory should not keep accumulating stale / redundant / noisy information forever
- recently useful or novel information should be integrated more strongly
- old, repeatedly visited, low-value memory should be softly downweighted during readout
- this must be done with **heuristics / rule-based logic only**
- **do not add any new trainable module**
- **do not introduce any retraining requirement**
- keep code changes as local and small as possible

## Important constraints
1. No new trainable parameters.
2. No architecture rewrite.
3. No retraining, no finetuning, no new loss.
4. Prefer modifying existing GridMM memory update / readout logic instead of adding large new files.
5. Default behavior must remain unchanged when the feature flag is OFF.
6. The implementation must be easy to ablate.

## What to implement
Implement a lightweight dynamic memory mechanism with two parts:

### Part A: dynamic update gate when writing into grid memory
When a grid cell already has stored memory and a new observation arrives, do not always overwrite / merge it with a fixed rule.

Instead, compute a **heuristic update gate** based on signals such as:
- novelty of the new feature relative to the stored feature
- age since last update
- visit count / repeated observation count

A good default design is:

- `novelty = 1 - cosine_similarity(old_feat, new_feat)`
- `age = current_step - last_update_step`
- `repeat_factor = function(visit_count)`

Then compute an update ratio such as:
- `update_gate = clamp(base + w_novelty * novelty + w_age * age_norm - w_repeat * repeat_norm, min_gate, max_gate)`

And update:
- `mem_feat = normalize((1 - update_gate) * old_feat + update_gate * new_feat)`

Also update metadata per cell:
- `last_update_step`
- `visit_count`

Do this in the smallest possible place where the grid memory is actually updated.

### Part B: soft forgetting / decay during memory readout
Do NOT implement destructive hard deletion unless absolutely necessary.

Instead, implement a **soft forgetting weight** per memory cell, based on:
- recency
- visit count saturation / repeated observations
- optional novelty history if easy to add

Example:
- cells that are very old and repeatedly visited without much novelty should get lower memory weights
- cells updated recently or with stronger novelty should keep higher weights

Then apply this soft memory weight at readout time, for example by:
- multiplying the grid memory embedding before attention / scoring, or
- modifying the grid memory score / mask / attention logits in a minimal and safe way

Prefer the least invasive implementation.

## Expected repository exploration
First inspect the repo and identify the real files that correspond to GridMM memory construction / update / readout.

Likely relevant places may include files or logic around:
- `VLN_CE/vlnce_baselines/models/gridmap/`
- `vilmodel.py`
- code paths involving `gmap_*`, `gridmap`, `global map`, `vp_pos_fts`, `gmap_pos_fts`, `gmap_embeds`, `fuse_weights`, or memory update / map aggregation logic

If exact paths differ in this repo, locate the correct ones before editing.

## Implementation requirements
1. Add a config flag to enable/disable the feature, e.g.:
   - `DYNAMIC_MEMORY.ENABLED`
2. Add tunable hyperparameters, for example:
   - `BASE_GATE`
   - `NOVELTY_WEIGHT`
   - `AGE_WEIGHT`
   - `REPEAT_WEIGHT`
   - `MIN_GATE`
   - `MAX_GATE`
   - `DECAY_ENABLED`
   - `DECAY_LAMBDA`
   - `MIN_MEM_WEIGHT`
   - `MAX_MEM_WEIGHT`

3. Add simple ablation modes if easy:
   - OFF
   - UPDATE_ONLY
   - DECAY_ONLY
   - FULL

4. Keep all new logic deterministic and easy to read.
5. Add brief comments explaining the algorithm.
6. Avoid broad refactors.

## Preferred design choice
I prefer a **soft, metadata-based approach** over a heavy redesign.

That means:
- store a small amount of metadata per grid cell
- compute update/decay weights from metadata
- integrate with existing memory tensors as locally as possible

Avoid redesigning the whole map representation.

## Deliverables
Please do the following in order:

### Step 1: inspect and plan
Before editing, inspect the repository and tell me:
- which file(s) implement grid memory update
- which file(s) implement memory readout / scoring
- your minimal edit plan in 5-10 bullet points

### Step 2: implement
Make the code changes.

### Step 3: expose configs
Add config options in the proper config file(s) used by this repo.

### Step 4: validate
Run the lightest possible validation:
- import / syntax checks
- one dry-run or minimal command if possible
- do not launch expensive training

If there is an existing eval command, prefer a smoke test rather than a full experiment.

### Step 5: summarize
At the end, report:
1. changed files
2. exact algorithm implemented
3. config flags added
4. how to run with the feature on/off
5. possible risks / edge cases
6. a suggested ablation order

## Research intent
This is a small, practical research modification for a single-GPU setup.
I care about:
- minimal code changes
- no retraining
- high implementation clarity
- easy ablation
- plausible SR improvement

## Additional guidance
- If there are multiple possible insertion points, choose the one with the smallest code footprint.
- If the repo already has any notion of map confidence / update weighting, reuse it instead of adding parallel machinery.
- If you need to choose between “more elegant” and “less invasive”, choose less invasive.
- If a hard deletion design is risky, prefer soft weighting.
- If instruction-conditioned signals would require touching too many modules, skip them for now. This task is only about dynamic memory update + forgetting, not instruction-aware reading.

Now start by inspecting the repo and giving me the Step 1 plan only. Do not edit code before identifying the exact files and insertion points.

---
Important: do not refactor unrelated code, do not change training logic, and do not introduce any new learnable module.