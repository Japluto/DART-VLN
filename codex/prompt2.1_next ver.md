You are working in my GridMM_ff repository.

We already implemented a minimal R2R-only dual STOP patch in:

- `map_nav_src/r2r/agent.py`
- `map_nav_src/r2r/parser.py`

The patch runs correctly, but the first experiment showed the following pattern:

- SR decreased
- OSR increased
- trajectory length increased
- SPL decreased

This strongly suggests that the current dual STOP is suppressing STOP too aggressively:
the agent reaches the goal region more often, but fails to stop correctly and keeps walking.

## Critical diagnosis
The current margin condition is effectively too weak / almost vacuous.

Right now the logic includes something like:

`stop_prob >= best_nonstop_prob - dual_stop_margin_thresh`

But dual STOP is only triggered when the original chosen action is already STOP (`a_t == 0`).
That means `stop_prob >= best_nonstop_prob` is already true by construction.
So with a positive margin threshold, this condition is almost always satisfied and provides little real filtering.

As a result, the current patch is behaving mostly like:
- require `t >= min_step`
- require `stop_prob >= score_thresh`

This is too blunt and is likely the reason for the observed metrics.

## Goal
Fix the dual STOP margin logic so that it becomes a meaningful confidence test,
while keeping the rest of the patch as unchanged as possible.

## Required change
Please modify the margin-based condition so that it has real discriminative power.

### Current weak form
Do NOT keep relying on:
- `stop_prob >= best_nonstop_prob - margin`

### Replace with a meaningful form
Use:
- `stop_prob - best_nonstop_prob >= dual_stop_margin_thresh`

This means STOP is only considered confident enough if it beats the best non-stop candidate by a clear positive margin.

## Important implementation constraints
1. Keep the patch R2R-only for now.
2. Do not redesign the whole dual STOP mechanism.
3. Do not touch training logic.
4. Keep all other current dual STOP structure unchanged unless absolutely necessary.
5. Preserve the existing `node_stop_scores` gating consistency.
6. Keep the patch minimal and local.

## Parameter update
Because the new margin definition is much stricter than the old one,
the old `dual_stop_margin_thresh = 0.15` is too large.

Please update the default margin threshold to a much smaller conservative value, for example:
- `0.03`

It is okay to choose `0.03` as the new default unless code context suggests a slightly better nearby value.

Keep the other defaults unchanged for now unless there is a strong reason:
- `dual_stop_enabled = False`
- `dual_stop_score_thresh = 0.55`
- `dual_stop_min_step = 3`
- `dual_stop_revisit_thresh = 2`

## Deliverables
Please do the following:

### Step 1
Inspect the current dual STOP implementation and confirm exactly where the current margin condition is applied.

### Step 2
Modify only the margin logic to:
- `stop_prob - best_nonstop_prob >= dual_stop_margin_thresh`

### Step 3
Update the parser default for `dual_stop_margin_thresh` to a small positive value such as `0.03`.

### Step 4
Run the lightest possible validation:
- syntax / import check
- smoke test if practical

### Step 5
Report back:
1. changed files
2. exact old margin logic
3. exact new margin logic
4. new default margin threshold
5. whether any other lines had to be adjusted
6. whether `node_stop_scores` gating remains consistent

## Very important
Do not broaden scope.
This task is only to repair the margin logic so the current dual STOP stops behaving like an overly harsh stop-threshold rule.