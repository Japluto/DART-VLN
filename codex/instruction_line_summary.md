# R2R Instruction-Side Experiments Summary

## Scope

This note summarizes the small `R2R-only`, `test-time-only`, `no-training` instruction-side experiments that were added on top of the current `GridMM_ff` discrete navigation pipeline.

The goal of this line was to test whether lightweight instruction-side heuristics could improve navigation without changing the model architecture or requiring retraining.

## 1. Instruction Augmentation

### Method

Before tokenization / encoding, append a short keyword summary extracted from the original instruction text:

`original_instruction + " [SEP] key cues: " + keyword_summary`

This was implemented as a very small, rule-based test-time augmentation.

### Result

This variant produced clear negative results.

Typical behavior:

- `SR` dropped
- `SPL` dropped
- `nDTW` dropped
- `lengths` increased
- `nav_error` worsened

### Interpretation

This suggests that direct text-side rewriting perturbs the instruction distribution too much for the frozen pretrained encoder. Even though the appended keywords looked reasonable to humans, they likely acted as repeated or biased extra text rather than useful guidance.

### Conclusion

This line should be stopped.

## 2. Global Instruction-Aware Rerank

### Method

Do not modify instruction text or encoder input.

Instead, extract a simple direction cue such as:

- `left`
- `right`
- `straight`

Then, during `R2R` inference only, apply a tiny rerank boost to non-stop candidate actions whose direction matches the extracted cue.

This rerank was:

- test-time only
- applied before final `argmax`
- uncertainty-gated
- early-stage gated
- limited to non-stop top-k candidates
- STOP untouched

### Result

Under conservative settings:

- `rerank_trigger_count` was low
- `rerank_action_changed_count` was near zero or one
- metrics were nearly unchanged

Under an aggressive sanity-check setting:

- `rerank_trigger_count` increased substantially
- `rerank_action_changed_count` increased from about `1` to about `5`
- but navigation metrics did not improve and instead slightly worsened

### Interpretation

This is an important diagnosis:

- the mechanism **can** change actions
- but the changes are not reliably helpful

The likely reason is that a global direction cue extracted from the whole instruction is too coarse. Later subgoals pollute early-step guidance, so the rerank can intervene but does not intervene intelligently enough.

### Conclusion

This line is not completely inert, but in its current minimal form it is not worth continuing to tune aggressively.

## 3. Local First-Clause + Explicit Action Phrase Rerank

### Method

To make the direction prior cleaner, the cue source was restricted to the first local clause only.

The extraction rule was also tightened:

- use the first clause only
- only accept explicit action phrases such as:
  - `turn left`
  - `turn right`
  - `go straight`
  - `walk straight`
  - `head straight`
- ignore bare spatial mentions like:
  - `hallway on the left`
  - `room to the right`

### Result

This version became too conservative.

Observed behavior:

- `rerank_trigger_count` dropped to about `1`
- `rerank_action_changed_count` dropped to `0`
- final metrics became effectively identical to the plain `full` baseline

### Interpretation

This version improved precision of cue extraction, but recall became too low. The rerank almost never engaged, so it effectively behaved like no rerank at all.

### Conclusion

This line is too conservative to be useful in practice.

## Overall Conclusion

The instruction-side test-time line has now explored two useful extremes:

1. stronger / broader rerank:
   - can change actions
   - does not improve final metrics

2. stricter / cleaner rerank:
   - barely changes actions
   - behaves like the baseline

This strongly suggests that the current family of minimal direction-word priors is near its practical limit in this setup.

The issue is no longer just parameter tuning. The shallow direction-word signal does not appear rich enough to reliably improve action choice in this trained agent.

## Practical Recommendation

- Stop `instruction augmentation`
- Stage-stop the current minimal `instruction-aware rerank` line
- Keep instruction-side work as a lower-priority exploration track
- Continue prioritizing the more stable memory-side work, especially:
  - `dynamic memory`
  - and within that, the relatively safer `decay_only` direction

## One-Line Summary

For the current `GridMM_ff` `R2R` setup, lightweight no-training instruction-side heuristics were either:

- strong enough to change actions but not improve outcomes, or
- safe enough to avoid harm but too weak to matter

So this line is best treated as diagnostically complete for now.
