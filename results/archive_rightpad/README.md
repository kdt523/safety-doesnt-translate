# Corrupted first Phase 4 run (kept for transparency)

These files come from the first steering run, which generated in batches with
**right padding**. Shorter prompts in each batch were followed by pad tokens,
and the model wrote broken text (loops, fragments) for most answers: 81% of
answers to harmful prompts and 47-67% to harmless ones were degenerate.

WildGuard scored that broken text as "not harmful" and often as "refused", so
both headline effects were artefacts: the drop in attack success and the rise
in over-refusal. The val sweep used to choose alpha had the same fault.

How it was found: in the gated run, harmless prompts that the gate did not
steer should match the unsteered model exactly. They matched 42% of the time
when the prompt was the longest in its batch (no padding) and 0% when it was
among the shortest.

Fix: `common.load_model` now loads the tokenizer with `padding_side="left"`,
`phase4_steer.encode` refuses to run otherwise, and every result table reports
a broken-output rate. Phase 3 (direction) and the gate fit were not affected:
both pad left or run one prompt at a time.
