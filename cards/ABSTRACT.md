# Abstract

Safety training for open models is mostly done and measured in English. Many people
type in code-mixed language instead, such as Hinglish (romanised Hindi mixed with English).
We asked whether a small multilingual model's refusals survive that switch, why they fail if
they don't, and whether the failure can be fixed cheaply.

We built a parallel benchmark of 449 prompts in English, Hindi, Marathi and Hinglish (1,796
rows):
- 199 harmful prompts from HarmBench.
- 250 harmless prompts that sound risky, from XSTest, to measure over-refusal.

CuratorKIT cleaned and de-duplicated the set, and AuditKIT's WildGuard judge scored every
answer. On Tiny Aya Fire, attack success rises from 7.1% in English to 17.7% in Hindi, 31.9%
in Marathi and 51.3% in Hinglish, while over-refusal stays low. The model is not more
permissive overall; it specifically fails to refuse harmful requests outside English.
Tiny Aya Global shows the same pattern.

Using SafeTune, we extracted the refusal direction from English prompts and tested two
explanations: Hinglish has its own direction, or it uses the same one weakly. The evidence
supports weak activation:
- Per-language directions are close to the English one (cosine 0.78–0.94).
- Harmful Hinglish prompts project onto it at 17.1, against 24.9 for English.
- Projection tracks attack success across the four languages (r = −0.97).

Because the concept already exists, we restore it at inference time with conditional
activation steering. A gate fitted with SafeTune decides per prompt whether to add the
direction. The steering strength was chosen on a validation split by a rule fixed in
advance; the numbers below are on a held-out test split. Attack success falls to 6.2% in
Hindi, 11.5% in Marathi and 31.0% in Hinglish, and over-refusal is identical to the
unsteered model in every language. The same push without the gate makes the model refuse
20–55% of harmless prompts.

Red-teaming and multi-turn audits show where the fix holds and where it doesn't:
- It holds against a fiction jailbreak and three-turn gradual escalation (Hinglish 55% → 22.5%).
- It does not hold against repeated sampling: in Hinglish, 60% of harmful prompts still break within five tries.
- It cannot stop a hidden instruction in a document, which leaks a secret 90–100% of the time in every language.

Analysis of the remaining failures shows the gate is not the bottleneck. It fires on 95–99%
of harmful prompts, and every Hinglish prompt that still got through had been steered. A
single English-derived direction at one layer is the limit. We also document an evaluation
bug that would have reported a false result, plus the check that caught it. We release the
benchmark, the redacted judged logs behind every number, the steering vectors and the code.
