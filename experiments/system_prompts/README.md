# System prompts (Section 4)

Can a system prompt on the unfinetuned model reproduce what the subliminal
students learned? Each condition reruns the evaluation of
[`random_mlp`](../random_mlp) or [`letter_counting`](../letter_counting) on
the base Qwen3.6-35B-A3B with one added system turn. The conditions are
instructions without examples, and one instruction followed by labeled
training examples. All prompt texts are in the two scripts and in the paper
appendix.

| Random MLP condition | Valid answers | MAE |
|---|---|---|
| No system prompt | 0.1% | – |
| Task and format instruction | 100% | 28.62 |
| Exact architecture description | 100% | 12.30 |
| Instruction + 8 / 32 / 128 examples | 100% | 6.05 / 5.87 / 5.59 |
| Mean predictor | | 5.46 |
| Subliminal student | | 3.73 |

On letter counting, the best instruction reaches 29.8% (no system prompt:
27.3%; subliminal student: 44.1%), and adding examples lowers accuracy.

```bash
python random_mlp.py             # results/random_mlp/<condition>/
python random_mlp_bootstrap.py   # MAE with 95% bootstrap intervals over parsed answers
python letter_counting.py        # results/letter_counting/
```

The architecture description gives the output range of the training targets
(−37 to 27 for the paper network).
