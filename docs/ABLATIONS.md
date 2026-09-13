# Ablation implementation status

The table below describes **historical pilot modes**, not the new study variants.
The [study workflow](STUDY_RUNBOOK.md) now has explicit `isolated_raw`,
`isolated_typed`, `shared_verified`, and `isolated_verified` systems alongside
instrumented/uninstrumented continuous baselines. They have context-capture and
real-interpreter tests; live development validation is ongoing. `shared_verified`
retains one native RLM history/interpreter rather than substituting summaries.
Select these study variants in a frozen JSON config, not via `ablation_mode`.

Only `ablation_mode="none"` is included in the pilot runner. The other modes are programmatic prototypes, not validated independent experimental conditions.

| Mode | Actual behavior | Limitation |
|---|---|---|
| `none` | Projects typed required facts and gates supported outputs with witnesses | Default pipeline |
| `no_verification` | Skips witness evaluation and directly promotes supported outputs | Still uses typed data and input contracts |
| `no_typing` | Supplies all canonical facts and skips the input contract check | Still uses Fact/StepResult models and witnesses; does not remove typing |
| `shared_history` | Appends prior step goals, values and statuses | Does not replay full RLM reasoning trajectories |
| `raw_summaries` | Supplies textual fact summaries and an empty typed input list | Deterministic witnesses still require typed inputs; not a functioning replacement architecture |
| `llm_verifier` | Raises NotImplementedError | No sealed adjudicator exists |
| deterministic-only | No separate mode | Requires task and assurance-policy design |

Mock workers use evaluator labels and scripted error behavior. Passing a mode through a mock test verifies plumbing only, not its scientific validity. A controlled ablation study remains future work.
