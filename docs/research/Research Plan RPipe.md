> **RPipe research plan: verifiable contracts for research execution and evidence consistency**
> Version 1.4; updated 2026-10-16. The [Chinese plan](<研究计划 RPipe.md>) is authoritative; this is its chapter-by-chapter English counterpart.
> Stage: plan revision and targeted preliminary literature review; no new scientific experiments under this plan.
> Implementation baseline: dev `8480d795d0cb09714deb9bdd4bd79f41bdc5b2a9`; the published reference is v0.4.1. Dev is not a release.
> This round uses no GPU, timing study, training or data download, and changes no file architecture or artifact layout.

## Human-review checkpoint (2026-10-16,12:53 instruction)

- **Completed:** the three contracts and overnight PR103–125; specifically PR117/119/121/123/125 final heads and actual dev all passed real CI. Original failures, mistaken hypotheses and scientific negative evidence are retained.
- **One plan:** this English counterpart and the Chinese authority describe the same project plan. Five Study/historical/template protocols are fully included in Chapter XI; separate PLAN.md files are removed, while reports/JSON/figures/Run artifacts stay put.
- **Conclusion:** supports measured engineering contracts, not complete current-dev scientific revalidation or external superiority. Literature remains a targeted preliminary review with explicit reading depth.
- **Gaps/next:** [Chapter IX](#gaps-next) prioritizes CPU correctness and literature; E3/E4 require separate resource/run authorization. No timing/GPU/main/tag activity is started here.
- **Navigation:** [overnight repairs](#overnight-correctness), [complete Study protocols](#study-plan-inventory), [literature evidence](literature.md).

# I. Inspiration and initial problem

## 1. Topic, origin and value

This is systems and applications research, informed by reproducibility methodology and a limited reproduction of historical implementations. RPipe is a source template and execution foundation for research projects, not a new classifier, optimizer or quantization method. Its object of study is the evidence chain from declarations through execution, persistence and read-only presentation to human conclusions.

Three observed failure families motivate it: failed writes replacing earlier evidence; damaged metadata breaking or misleading an entire display; and concurrent publication colliding or exposing incomplete text. The recent compare counterexample adds an important boundary: equal numbers alone do not establish correctness when both sides load the same checkpoint outside their Run. See the [artifact consistency design](../development/artifact_consistency_design.md), [development record](../development/record.md) and [#101](https://github.com/DreamSoul-AI/RPipe/issues/101). These are observations about this repository, not evidence that other frameworks share the defects.

The initial question is whether a small set of evidence-preservation, optional-read and atomic-publication rules can make results traceable, failures localizable and local corruption incapable of producing false success, under fixed computation and disk contracts. Reducing ambiguity in research evidence is the intended value. Novelty and external generalizability remain unproven.

## 2. Constraints and responsibility

- Enmao owns scope, resources and research direction. The Agent implements, searches, organizes evidence and prepares PRs. CI checks engineering contracts; it does not replace research judgment.
- This delivery is the Chinese plan, this complete English counterpart, literature evidence and applicable standards, integrated through a work PR into dev. Study protocols are preserved in full in Chapter XI; historical results remain in their Studies and are not relabeled as preregistrations of later lessons.
- Allowed work is read-only literature/code review, synthetic CPU correctness regression, necessary CI and authorized cleanup of ended remote branches. [layout.md](../code/layout.md) governs paths and persistent artifacts.
- This round excludes GPU/long training, speed or duration studies, the historical 480s investigation, new directories or result formats, shared-path/logger migration, main merges and tags. Four tasks share one 24GB GPU; future scientific runs need a separate resource arrangement. This plan grants no device use.

# II. Literature review

Use the [Research Plan Standard](standards/research_plan_standard_en.md), [template](standards/research_plan_template_en.md) and the byte-preserved [Chinese Literature Review Standard](<standards/文献调研规范 Literature Review Standard.md>). [literature.md](literature.md) records the repository adaptation, searches and actual reading. Copying standards does not complete a systematic review.

- Information cutoff and actual search date: 2026-10-16. This is a question-driven preliminary review, not an exhaustive or systematic review.
- Concept groups: ML reproducibility / experiment provenance; failure evidence / artifact integrity; workflow / optional metadata / atomic publication. Prefer original paper pages, public author versions and versioned official documentation.
- Include sources directly explaining reproducibility evidence and design, implementation, evaluation or documentation factors, and identifiable workflow-method sources. Exclude unrelated model leaderboards, secondary summaries and marketing-only material; consolidate versions.
- Outstanding work includes cross-machine artifact reuse, external workflows under the same fault protocol, newer literature and citation tracing. Failure to find a source does not prove a research gap.

| **Source and location** | **Question and reading depth** | **Data and models** | **Algorithm** | **Metrics and conditions** | **Use and limitation** |
|---|---|---|---|---|---|
| [L1 Pineau et al., JMLR 2021](literature.md#l1) | Official abstract; components of a reproducibility program | Research process; no borrowed model results | Code, reproduction challenge and reporting checks | No reproduction of its program survey | Reference for separating reporting and verification; no claim of external RPipe reproduction |
| [L2 Gundersen et al., arXiv v2 2023](literature.md#l2) | Targeted abstract and sections on design, implementation, evaluation and documentation | Classification of reproducibility factors | Links design choices to conclusions | No transfer of cross-paper numbers | Distinguishes execution, analysis, conclusions and scope; not an RPipe measurement |
| [L3 Snakemake, 2021 v1](literature.md#l3) | Preliminary official search excerpt; direct reading restricted | Workflow system; configuration not reproduced | Candidate traceable-workflow approach | No same-fault/same-configuration comparison | Candidate external baseline; no capability or performance ranking |
| [L4 PyTorch 2.9 documentation](literature.md#l4) | Reproducibility page | Platform, runtime and RNG conditions | Randomness and determinism controls | Dependencies remain fixed by each Study | A seed alone does not prove equivalence across versions/platforms; no precision change here |

The decision is to continue, narrowed to verifiable contracts. Real counterexamples justify engineering failure-mechanism work; literature motivates evidence boundaries but does not establish superiority over other systems. Cross-project sharing [#89](https://github.com/DreamSoul-AI/RPipe/issues/89) and monitoring/logger [#90](https://github.com/DreamSoul-AI/RPipe/issues/90) remain independent explorations, not prerequisites.

# III. Research questions and hypotheses

| **ID** | **Question and hypothesis** | **Supporting observation** | **Counterexample or insufficient evidence** |
|---|---|---|---|
| RQ1 / H1 | Do unified existing freeze-rejection and process-failure decisions preserve protected successful evidence byte for byte? | Protected result/peer hashes remain unchanged and the original error is available in declared cases | Any changed protected file refutes that case; do not extend preservation to ordinary execute failures |
| RQ2 / H2 | Can existing safe-read, type and reference-ownership rules prevent local bad input from hiding peers or producing false success? | Optional bad input degrades by its contract, strict prerequisites reject, and out-of-bound compare rejects without replacing earlier output | Missing rows, invented metrics, crossed references or false pass; absence of an exception alone is insufficient |
| RQ3 / H3 | Does the existing atomic entry point preserve complete old/new bytes and cleanup ownership for replaceable text? | Controlled failures/interleavings produce only complete valid versions and rejection preserves peers | Torn text, collision overwrite or cleanup of another call's staging file |
| RQ4 / H4 | Can provenance-controlled historical-source comparisons test whether the template preserves selected computation? | A future complete comparison at the current baseline passes the original Study gates under aligned conditions | Historical success cannot validate current dev; missing raw data, weights or conditions means insufficient evidence |

The six planning dimensions connect as follows: literature bounds claims; synthetic inputs and existing Studies define data; state relations and neural networks provide the system model and scientific workload respectively; shared contracts alter execution/evidence algorithms; metrics check invariants and registered numerical gates; paired implementations and fault injection separate system changes from algorithm changes.

The intended contribution is understandable, regression-tested engineering contracts and evidence. There is no current claim of a new ML method, universal reduction in error rate, statistical significance, independent external reproduction or performance advantage.

# IV. Data

| **Collection** | **Origin and unit** | **Version, partition and quality checks** | **Location and limit** |
|---|---|---|---|
| D1 Contract inputs | Self-created JSON/YAML/text, paths/aliases and tiny CPU tensors; a fault scenario and valid peer are the units | Distinguish development cases, valid controls and later boundaries by fault family; retain red/green test bytes | `tests/_data/` and mirrored tests; runtime outputs in `.tmp/` are not cloned |
| D2 Short comparison evidence | Paired historical/current-chain Runs in main_probe; dataset × model pair | Original precision, seed, inputs, parameters, RNG and scheduling conditions; new source requires a new version and original gates | [Plan](<Research Plan RPipe.md#study-main-probe>), [report](../../studies/main_probe/docs/STUDY_REPORT.md); read-only here |
| D3 Long-curve evidence | Cross-seed main_exp curves and each Run's own best evaluation | Preserve original training, interruption and new-identity records; do not relabel old provenance | [Plan](<Research Plan RPipe.md#study-main-exp>), [report](../../studies/main_exp/docs/STUDY_REPORT.md); raw artifacts are not cloned |
| D4 Literature evidence | Original paper pages, official documentation and actual notes; an identified source version is a unit | Deduplicate by DOI/arXiv/version; separate author claims, local measurements and this plan's judgment | [Evidence ledger](literature.md); no invented local PDF or full-text reading |

No human-subject or private-data collection is added. Future MNIST/CIFAR10 work must check origin, version, splits and license/distribution terms in the original Study. Licenses were not reverified and data were not downloaded in this round. Immutable shared inputs remain at project-root `shared/{data,model}`; model objects, checkpoints, sampler/RNG and mutable caches belong to each Run.

Fault cases are not IID training/test samples or an estimate of real-world failure frequencies. Later cases must identify exposure during development; success after debugging the same cases is not evidence of generalization to unknown defects. Scientific train/development/test separation and leakage checks remain in each Study, without repartitioning historical data.

# V. Models

The system model is the relationship between Study/Experiment/Run states and evidence. Study declarations generate index and Run config; execution writes result and assets; process generates derived summaries; status/logs/report/compare read existing evidence; human reports form conclusions. This is an analytical view, not a new disk field or class.

Scientific workloads retain the linear, MLP, CNN and ResNet18 recipes in main_probe/main_exp. Initialization, weights, parameter structure, preprocessing and licensing follow their archived sources; no new architecture or invented parameter count is introduced. Data/model/algorithm/precision/budget stay fixed; only system boundaries supported by failing contracts vary.

Baselines comprise before/after implementations on identical fault inputs, minimal direct file operations or earlier local decisions, and scientific pairs against fixed historical RPipe source. External workflows remain pending literature/protocol baselines; neither functional equivalence nor comparability is assumed.

# VI. Algorithm

1. Form a bounded question from literature and real failure, recording input, expectation, observation and source; commit a failing regression first.
2. Reuse shared decisions, reads or atomic publication inside existing functions. Preserve original exception classes/propagation and avoid large refactors without failure evidence.
3. Compare before/after on identical inputs, checking protected bytes, peers, state, reference ownership and numerical meaning. Running successfully is not the contract.
4. Run targeted and necessary CPU flow checks, then integrate through a PR into dev. Accept actual CI separately for the final work commit and actual dev. Check and clean ended remote branches under authorization.
5. If scientific comparisons are authorized later, retain `prepare → execute → collect → summarize → write → process` and complete success-critical gates before finalization. Derived process failure must not rewrite a successful result.

Implementation reuses `src/rpipe/`. The private evidence-preservation decision does not broaden protection of old success. Optional-read fallback does not apply to strict index/process prerequisites or training-checkpoint resume. Atomic replacement concerns replaceable text; append-only logs, diagnostic checkpoint staging and locks retain the exemptions in the [approved design](../development/artifact_consistency_design.md). No attempt directory or change to result.json location, format or meaning is introduced.

Separate official, third-party and in-house sources. The historical comparisons use this project's own archived RPipe source: `98648f3` for the short comparison and `4ccb28d` for the long recipe. They reproduce project implementations, not algorithms or training in L1–L4. Exact archives and conditions remain in the Studies.

No loss function, hyperparameter search or performance algorithm is added. Deterministic contracts use prescribed inputs and barrier-controlled interleavings. Scientific budgets, seeds, RNG, sampler and precision stay fixed by the original plans. Stop a judgment batch on provenance drift, missing inputs, false success or changed protected evidence; retain the failure and correct it rather than relaxing thresholds.

# VII. Metrics

These are analysis definitions, not new result fields, CLI options or automatic schema metrics.

| **Role and metric** | **Definition, unit, direction** | **Data and aggregation** | **Implementation and condition** | **Criterion and basis** |
|---|---|---|---|---|
| Primary: evidence-preservation violations | Number of protected files whose bytes changed; lower is better | List files per applicable rejection/failure; exclude ordinary failures from protection | Byte/hash assertions, peers and original errors | Zero in enumerated cases; existing preservation contract |
| Primary: false passes | Cases that should reject an out-of-bound/mismatched comparison but pass | Pair each invalid case with valid controls | Public API/CLI; assert no external payload load where needed | Zero; exit code or overall pass rate is insufficient |
| Primary: presentation/ownership violations | Missing peers, fake metrics, broken rows/columns or crossed references | Separate TSV/Markdown/compare and strict/optional prerequisites | Structure, original keys/metrics, state and output-preservation assertions | Zero under declared contracts; no repair-writing of bad inputs |
| Primary: publication violations | Incomplete final text, collision overwrite or cleanup owned by another call | Report each explicit interleaving/fault point; no probability inference | Existing atomic entry point; logs/locks/checkpoints follow their protocols | Zero where applicable; no universal power-loss guarantee |
| Auxiliary: engineering execution status | Actual executed/pass/fail/skip/not-run status of necessary nodes | Separate version/platform/Python/case | JUnit, manifest and actual CI jobs | Required gates pass; skips are not passes and counts alone do not prove stability |
| Conditional scientific metrics | Parameters/inputs/RNG/scheduling equivalence, loss/correct counts, curves/model ordering | Original Study definitions; no mixed-version aggregation | [main_probe gates](<Research Plan RPipe.md#study-main-probe>), [main_exp gates](<Research Plan RPipe.md#study-main-exp>) | Retain original gates; an unexecuted current baseline is not accepted |

No throughput, latency, duration or cost rankings are measured here. Deterministic cases need no p-values. Controlled concurrency retains every repetition without estimating population accident probability. For future stochastic science, independent Runs/seeds are the unit; the Study specifies repetitions, sample size and analysis. Points from one trajectory are not independent repeats.

# VIII. Experimental design and execution order

| **Experiment and question** | **Method and control** | **Factors held fixed** | **Data, metrics, repetition** | **Budget and interpretation** |
|---|---|---|---|---|
| E0 Literature/provenance | Separate primary sources and repository facts | Actual versions and reading dates | D4; classify and map evidence before citation tracing | Preliminary review this round, not a completed systematic review |
| E1 Three contract families | Before/after on identical original bytes with valid peers | One boundary at a time; valid output/error contracts | D1/H1–H3; execute necessary cases, separate OS results | CPU allowed; red before fix, no selecting favorable repeats |
| E2 Native CPU flow/install | Current-source make/launch/process/readout/package install | Toy/Stub, tiny inputs, isolated basetemp, hidden GPU | Existing gates; final head and actual dev separately | Necessary local checks and existing CI; not scientific precision verification |
| E3 Numerical probe | Historical source vs current chain, original short recipe | Data, initialization, precision, RNG, evaluation and operation placement | D2; original complete matrix/gates, new version | **Not run this round**; requires future authorization, resources and inputs |
| E4 Long curves | Historical recipe vs current chain and original estimated-curve gates | Budget/seeds/full evaluation; preserve interrupted identities | D3; E1/E2 or short prefixes cannot substitute | **Not run this round**; no automatic long-training authorization |

## 1. Baseline reproduction and follow-up

Current reproduction work checks historical documents and committed evidence; it does not retrain or reproduce an external paper's training. The two formal Study reports at `v0.3.0 / 804b3823` record their historical acceptance. Subsequent dev changes mean those results cannot be relabeled as current-baseline results. Before future E3/E4, check archives, raw inputs/splits, environment and weight sources, then retain original entry gates and stop conditions. Resource shortages do not make a shortened run a complete reproduction.

| **Target and condition** | **Planned target and basis** | **Existing results/source** | **New current-dev measurement** | **Judgment** |
|---|---|---|---|---|
| Historical short computation pairs | Original complete matrix and segment-wise numerical gates | [main_probe report](../../studies/main_probe/docs/STUDY_REPORT.md) only | Not run | Interpretable only at its recorded version; no current-equivalence claim |
| Historical long curves/own best | Original full budget/four seeds/curve gates | [main_exp report](../../studies/main_exp/docs/STUDY_REPORT.md) only | Not run | Old plots or estimated agreement do not imply bitwise identity with original weights |
| Current engineering contracts | Enumerated byte/state/reference/publication invariants | [Development record](../development/record.md), red/green commits | Existing engineering regressions, bounded in Chapter IX | Supports tested contracts, not all research correctness |

## 2. Ablations and components

Quantization, scaling, rotation and model-training ablations are inapplicable here. H1–H3 compare earlier local behavior with unified rules while replaying fixed faults, inputs, operations and observation points. Do not remove established protections from real Studies for an ablation. Isolated copies/test inputs write to `.tmp/`, retaining original failures.

## 3. Schedule and stopping rules

Proceed E0 → E1 for real established gaps → E2. E3/E4 are separate conditional work. A documentation batch needs document/source/link/layout consistency checks and required CI, not retraining to fill a template. CPU cases use existing C1/C2 categories and independent temporary directories, without new timing or resource calibration.

Threats include development-exposed faults, synthetic files unlike all filesystem failures, one implementation team unlike independent reproduction, incompatible platform/precision/dependency conditions, and ignored historical assets unavailable in a clone. GPU contention may have influenced old timings, but this round neither tests that inference nor declares the historical cause resolved.

DOE/OR/BO/Bayesian experimental design are not used; contract repair needs no search framework. Any future configuration/resource optimization requires a separate objective, budget, search/final-evaluation separation and authorization, not an H1–H4 acceptance claim.

# IX. Execution record and conclusions

This chapter separates existing facts from new execution of this plan; it does not turn post-fix lessons into prior predictions. Formal scientific numbers, figures, commands and conclusions remain only in their Study.

| **Record** | **Evidence** | **Status** | **Supported and unsupported claims** |
|---|---|---|---|
| Two historical scientific Studies | Chapter XI protocols, Study reports and formal JSON | Historically accepted; current baseline not rerun | Only their recorded provenance and conditions, not current dev |
| Three correctness families | Approved design and linked red/green records | Existing repairs/regressions integrated into dev | Tested invariants, not absence of all bugs or scientific equivalence |
| PR98 / B086 | Record §§111–114; 56 corrected regression nodes | Final-head and actual-dev CI accepted | Text-cell structure for its enumerated cases |
| PR100 / B087 | Record §§113–115 and acceptance evidence; 24 nodes | Actual dev `187bb2e` accepted; ended remote deleted | Optional tracker decode boundary, not training-resume fallback |
| PR102 / B088 | Record §116 and acceptance evidence; 17 nodes, compare 56 | Actual dev `ef608468` accepted; ended remote deleted | Checkpoint ownership and comparison label structure, not unrestricted deserialization safety |
| This plan and review | This file, Chinese counterpart and literature ledger | Documentation only; no new scientific runs | Search/read scope and planned tests, not external superiority |

Negative results, old failures and canceled/pending checks remain in the development record and local evidence; `.tmp/` is not cloned. Missing original weights, exact sampler/resume equivalence outside supported contracts and universal external reproduction remain unresolved or unverified. Do not convert historical success or skipped tests into current acceptance.

<a id="overnight-correctness"></a>
## 1. Overnight PRs, results and conclusions for human review

The five overnight repairs were accepted through dev `82205cbe7fc89a46853eb61c1734d123b7d153f0` (PR125); main remains released v0.4.1 / `00e70b1`. Each of the five final heads and its actual dev integration passed every real matrix and aggregate. Local skips are not passes. Under the latest2026-10-16 cleanup authorization, completed and accepted Issues are now closed; closure does not mean a release.

| **Merged PR / Issue** | **Established gap** | **Actual dev and independent CI** | **Local CPU scope** | **Conclusion and retained boundary** |
|---|---|---|---|---|
| [PR #117](https://github.com/DreamSoul-AI/RPipe/pull/117) / [Issue #116](https://github.com/DreamSoul-AI/RPipe/issues/116) | Optional JSON integer-limit/deep-recursion failures | `20027afb231b`; [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38815660139) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38815660146) | core 1432 passed / 1 platform skip; CPU Flow 189 passed | Optional hints/state/JSONL preserve valid peers/trajectories; strict loaders stay strict. |
| [PR #119](https://github.com/DreamSoul-AI/RPipe/pull/119) / [Issue #118](https://github.com/DreamSoul-AI/RPipe/issues/118) | Conservative scheduling queries and implemented-design status | `4d12719d7eba`; [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38817527778) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38817527562) | core 1452 passed / 1 platform skip; CPU Flow 189 passed | Bad results do not imply success; unusable cached plans request make; approved design is implemented. |
| [PR #121](https://github.com/DreamSoul-AI/RPipe/pull/121) / [Issue #120](https://github.com/DreamSoul-AI/RPipe/issues/120) | Optional state metadata I/O boundary | `039db15f4df4`; [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38819287687) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38819287560) | core 1482 passed / 1 platform skip; CPU Flow 189 passed | Metadata refusal cannot hide valid JSONL; ordinary RuntimeError still propagates. |
| [PR #123](https://github.com/DreamSoul-AI/RPipe/pull/123) / [Issue #122](https://github.com/DreamSoul-AI/RPipe/issues/122) | Integer tensor precision with nonzero tolerance | `17460b1b579d`; [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38850576040) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38850576054) | core 1507 passed / 1 platform skip; CPU Flow 189 passed | int64 near 2**60 and uint64 no longer falsely pass after float64 truncation; bool/exact-zero behavior remains. |
| [PR #125](https://github.com/DreamSoul-AI/RPipe/pull/125) / [Issue #124](https://github.com/DreamSoul-AI/RPipe/issues/124) | Finite tolerance arithmetic overflow | `82205cbe7fc8`; [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38852496728) / [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38852496656) | core 1541 passed / 1 platform skip; CPU Flow 189 passed | Finite overflowing diff/bound and complex 0*inf=NaN no longer cause false passes; training precision is unchanged. |

Within #117–#125, #118/#120/#122/#124 identify Issues; #117/#119/#121/#123/#125 are repair PRs. The PR117 batch adds27 boundaries, PR119 adds20, PR121 expands two controls to32 instances (net30), PR123 adds25, and PR125 adds34. Their distinct red/control/green evidence is retained. PR125 first committed30 contracts with15 failures/15 controls; four additional cases replayed the frozen pre-fix comparator with2 failures/2 controls, explicitly not current-HEAD red. Final local scope was readout559/core1541 plus one existing POSIX skip/CPU Flow189. These are engineering checks, not new model results. Raw `.tmp/audit095`–`.tmp/audit099` materials are not supplied by clone; public PR/CI and [record §§132–145](../development/record.md) remain verifiable.

Earlier overnight integrations also matter: [PR103](https://github.com/DreamSoul-AI/RPipe/pull/103) established bilingual plans and original standards; [PR105](https://github.com/DreamSoul-AI/RPipe/pull/105) unified nonfinite identity; [PR107](https://github.com/DreamSoul-AI/RPipe/pull/107) cleans the writer's own read-only text staging; [PR109](https://github.com/DreamSoul-AI/RPipe/pull/109) retains the original shared-publication error; [PR112](https://github.com/DreamSoul-AI/RPipe/pull/112) guards count cells; [PR113](https://github.com/DreamSoul-AI/RPipe/pull/113) preserves test-failure evidence with unpaired Unicode; [PR115](https://github.com/DreamSoul-AI/RPipe/pull/115) isolates optional recursive YAML parsing failures. These use the approved three contracts; no attempt directories, new artifact formats, GPU mutex or logger replacement were introduced, and no long experiment was rerun. Reverted B065 performance deduplication stays reverted.

Historical scientific scope: main_probe's complete eight-cell60-step acceptance at v0.3.0 / `804b3823` passed, with maximum parameter difference0 and maximum train/test Loss difference4.44e-16. Subsequent shared-input CPU single/two-cell checks in its report are not a new complete matrix. main_exp at the same release passed eight-cell600-step preflight,32 continuous80000-step train Runs,32 own-best eval Runs and fixed curve gates; maximum endpoint-mean difference from the old figure estimates was0.254999 percentage points, not bitwise identity with all original historical weights. Neither complete scientific matrix was rerun at current dev. Formal numbers remain in the [probe report](../../studies/main_probe/docs/STUDY_REPORT.md) and [exp report](../../studies/main_exp/docs/STUDY_REPORT.md); original Runs/JSON/figures and failures are neither moved nor rewritten.

Later correctness update,2026-10-16: [PR128](https://github.com/DreamSoul-AI/RPipe/pull/128) merged normally into dev2008753; final-head CI and actual-dev [Unit38857214651](https://github.com/DreamSoul-AI/RPipe/actions/runs/38857214651)/[Package38857214660](https://github.com/DreamSoul-AI/RPipe/actions/runs/38857214660) passed all real matrices and aggregates. All138 B100 mixed-numeric contracts executed; local readout697/core1679 plus one existing platform skip/CPU Flow189 passed. Original red and frozen-old extreme-value replay are distinguished in [record §§148–150](../development/record.md). Issue127 and other completed Issues are closed under the latest cleanup authorization. Code remains dev-only; this is engineering acceptance, not a new scientific experiment, literature conclusion or main release.


Further review on 2026-10-16: [Issue130](https://github.com/DreamSoul-AI/RPipe/issues/130) / B101 reproduces startup logging bypassing failed-result publication and diagnostic logger initialization replacing a phase error. Red commit `8a351df` has 10 failed nodes, followed by design `77986d9` and minimal production fix `9e345c8`. Local adjacent checks passed 123 cases; core passed 1679 with one existing platform skip; CPU Flow passed 199. All 10 new nodes executed. PR131 was merged and accepted on actual dev `a458eae`; final-head and actual-dev CI all passed. Issue130 was closed and its local/remote work branch removed. There was no layout, logging-protocol, scientific-gate or success-preservation expansion. PR129 was accepted on actual dev `76f7b3b`; cleanup removed 37 local and 2 remote branches; see [record§151–153](../development/record.md).

Further follow-up: [Issue132](https://github.com/DreamSoul-AI/RPipe/issues/132) / B102 confirms that the native training finalizer can replace an original execution failure with a tracker-state write error. Red `1e8c5d1` has 20 failures and 2 successful-path controls; design `75e9885` precedes minimal production fix `384bf2c`. Existing failures and termination retain their identity; final write failures on an otherwise successful body still propagate. All 22 new nodes passed; core passed 1693 with one existing platform skip, and CPU Flow passed 207. Supplement `a758de4` adds three actual staging-directory assertions: frozen old-function replay remains 20 failures / 2 controls, while current code passes 22. This is local validation awaiting final-head and actual-dev CI, with no changes to numerics, periodic flush, checkpoint/RNG, schemas or layout and no new formal experiment. See [record§155–156](../development/record.md).

<a id="gaps-next"></a>
## 2. Open gaps, tonight's review and next experiments

| **Priority / question** | **Next action and entry conditions** | **Fixed comparison, budget and interpretation** | **State / stop rule** |
|---|---|---|---|
| P0 Human review | Enmao reviews both languages, five complete Study sections, overnight repairs and remaining gaps; revise direction explicitly | One project plan; historical and current authorization separated | Documentation only; not new experimental acceptance |
| P1 E0 Literature | Obtain readable L3/original workflow semantics, then targeted citation tracing; register versions, actual reading and screening | Establish the same-fault/same-semantics protocol before any external comparison | Existing L1 abstract/L2 targeted sections/L3 excerpts/L4 full page retained; this consolidation adds no online search or new full-text reading |
| P2 E1/E2 Correctness | Only real reproducible counterexamples; Issue/red contract, minimal shared repair, local CPU plus final-head/actual-dev CI | Three contracts, original/peer bytes, ordinary programming errors preserved, strict prerequisites not weakened; isolated basetemp | Fixed Issues wait for release; no invented change/empty PR,480s investigation or speed test |
| P3 E3 main_probe | Separate explicit GPU/new-run authorization; verify source archive, raw/Stats, environment, actual FP32 and provenance; new version/make | Eight combinations×seed0,60 continuous steps, full10000-sample test at30/60, each own Loss-best eval; original atol1e-6/rtol1e-5, Loss≤1e-6 and exact integers | Not run at current baseline; old v0.3.0 does not count. Missing cells/source or numerical failure stops without relaxing gates |
| P4 E4 main_exp | Only after P3 and its own eight-cell600-step/eval200 entry gate, plus separate full-budget/resource authorization; preserve old identities | Eight data/model combinations×four seeds,32 continuous80000-step train+32 own-best eval,400 full-test points; original TARGET/REFERENCE_CURVES gates | Not run; two200-step cells cannot unlock it. Exact custom-sampler resume remains unaccepted; interruption needs a new identity/start, not altered old evidence |
| Independent explorations | Cross-project shared [#89](https://github.com/DreamSoul-AI/RPipe/issues/89) and TensorBoard/legacy logger [#90](https://github.com/DreamSoul-AI/RPipe/issues/90) remain independent research | Do not bind them to repairs or migrate directories/replace logging first | Initial deliveries89/90 are complete and closed; implementation remains unauthorized, project-root inputs immutable and Run state independent |

Further unknowns are complete scientific validation at current source, recoverability of all original historical weights/seeds, custom-sampler exact recovery, cross-machine/independent external reproduction and systematic literature coverage. Engineering success cannot close them. Four tasks sharing a24GB GPU is a known constraint only: this round neither requests nor uses nor interrupts GPU work. Future scheduling needs authorization and actual co-running resource checks; historical throughput/cache observations are not fresh cold-start or speed evidence.

# X. Artifacts, review and change history

The sole macro-plan authority is [研究计划 RPipe.md](<研究计划 RPipe.md>), with this English counterpart and [literature.md](literature.md). Do not duplicate it at `docs/research-plan.md`. Current, template and historical Study plans are consolidated in full in Chapter XI; no parallel Study PLAN.md remains. Scientific figures and formal results stay in their original Study; source standards stay in [standards/](standards/README.md).

Link each implementation batch to its RQ, real failing case, regression, PR and actual-dev acceptance. Under the latest2026-10-16 authorization, completed and accepted Issues close without waiting for a release; research89/90 close only their delivered preliminary reports, not future implementation. Delete merged local/remote branches after preservation and actual CI checks, retain unmerged/GPU-active branches and worktrees, and perform a full cleanup approximately every five merged PRs. This round does not merge main, tag, delete data or trigger experiments.

| **Date** | **Change** | **Reason and impact** |
|---|---|---|
| 2026-10-16 | Establish the template-based Chinese macro plan, full English counterpart and literature evidence/standard | Connect RQs, six dimensions, evidence levels and execution status; leave historical Study plans/results, runtime code and authorization unchanged |

Source provenance and actual reading limits are in [literature.md](literature.md). The two research-plan masters retain the bytes recorded in [standards/README](standards/README.md).

<a id="study-plan-inventory"></a>
# XI. Complete Study protocols and historical scopes

Consolidated under Enmao's2026-10-16 12:53 instruction: all five source documents are included completely, with matching English sections. Chapters VIII/IX govern current direction and authorization; this chapter retains historical protocols, restrictions and template content, not a GPU queue for tonight. Each historical copy differs from its current source only by the later2026-10-13 authorization paragraph and is still retained in full. Original bytes remain verifiable at the pre-relocation Git commit. Machine execution/capacity/recovery JSON is evidence, not another research plan, and stays byte-for-byte in place. Four broken relative links in the old snapshots are repaired to existing targets. The historical recipe link explicitly uses pre-relocation dev as an available code reference, not a substitute for a historical Run's frozen source.

<a id="study-main-probe"></a>
## 1. main_probe: complete short-probe protocol

The complete protocol follows. Its dates, old authorization, calibration and scheduling text are historical scope, not current GPU/timing/training authorization. Commands and code paths retain the original Study working directory; clickable links are relocated.

[Original before relocation](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_probe/docs/PLAN.md) · SHA-256 `f8b389ae118a16dc6787945aaf13809549f1bac42fcf333ef470a152f669abbf`.

<!-- BEGIN study-main-probe -->
<a id="study-main-probe--main_probe-计划"></a>
### main_probe plan

2026-10-13 post-release authorization: prepare, validate, calibrate and execute all eight paired combinations locally at fixed v0.3.0 / `804b3823`; see [execution record](../../studies/main_probe/docs/v030_execution.md). The short-run entries below retain their historical scope. Recipes, budgets and acceptance gates are unchanged.

<a id="study-main-probe--一目标"></a>
#### I. Objective

On the same device and under the same deterministic conditions, compare the original code at fixed main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` against the current RPipe chain over 60 steps, segment by segment, to check whether the rewrite changed numerical behavior. This is a short-budget probe, not a substitute for [main_exp](../../studies/main_exp/README.md) long-curve reproduction.

<a id="study-main-probe--二配方"></a>
#### II. Recipe

MNIST / CIFAR10 × linear / mlp / cnn / resnet18, seed0; complete train/test, batch250 / test1000, 60 continuous optimizer steps, complete test at steps30/60; SGD lr0.1, momentum0.9, Nesterov, wd0.0005, no gradient clipping, cosine T_max60; current no-BN CNN, native Factory, and Kornia augmentation before the model. Recompute statistics over the entire train set with archived original Stats(dim1), in sequential batch250 order. Both sides use deterministic=true, benchmark=false and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Version03 on 2026-10-11 additionally sets cudnn_allow_tf32=false and matmul_allow_tf32=false. Both sides share the precision declaration; recipe rejects undeclared precision. Thresholds do not change. Version02 evidence remains in its [version report](../../studies/main_probe/docs/versions/main-probe-flow-20261010-02/STUDY_REPORT.md); it cannot replace an eight-cell run for new source/precision. The reason is recorded in main_exp's [device numerical diagnosis](../../studies/main_exp/docs/EVAL_DIAGNOSTIC.json).

<a id="study-main-probe--三统一-flow-入口"></a>
#### III. Unified Flow entry

Dedicated computation and registration live in `structure/algorithm/`; six-stage implementations live in `flow/<phase>/run.py`. Each `__init__.py` only exports. The collect comparison helper is `flow/collect/comparison.py`, data preparation is `flow/prepare/data.py`; recipe only validates and registers dedicated sources. Both Python source pillars enter provenance automatically, without a handwritten source include list.

`rpipe make / launch studies/main_probe` expands eight paired Runs. Each computes only its data/model combination, including training on both sides and independent evaluation of each side's best. Algorithm source is `main_probe`; recipe registers this Study's Data and paired Algorithm. CPU preparation runs through Study `prepare.before(ctx)` after library initialization and before recipe registration, so original full-train Stats exists before Data/Model construction. Workspace is this Run's `assets/probe/`; mutable original config is not shared.

- prepare: archive fixed original source, copy this dataset's raw input, recompute Stats, check matching CPU initialization and forward output, then restore this Run's runtime seed. Ordinary make expands declarations without executing probe preparation.
- execute: the paired Algorithm runs original first, then current using Data/Model/System/Tracker created by library prepare. Restore initialization RNG recorded during preparation, retaining original-first/current-second order and seed0. A Run handles one combination, without a separate script or full-matrix launch.
- collect: read actual inputs, RNG, parameters, optimizer, scheduler, complete sample counts and independent evaluation comparisons from execute's OBSERVATIONS.pt. summarize combines reports and evidence paths, rejects failed numerical gates and retains failures.
- write: before library result finalization, project the existing observations from both sides, call library compare and write the complete comparison report. A failed extra gate writes failed; success lets the library register the complete asset inventory and finalize. No additional training is introduced.
- process: Run scope checks this combination; Study scope summarizes only the eight groups in the current index. Missing, duplicate, failed or different-source groups cannot count as complete success.

Failures retain original evidence and let Flow write failed status. Checkpoint resume is rejected; retry needs a new Run directory and cannot overwrite earlier probe workspace.

<a id="study-main-probe--四门限"></a>
#### IV. Thresholds

At steps30/60, floating parameters and SGD momentum use atol1e-6 / rtol1e-5; integer buffers and optimizer configuration match exactly. Train/test Loss absolute difference ≤1e-6 and correct-sample counts match. Initial parameters, CPU/CUDA RNG, training sample indices, actual inputs and segment schedulers match. Both select best by test Loss, with matching selection and weights. Independent eval correct counts match and Loss difference ≤1e-6. Only all eight combinations passing with unchanged provenance constitute complete acceptance.

<a id="study-main-probe--五验证边界"></a>
#### V. Validation boundaries

<a id="study-main-probe--2026-10-13-发布前完整短探针"></a>
##### 2026-10-13 complete pre-release short probe

The user authorized continued defect repair and release preparation. This run executes all eight 60-step combinations, not main_exp long training. First preserve every obtainable version05 declaration, index/provenance/process and report byte for byte; old Runs are retained. Use main-probe-flow-20261013-06, remake and launch serially. Complete raw caches have been checked, source and actual FP32 conditions remain fixed, and original gates are not relaxed. Only all eight groups permit Study acceptance.

CPU contract gates use synthetic computation and tiny tensors to verify stage responsibilities, normal finalization, numerical failure and library compare failure; they do not prove scientific numerical equivalence. Formal reruns use a new version/Run ID and retain current baseline, data, device and failures. Earlier numerical evidence cannot replace the new version's acceptance. Historical authorization and the current limited scope are separate and cannot automatically restart the full matrix.

<a id="study-main-probe--2026-10-12-本轮短实验"></a>
##### 2026-10-12 short run

Run only the MNIST/linear and MNIST/cnn seed0 pairs of main-probe-flow-20261012-05. Retain 60-step, complete test, FP32 and original numerical gates; budgets and thresholds do not change. Preserve obtainable version03 reports/declarations before make. This workspace lacks old index/provenance/process and does not reconstruct history. Version04 first failed prepare because evaluate was missing; retain its Run/snapshot, repair isolated dependencies and use new version05. The second planned combination was CIFAR10/cnn, but raw input was absent and downloading was slow; after MNIST validation, use MNIST/cnn to bound this run's cost. CIFAR10 is outside this measurement. Execute the two new Runs sequentially with `rpipe run-one`, without launching the other six. Explicit process must then classify the subset incomplete: expected rejection, not eight-group acceptance. Raw evidence remains in new Runs; segment values and conclusions belong in the Study report.
<!-- END study-main-probe -->

<a id="study-main-exp"></a>
## 2. main_exp: complete long-curve protocol

The complete protocol follows. Its dates, old authorization, calibration and scheduling text are historical scope, not current GPU/timing/training authorization. Commands and code paths retain the original Study working directory; clickable links are relocated.

[Original before relocation](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_exp/docs/PLAN.md) · SHA-256 `6ba2017b66ac2a422fd5753da284296042fe036a69f136e56d17270517c9800a`.

<!-- BEGIN study-main-exp -->
<a id="study-main-exp--main_exp-计划"></a>
### main_exp plan

2026-10-13 post-release authorization: prepare, validate and calibrate locally at fixed v0.3.0 / `804b3823`, then execute the full matrix only after its entry gate; see [execution record](../../studies/main_exp/docs/v030_execution.md). Earlier short-run restrictions below retain their release-preparation context. Recipes, budgets and thresholds remain unchanged.

<a id="study-main-exp--一目标"></a>
#### I. Objective

Use the recipe at historical main `4ccb28d0496110253e9f8e3f3df658853f07996b` on the current RPipe training loop to reproduce 80000-step curves for MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seed0–3, and compare with the two original main README figures. Original run records were not recovered. Separately assess complete recipe execution, same-environment numerical equivalence of original/current computation, and four-seed mean curves against the figures.

<a id="study-main-exp--二配方"></a>
#### II. Recipe

1. All training data and the complete 10000-image test set. Data, models, the old CNN's four BN layers, CPU torchvision augmentation, constant statistics and sampling order directly reuse archived source, registered by [recipe.py](../../studies/main_exp/recipe.py) as `historical_4ccb28d`.
2. Continuous 80000 optimizer steps, complete test every200 steps, 400 curve points. Train/test batch250; SGD lr0.01, momentum0.9, Nesterov, wd0.0005, clip1, cosine T_max80000; see [experiment_config.yaml](../../studies/main_exp/experiment_config.yaml).
3. Best is selected by RPipe's test Accuracy over the whole run. The archived best-comparison bug is not ported. Curves use in-training test history.
4. Deterministic conditions: deterministic=true, benchmark=false, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, CPU2 threads. New version03 on 2026-10-11 explicitly sets cudnn_allow_tf32=false and matmul_allow_tf32=false for both preflight sides, training and independent eval. Version02 lacked explicit precision and failed independent-eval consistency; complete evidence is retained. Do not mix precision baselines. Changes and restart conditions are in [eval_recovery.md](../../studies/main_exp/docs/eval_recovery.md).
5. Training must be continuous. Generic native resume locally covers RNG/data position under a single-process contract, but this Study's custom sampler/data adapter has not passed exact-resume acceptance. Continue rejecting interrupted training resume; start from scratch under a new version.

<a id="study-main-exp--三门限"></a>
#### III. Gates

1. Preflight: all eight same-device original/current 600-step / eval200 pairs must pass before launch. Parameters use atol1e-6 / rtol1e-5; integer buffers match; Loss difference ≤1e-6, correct counts, scheduler, RNG and actual inputs match. Scheduler and sampling retain total budget80000.
2. Final: all32 train and32 eval succeed, with all four seeds aligned at every evaluation point. Gates fixed in [TARGET.md](../../studies/main_exp/docs/TARGET.md) and [REFERENCE_CURVES.json](../../studies/main_exp/docs/REFERENCE_CURVES.json): endpoint mean and last50-point mean within0.25 percentage points for MNIST /1.00 for CIFAR10 of original figure estimates; seven-anchor MAE≤0.25/1.00, max deviation≤0.60/2.00; final-segment model order and temporal-variation gates. Never change thresholds after observing results.
3. Figure-reading error and original seeds are unknown; conclusions are limited to these fixed estimation gates.

<a id="study-main-exp--四执行"></a>
#### IV. Execution

<a id="study-main-exp--2026-10-13-校准与精度复验"></a>
##### 2026-10-13 capacity and precision recheck

First use fixed source snapshots for CIFAR10 CNN/ResNet18/mixed-task capacity measurements, three seeds each, 800 steps, round1/2/3. Then use new short CNN best weights for16 explicit FP32 and16 TF32 full-test evaluations, covering serial/concurrent execution; preserve per-sample predictions, input/model hashes and obtainable kernel traces. New weights do not stand in for missing old B-023 76800-step weights; old counterexamples/JSON are not overwritten.

Capacity uses existing system.config.vram_bytes; library heuristics and safety=0.5 remain unchanged. Subsequent fresh make for main_exp uses4.5GiB per process, conservatively covering recipe models. This limits automatic concurrency for light jobs and does not generalize to other devices/recipes. A new eight-cell600-step preflight is still required. This phase executes entry gates and short calibration only, not the full matrix, and preserves old version03.

Commands are in the [Study entry](../../studies/main_exp/README.md). `rpipe launch` packs same-wave concurrency by VRAM, optionally `--round N`; all train jobs finish before eval. The report is `docs/STUDY_REPORT.md`; `rpipe report` generates the number table.

Dedicated Data/Model and archive support live in `structure/`; `flow/prepare/run.py` checks provenance and actual runtime conditions, and `flow/process/run.py` calls `curves.py` only at Study scope. `__init__.py` only exports. Raw preparation uses `flow/prepare_data.py`; recipe calls `flow/execute/speed.py` for calibration. Manual tools are not automatically hooks; both source pillars freeze automatically.

<a id="study-main-exp--2026-10-12-本轮短实验"></a>
##### 2026-10-12 short run

Only existing `recipe.preflight` runs a same-device original/current MNIST/linear seed0 prefix,200 steps with complete test, retaining sampler/scheduler total budget80000, batch250, original precision and gates. Validation identity main-exp-flow-smoke-20261012-01 writes a new directory under this Study's `shared/validation/`, with segment values in `docs/`. This is not the formal64-Run experiment; it changes no version03 declaration/index/provenance/Run/PREFLIGHT.json and does not trigger formal Flow launch. Existing loader/scope contracts are checked separately. A200-step single cell is not the eight-cell600-step entry gate and cannot unlock long training. A formal rerun still requires new version, make and the full entry gate.
<!-- END study-main-exp -->

<a id="study-template"></a>
## 3. Skeleton for a new Study section

The complete protocol follows. Its dates, old authorization, calibration and scheduling text are historical scope, not current GPU/timing/training authorization. Commands and code paths retain the original Study working directory; clickable links are relocated.

[Original before relocation](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/_template/docs/PLAN.md) · SHA-256 `cf5e709413dfc15136af1f2c8e0438ff8dad23bae8dd2fb1a769ef6da4f76925`.

<!-- BEGIN study-template -->
<a id="study-template--study-plan-my_study"></a>
### Study plan skeleton: my_study

Use the [research requirements](../development/research.md) to specify this round's question, hypothesis, literature/data/model/algorithm/metrics/design. Reference an established macro-plan; explain inapplicable dimensions. This is an execution skeleton, not a complete research design. Scheduling guidance is the [root Study guide §3](../../README.md#3-高效率实验设计); record results in the Study's `STUDY_REPORT.md`.

<a id="study-template--1-研究问题"></a>
#### 1. Research question

What does this round compare, and what observation constitutes success?

<a id="study-template--2-study--experiment--run"></a>
#### 2. Study / Experiment / Run

- Study: `my_study`.
- Experiment: expanded from axes, excluding seed.
- Run: each Experiment × seeds.

<a id="study-template--3-变量轴"></a>
#### 3. Variable axes

| **Axis** | **Field** | **Values** |
|---|---|---|
| Factor | To specify | To specify |

<a id="study-template--4-固定条件"></a>
#### 4. Fixed conditions

| **Item** | **Value** |
|---|---|
| seeds | To specify |
| system.device | cuda |

<a id="study-template--5-高效率排班3"></a>
#### 5. Scheduling (§3; historical skeleton, not current resource authorization)

- Group similar runs: same model.name and similar duration; avoid putting resnet and linear into one wait.
- Wait gate: each wave waits for all processes to exit and memory to release before the next; otherwise waves overlap and can OOM.
- GPU packing (the old skeleton sought to fill available GPU capacity; this grants no device allocation now): `python -m rpipe make studies/<name> --num-gpus 1` defaults to `--round auto`; inspect make's `pack N waits`, which launch does not reprint. The conservative wall estimate sums the slowest Run in each wave.
- Dependency: when algorithm.mode=eval exists, wait for all train jobs before eval.
- Error: record run_id; after the wave, retry not-succeeded cells with resume:latest according to the applicable recovery contract. Only successful training unlocks dependent eval; failed dependencies do not become success.
- Seed is not concurrency.

```bash
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
```

<a id="study-template--51-时长预估"></a>
##### 5.1 Duration estimate

Fill after make and before launch. Each row estimates one Run in seconds. A wait wave takes its slowest Run; the whole round sums waves. This does not estimate VRAM.

| **wait** | **factors** | **mode** | **seed** | **id** | **est** |
|---|---|---|---|---|---|
| To specify | To specify | To specify | To specify | To specify | To specify |
| | Whole round | | | | To specify |

<a id="study-template--6-成功标准"></a>
#### 6. Success criteria

- State numerical, quality or other evidence criteria for the research question; execution/artifact presence alone cannot prove research quality.
- All planned Runs have status:succeeded.
- docs/figures/learning_curves.png exists.
- STUDY_REPORT.md embeds the plot and records actual pack scheduling.

<a id="study-template--7-执行范围与验证边界"></a>
#### 7. Execution scope and validation boundary

State authorized scale, resources/budget and uncovered scenarios. Explain retained failures/repeats and separate short-probe and long-experiment acceptance conditions.
<!-- END study-template -->

<a id="archive-main-probe-v030"></a>
## 4. main_probe: complete pre-local-v0.3.0 historical plan

The complete protocol follows. Its dates, old authorization, calibration and scheduling text are historical scope, not current GPU/timing/training authorization. Commands and code paths retain the original Study working directory; clickable links are relocated.

[Original before relocation](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_probe/docs/versions/v030-before-local-20261013/docs/PLAN.md) · SHA-256 `49675c7d7fbcc82e918d1e15b976aecb03569a4037ad3eec7e99dc07634a28e3`.

<!-- BEGIN archive-main-probe-v030 -->
<a id="archive-main-probe-v030--main_probe-计划"></a>
### main_probe plan

<a id="archive-main-probe-v030--一目标"></a>
#### I. Objective

On the same device and under the same deterministic conditions, compare the original code at fixed main `98648f3a5c7db7dccf3ca806410d5b6fdee9484c` against the current RPipe chain over 60 steps, segment by segment, to check whether the rewrite changed numerical behavior. This is a short-budget probe, not a substitute for [main_exp](../../studies/main_exp/README.md) long-curve reproduction.

<a id="archive-main-probe-v030--二配方"></a>
#### II. Recipe

MNIST / CIFAR10 × linear / mlp / cnn / resnet18, seed0; complete train/test, batch250 / test1000, 60 continuous optimizer steps, complete test at steps30/60; SGD lr0.1, momentum0.9, Nesterov, wd0.0005, no gradient clipping, cosine T_max60; current no-BN CNN, native Factory, and Kornia augmentation before the model. Recompute statistics over the entire train set with archived original Stats(dim1), in sequential batch250 order. Both sides use deterministic=true, benchmark=false and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Version03 on 2026-10-11 additionally sets cudnn_allow_tf32=false and matmul_allow_tf32=false. Both sides share the precision declaration; recipe rejects undeclared precision. Thresholds do not change. Version02 evidence remains in its [version report](../../studies/main_probe/docs/versions/main-probe-flow-20261010-02/STUDY_REPORT.md); it cannot replace an eight-cell run for new source/precision. The reason is recorded in main_exp's [device numerical diagnosis](../../studies/main_exp/docs/EVAL_DIAGNOSTIC.json).

<a id="archive-main-probe-v030--三统一-flow-入口"></a>
#### III. Unified Flow entry

Dedicated computation and registration live in `structure/algorithm/`; six-stage implementations live in `flow/<phase>/run.py`. Each `__init__.py` only exports. The collect comparison helper is `flow/collect/comparison.py`, data preparation is `flow/prepare/data.py`; recipe only validates and registers dedicated sources. Both Python source pillars enter provenance automatically, without a handwritten source include list.

`rpipe make / launch studies/main_probe` expands eight paired Runs. Each computes only its data/model combination, including training on both sides and independent evaluation of each side's best. Algorithm source is `main_probe`; recipe registers this Study's Data and paired Algorithm. CPU preparation runs through Study `prepare.before(ctx)` after library initialization and before recipe registration, so original full-train Stats exists before Data/Model construction. Workspace is this Run's `assets/probe/`; mutable original config is not shared.

- prepare: archive fixed original source, copy this dataset's raw input, recompute Stats, check matching CPU initialization and forward output, then restore this Run's runtime seed. Ordinary make expands declarations without executing probe preparation.
- execute: the paired Algorithm runs original first, then current using Data/Model/System/Tracker created by library prepare. Restore initialization RNG recorded during preparation, retaining original-first/current-second order and seed0. A Run handles one combination, without a separate script or full-matrix launch.
- collect: read actual inputs, RNG, parameters, optimizer, scheduler, complete sample counts and independent evaluation comparisons from execute's OBSERVATIONS.pt. summarize combines reports and evidence paths, rejects failed numerical gates and retains failures.
- write: before library result finalization, project the existing observations from both sides, call library compare and write the complete comparison report. A failed extra gate writes failed; success lets the library register the complete asset inventory and finalize. No additional training is introduced.
- process: Run scope checks this combination; Study scope summarizes only the eight groups in the current index. Missing, duplicate, failed or different-source groups cannot count as complete success.

Failures retain original evidence and let Flow write failed status. Checkpoint resume is rejected; retry needs a new Run directory and cannot overwrite earlier probe workspace.

<a id="archive-main-probe-v030--四门限"></a>
#### IV. Thresholds

At steps30/60, floating parameters and SGD momentum use atol1e-6 / rtol1e-5; integer buffers and optimizer configuration match exactly. Train/test Loss absolute difference ≤1e-6 and correct-sample counts match. Initial parameters, CPU/CUDA RNG, training sample indices, actual inputs and segment schedulers match. Both select best by test Loss, with matching selection and weights. Independent eval correct counts match and Loss difference ≤1e-6. Only all eight combinations passing with unchanged provenance constitute complete acceptance.

<a id="archive-main-probe-v030--五验证边界"></a>
#### V. Validation boundaries

<a id="archive-main-probe-v030--2026-10-13-发布前完整短探针"></a>
##### 2026-10-13 complete pre-release short probe

The user authorized continued defect repair and release preparation. This run executes all eight 60-step combinations, not main_exp long training. First preserve every obtainable version05 declaration, index/provenance/process and report byte for byte; old Runs are retained. Use main-probe-flow-20261013-06, remake and launch serially. Complete raw caches have been checked, source and actual FP32 conditions remain fixed, and original gates are not relaxed. Only all eight groups permit Study acceptance.

CPU contract gates use synthetic computation and tiny tensors to verify stage responsibilities, normal finalization, numerical failure and library compare failure; they do not prove scientific numerical equivalence. Formal reruns use a new version/Run ID and retain current baseline, data, device and failures. Earlier numerical evidence cannot replace the new version's acceptance. Historical authorization and the current limited scope are separate and cannot automatically restart the full matrix.

<a id="archive-main-probe-v030--2026-10-12-本轮短实验"></a>
##### 2026-10-12 short run

Run only the MNIST/linear and MNIST/cnn seed0 pairs of main-probe-flow-20261012-05. Retain 60-step, complete test, FP32 and original numerical gates; budgets and thresholds do not change. Preserve obtainable version03 reports/declarations before make. This workspace lacks old index/provenance/process and does not reconstruct history. Version04 first failed prepare because evaluate was missing; retain its Run/snapshot, repair isolated dependencies and use new version05. The second planned combination was CIFAR10/cnn, but raw input was absent and downloading was slow; after MNIST validation, use MNIST/cnn to bound this run's cost. CIFAR10 is outside this measurement. Execute the two new Runs sequentially with `rpipe run-one`, without launching the other six. Explicit process must then classify the subset incomplete: expected rejection, not eight-group acceptance. Raw evidence remains in new Runs; segment values and conclusions belong in the Study report.
<!-- END archive-main-probe-v030 -->

<a id="archive-main-exp-v030"></a>
## 5. main_exp: complete pre-local-v0.3.0 historical plan

The complete protocol follows. Its dates, old authorization, calibration and scheduling text are historical scope, not current GPU/timing/training authorization. Commands and code paths retain the original Study working directory; clickable links are relocated.

[Original before relocation](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_exp/docs/versions/v030-before-local-20261013/docs/PLAN.md) · SHA-256 `257f348f1151f24c902145d2d386506e295217da4c3131cb72c4096c1cea832c`.

<!-- BEGIN archive-main-exp-v030 -->
<a id="archive-main-exp-v030--main_exp-计划"></a>
### main_exp plan

<a id="archive-main-exp-v030--一目标"></a>
#### I. Objective

Use the recipe at historical main `4ccb28d0496110253e9f8e3f3df658853f07996b` on the current RPipe training loop to reproduce 80000-step curves for MNIST / CIFAR10 × linear / mlp / cnn / resnet18 × seed0–3, and compare with the two original main README figures. Original run records were not recovered. Separately assess complete recipe execution, same-environment numerical equivalence of original/current computation, and four-seed mean curves against the figures.

<a id="archive-main-exp-v030--二配方"></a>
#### II. Recipe

1. All training data and the complete 10000-image test set. Data, models, the old CNN's four BN layers, CPU torchvision augmentation, constant statistics and sampling order directly reuse archived source, registered by [recipe.py](https://github.com/DreamSoul-AI/RPipe/blob/82205cbe7fc89a46853eb61c1734d123b7d153f0/studies/main_exp/recipe.py) as `historical_4ccb28d`.
2. Continuous 80000 optimizer steps, complete test every200 steps, 400 curve points. Train/test batch250; SGD lr0.01, momentum0.9, Nesterov, wd0.0005, clip1, cosine T_max80000; see [experiment_config.yaml](../../studies/main_exp/docs/versions/v030-before-local-20261013/experiment_config.yaml).
3. Best is selected by RPipe's test Accuracy over the whole run. The archived best-comparison bug is not ported. Curves use in-training test history.
4. Deterministic conditions: deterministic=true, benchmark=false, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, CPU2 threads. New version03 on 2026-10-11 explicitly sets cudnn_allow_tf32=false and matmul_allow_tf32=false for both preflight sides, training and independent eval. Version02 lacked explicit precision and failed independent-eval consistency; complete evidence is retained. Do not mix precision baselines. Changes and restart conditions are in [eval_recovery.md](../../studies/main_exp/docs/versions/v030-before-local-20261013/docs/eval_recovery.md).
5. Training must be continuous. Generic native resume locally covers RNG/data position under a single-process contract, but this Study's custom sampler/data adapter has not passed exact-resume acceptance. Continue rejecting interrupted training resume; start from scratch under a new version.

<a id="archive-main-exp-v030--三门限"></a>
#### III. Gates

1. Preflight: all eight same-device original/current 600-step / eval200 pairs must pass before launch. Parameters use atol1e-6 / rtol1e-5; integer buffers match; Loss difference ≤1e-6, correct counts, scheduler, RNG and actual inputs match. Scheduler and sampling retain total budget80000.
2. Final: all32 train and32 eval succeed, with all four seeds aligned at every evaluation point. Gates fixed in [TARGET.md](../../studies/main_exp/docs/versions/v030-before-local-20261013/docs/TARGET.md) and [REFERENCE_CURVES.json](../../studies/main_exp/docs/versions/v030-before-local-20261013/docs/REFERENCE_CURVES.json): endpoint mean and last50-point mean within0.25 percentage points for MNIST /1.00 for CIFAR10 of original figure estimates; seven-anchor MAE≤0.25/1.00, max deviation≤0.60/2.00; final-segment model order and temporal-variation gates. Never change thresholds after observing results.
3. Figure-reading error and original seeds are unknown; conclusions are limited to these fixed estimation gates.

<a id="archive-main-exp-v030--四执行"></a>
#### IV. Execution

<a id="archive-main-exp-v030--2026-10-13-校准与精度复验"></a>
##### 2026-10-13 capacity and precision recheck

First use fixed source snapshots for CIFAR10 CNN/ResNet18/mixed-task capacity measurements, three seeds each, 800 steps, round1/2/3. Then use new short CNN best weights for16 explicit FP32 and16 TF32 full-test evaluations, covering serial/concurrent execution; preserve per-sample predictions, input/model hashes and obtainable kernel traces. New weights do not stand in for missing old B-023 76800-step weights; old counterexamples/JSON are not overwritten.

Capacity uses existing system.config.vram_bytes; library heuristics and safety=0.5 remain unchanged. Subsequent fresh make for main_exp uses4.5GiB per process, conservatively covering recipe models. This limits automatic concurrency for light jobs and does not generalize to other devices/recipes. A new eight-cell600-step preflight is still required. This phase executes entry gates and short calibration only, not the full matrix, and preserves old version03.

Commands are in the [Study entry](../../studies/main_exp/docs/versions/v030-before-local-20261013/README.md). `rpipe launch` packs same-wave concurrency by VRAM, optionally `--round N`; all train jobs finish before eval. The report is `docs/STUDY_REPORT.md`; `rpipe report` generates the number table.

Dedicated Data/Model and archive support live in `structure/`; `flow/prepare/run.py` checks provenance and actual runtime conditions, and `flow/process/run.py` calls `curves.py` only at Study scope. `__init__.py` only exports. Raw preparation uses `flow/prepare_data.py`; recipe calls `flow/execute/speed.py` for calibration. Manual tools are not automatically hooks; both source pillars freeze automatically.

<a id="archive-main-exp-v030--2026-10-12-本轮短实验"></a>
##### 2026-10-12 short run

Only existing `recipe.preflight` runs a same-device original/current MNIST/linear seed0 prefix,200 steps with complete test, retaining sampler/scheduler total budget80000, batch250, original precision and gates. Validation identity main-exp-flow-smoke-20261012-01 writes a new directory under this Study's `shared/validation/`, with segment values in `docs/`. This is not the formal64-Run experiment; it changes no version03 declaration/index/provenance/Run/PREFLIGHT.json and does not trigger formal Flow launch. Existing loader/scope contracts are checked separately. A200-step single cell is not the eight-cell600-step entry gate and cannot unlock long training. A formal rerun still requires new version, make and the full entry gate.
<!-- END archive-main-exp-v030 -->


## Latest engineering acceptance, 2026-10-16

B102 merged via PR133 into dev bc3743e; final-head and actual-dev CI passed all real jobs. Issue132 is closed and its local/remote branch removed; earlier pending text is historical. B103/Issue134 has four native Windows red regressions for shared-lock path-alias self-contention. Only the in-memory reentry identity changed; shared tests passed 72, core passed 1697 with one existing platform skip. Flow and CI remain pending. This is not a new scientific experiment and changes no layout or Study protocol; see [record§157–158](../development/record.md).


B103 local completion: CPU Flow 207 passed, no failures/errors/skips; final-head and actual-dev CI remain pending.


## B103 acceptance and B104 follow-up, 2026-10-16

PR135 merged into dev e80724c; final-head and actual-dev CI passed all real jobs. Issue134 is closed and its local/remote branch removed. B104/Issue136 reproduces namespace-dependent multi-root lock ordering with native concurrent locks. Four regressions produced two failures and two valid controls; the minimal shared identity/order fix passes 76 shared tests and two actual concurrent empty-directory migrations. Core/Flow/CI remain pending. No new scientific result, Study protocol or artifact layout change; see [record§159–160](../development/record.md).


B104 local completion: shared76/core1701/CPU Flow207 passed; core has one existing platform skip. All four new nodes executed. Final-head/actual-dev CI remains pending.


## B104 acceptance and B105 failure evidence, 2026-10-16

PR137 final-head and actual-dev CI passed all actual jobs; Issue136 is closed and its local/remote branch removed. B105/Issue138 reproduces stdout failure bypassing the existing success invalidation after a nonzero child exit. Three error families across retry modes plus valid controls yielded 6 failures / 2 passes before a one-line output reorder. All 88 scheduler tests pass; core/Flow/CI remain pending. No new scientific experiment, Study protocol or layout change; see [record§161–162](../development/record.md).


B105 local completion: scheduler88/core1709/CPU Flow207 passed, with one existing core platform skip. All eight new nodes executed. Final-head and actual-dev CI acceptance remains pending.


## B105 acceptance and B106 PNG evidence, 2026-10-16

PR139 final-head/actual-dev CI passed all real jobs; Issue138 is closed and its local/remote branch removed. The fifth-batch cleanup baseline is now139. B106/Issue140 reproduces partial PNG saving destroying the previous figure and leaking its owned figure. Nineteen test-first nodes gave17 failures/2 valid controls, then all19 passed after local atomic publication/owned-figure cleanup; core passed1728 with one existing platform skip. Flow/CI remain pending. Frozen-old/current rendering produced identical95233-byte PNGs for the same synthetic input; plot data/coordinates/dpi, Study protocols and layout are unchanged. This is not a new scientific result; see [record§163–164](../development/record.md).


B106 local completion:19 new nodes/core1728/CPU Flow207 passed; core has one existing platform skip. All19 new nodes, including native Windows readonly cleanup, executed. Final-head/actual-dev CI remains pending.


B106 remains one PR141: six supplemental close-error regressions yielded4 failures/2 valid controls before66f6f16 retained the original save/cancellation error; close failure after successful saving stays strict. All25 new nodes and core1734 with one existing skip pass; final Flow/CI remain pending. Old-head green is not new-head acceptance. Frozen-old/current PNG bytes still match at95233 bytes; no new scientific experiment or layout change.


B106 final local acceptance after stream-close supplement:25 new nodes/core1734/CPU Flow207 passed; core has one existing platform skip. All25 new nodes executed, including native readonly cleanup. PNG bytes remain identical. Final updated-head and actual-dev CI remain required; old9da02b4 green is historical.


## B106 acceptance and B107 cleanup priorities, 2026-10-16

PR141 updated-head/actual-dev CI passed all real jobs; Issue140 is closed and its local/remote task branch removed. One batch is accepted after cleanup baseline139. B107/Issue142 groups shared-lock unlock/close primary-error handling, acquisition causes, reentry and builder staging:25 test-first nodes produced19 failures/6 valid controls, then a local ExitStack fix passes101 shared tests. Core/Flow/CI remain pending. Budgets/OS primitives, shared-input identity and layout are unchanged; no new scientific conclusion. See [record§165–166](../development/record.md).


### Engineering progress: B108 / Issue144 (2026-10-16)

PR143 passed all actual final-head CI jobs and merged normally into dev e589862d; actual-dev acceptance remains pending. Twelve probes across four text/ledger writers confirmed eight primary-error/cancellation overrides by close IO. Thirty committed regressions first produced16 failures/14 valid controls, then design and minimal fixes made all30 green; adjacent98 passed/1 existing platform skip and core1789 passed/1 existing skip. Flow and both CI gates remain pending. Valid UTF8/newline, ledger and frozen-timestamp log bytes match frozen old sources; tracker state matches except root paths. This is engineering evidence, not new scientific results; Study protocols/numerical gates/budgets remain unchanged. Complete engineering acceptance before considering planned experiments; no GPU/timing/formal Study/layout changes. See[record §167](../development/record.md).


PR143 final head and actual dev passed all real CI; Issue142 closed and its local/remote work branch was deleted normally. B108 final local checks executed all30 new nodes; core1789 passed/1 existing skip and CPU Flow207 passed. Final-head/actual-dev CI remain pending. Engineering checks do not replace scientific experiments or relax numerical gates; see record §168.


### Engineering progress: B109 / Issue146 (2026-10-16)

PR145 passed all actual final-head CI jobs and merged normally into dev fc2b9b6; actual-dev acceptance remains pending. AST-only synthetic inputs confirmed the remaining Study comparison PNG writer:2 of4 probes overwrite the prior PNG and3 leak an owned Figure. Twenty-five committed nodes first produced23 failures/2 valid normal-save controls, then the minimal fix made the new and adjacent PNG suite67 green; core/Flow/both CI gates remain pending. Frozen old/new PNGs match at76906 bytes and identical SHA; scientific gates/thresholds/comparison/read functions and original drawing body remain unchanged. No Study/training/download or historical Run was executed/read. Future formal reruns still require new version/provenance, original numerical gates and budget; engineering image checks are not scientific results. See[record §169](../development/record.md).


PR145 final head and actual dev passed all actual CI; Issue144 closed and its local/remote branch was deleted normally. B109 final local25 new nodes, PNG adjacent67, core1814 passed/1 existing platform skip and CPU Flow207 passed; final-head/actual-dev CI remain pending. Engineering checks did not run a Study or replace scientific protocols/gates; see record §170.


### Engineering progress: B110 / Issue148 (2026-10-16)

PR147 passed all actual final-head CI and merged into dev9a7af37; actual-dev acceptance remains pending. Real12000-level JSON raises an unnormalized RecursionError at both outer/native-prepare freeze gates and overwrites prior success without data/model/execute calls. The initial accepted2000-level probe is retained and excluded as an invalid counterexample. Twelve committed nodes first5 failed/7 valid controls, then a minimal one-line classification fix made all12 and adjacent204 green; core/Flow/both CI gates remain pending. Strict parser/schema/recursion limits, scientific gates and real phase-failure policies remain unchanged. Engineering preservation is not a Study rerun or new science; no GPU/timing/formal Study. See[record §171](../development/record.md).


PR147 final head and actual dev passed all actual CI; Issue146 closed and its local/remote branch was deleted normally. B110 final checks executed all12 new nodes (core4/Flow8), adjacent204, core1818 passed/1 existing platform skip, CPU Flow215 passed. Final-head/actual-dev CI remain pending. Freeze classification preservation does not execute a formal Study, relax scientific gates or change JSON limits; see record §172.


## 2026-10-16 engineering update: consolidated read-boundary milestone

PR149 passed all actual final-head and merged-dev CI jobs; Issue148 is closed. The five-PR cleanup completed with baseline149. B111/#150 and B112/#151 share one milestone branch for optional peer-result reads and strict CLI parse diagnostics. The 20-entry inventory distinguishes source review, existing contracts and executed evidence; checkpoint/scientific evidence remains strict. All48 new nodes executed and passed; core1834 passed with1 existing platform skip, CPU Flow247 passed. Final-head and actual-dev CI await one consolidated PR. See[record§173–175](../development/record.md); raw.tmp evidence is not included in clones.

These are engineering contract results, not new scientific experiments or complete Study numerical revalidation on current dev. ChapterIX literature gaps, experiment order, actual precision/provenance, new versions and existing numerical gates remain unchanged. No GPU, timing study, training or download was performed. Next accept both CI sides of this milestone, then continue reproducible correctness checks in the three authorized classes; scientific execution still follows the existing plan and resource authorization.


Read-boundary supplement: continued source review found the omitted Run-process result fallback. B113/#153 reproduced three actual Runner cases that overwrite original result evidence. Four of14 formal tests failed first, with10 valid controls;22a4d57 extends the existing guard. It shares PR152 and the21-entry inventory with#150/#151. All62 new nodes executed and passed; core1834 passed/1 existing platform skip, CPU Flow261 passed. Final-head/actual-dev CI remain pending. Runner policies, scientific gates, metric conversion, Study experiment order and resource authorization remain unchanged; no new scientific experiment. See[record§176–177](../development/record.md).


## 2026-10-16 engineering update: final evidence writes at shared execute

PR152 passed all actual final-head and merged-dev CI;#150/#151/#153 are closed and the topic branch was cleaned. The accepted baseline is now dev f956c635, not a main release. B114/#154 found that shared execute swallowed final tracker-write refusal after a successful body. Six synthetic Runner probes confirmed two false-success cases;24 formal tests first yielded6 failures/18 controls.82890ce adds only a local completion flag to distinguish strict successful finalization from best-effort failure finalization. All24 new nodes executed/passed; core1846 passed/1 existing platform skip, CPU Flow273 passed. Final-head/actual-dev CI remain pending. See[record§178–180](../development/record.md) for the write-boundary inventory and evidence limits; raw.tmp evidence is not cloned.

No new scientific experiment or numerical algorithm, training step, periodic flush, checkpoint/RNG, architecture or layout change is involved. External Algorithms and Studies will no longer finalize success after the final evidence write fails. No format/directory migration is needed; formal revalidation still uses a new version, actual provenance and unchanged numerical gates. Literature gaps, ChapterIX experiment order and resource authorization remain unchanged; CPU engineering checks do not replace Study scientific validation.


## 2026-10-16 engineering update: immutable archive creation and admission policy

PR155 passed every actual final-head/merged-dev CI job;#154 is closed and its local/remote topic was cleaned. Accepted baseline: dev720a13df. B115/#156 consolidates the same race in main_exp/main_probe source exporters: another creator's bytes could be overwritten after the existence observation. Both now create exclusively, preserve primary write errors/partial diagnostics and retain the prior manifest on refusal. The28 formal tests first yielded12 failures/16 controls; after832c1c9 all28 executed/passed. Core1874 passed/1 existing platform skip; CPU Flow273 passed. Final-head/actual-dev CI remain pending. All other source bytes, pinned refs, valid Git blobs, manifest/hash formats, paths and scientific gates remain unchanged. See[record§181–183](../development/record.md). Synthetic Git/isolated AST tests load no historical code or formal archive and run no data, training or timing experiment. Future scientific revalidation still uses a new version/actual provenance; previous Runs/negative evidence remain, with no format/directory migration.

The19:53 authorized future GPU-admission policy combines a display whitelist with process type: Figma, browsers, dwm, explorer, Office and RayLink(including RayLinkCapturer), ToDesk, Sunlogin, Parsec and TeamViewer of Graphics/G or C+G type with no/negligible CUDA-compute allocation are display processes. Unknown identity alone must not block admission; never kill or suspend them. Admission considers actual CUDA-compute occupants, actual free VRAM and host RAM equal to measured peak plus2GB. Unknown processes block only with real CUDA-compute allocation, with evidence reported to supervision. Tracked capacity entry points have no identity filter, only free-memory checks. Historical.tmp calibration/recovery scripts remain evidence, not current admission authority; classification/RAM checks must be implemented and verified before future use. No GPU/performance work is started; resource authorization, scientific experiment order and numerical gates remain unchanged.


## 2026-10-16 engineering update: read-only log chronology across offsets

PR157 passed every actual final-head/merged-dev CI job;#156 is closed and its local/remote topic was cleaned. The accepted baseline is dev50d5a6bd. B116/#158 confirmed three wall-clock-text ordering errors in six actual CLI cases (DST fold, UTC day carry and different offsets). The28 formal tests first yielded8 failures/20 controls;ba665de changes only the shared sort key. All28 new nodes executed/passed; core1902 passed/1 existing platform skip, CPU Flow273 passed. Final-head/actual-dev CI remain pending. Aware native timestamps within datetime precision sort by UTC instant without rewriting output lines/formats/inputs. Missing, malformed, unzoned, overflow and legacy sub-microsecond keys keep their fallback; no new sub-microsecond UTC semantics are claimed. See[record§184–186](../development/record.md).

This orders engineering events, not scientific curve/sampler coordinates. Status, metrics, experiment order, numerical gates, architecture/layout and original Run evidence remain unchanged. No new scientific experiment, GPU/timing/main/tag action occurs. Future actual revalidation still uses a new version/actual provenance and retains previous numerical/negative evidence. The19:53 GPU-admission policy and resource authorization remain in force; no GPU task is started or interrupted.


## Concurrent endpoint isolation engineering update, 2026-10-16

PR159 passed every actual final-head and merged-dev CI job;#158 is closed and its topic was cleaned. The accepted baseline is dev ab5c825. B117/#160 reproduced two incorrect endpoint reads and residual dataset class state when different cache roots bind the same class concurrently. Red commit9c95667 produced4 failures/4 nested controls;fffd872 holds a standard-library RLock across saving, binding, construction and restoration, preserving nested and exceptional exits. All13 endpoint tests passed; core1910 passed/1 existing skip, with all8 new nodes executed. CPU Flow273 passed with no failure/error/skip; final-head/actual-dev CI remain pending. A real Factory/two-root probe uses synthetic builders, with no network, download or scientific run. Endpoint definitions, cache layout, download/hash, precision and numerical gates remain unchanged. See[record§187–189](../development/record.md). Future scientific validation still needs a new version, actual provenance and the existing prerequisite gates; this update adds no scientific conclusion or changes to the five Study protocols.


## Temporary environment concurrency follow-up, 2026-10-16

B118/#162 extends the same temporary-global-state audit: independent project preparations can observe another call's restored TQDM_DISABLE and leave its temporary value behind. Red9 tests yielded4 failures/5 controls;0aaae0b adds only RLock ownership around the existing environment save/set/prepare/restore scope. It is batched with B117 in PR161 rather than a separate PR. All27 new/adjacent tests passed; combined core1919 passed/1 existing skip, all17 new nodes executed. CPU Flow273 passed with no failure/error/skip; final-head/actual-dev CI remain pending. Synthetic preparation bodies exercise the real public prepare path/cache locks and isolated projects without network, data, GPU or science. The five Study protocols and numerical/recovery gates remain unchanged. See[record§190–191](../development/record.md). The20:56 cadence requires related B items to accumulate locally into complete milestones and only a few PRs per day; existing PR161 finishes normally.


## Loader isolation progress and an unpassed boundary, 2026-10-16

PR161 passed every actual final-head and merged-dev CI job;#160/#162 are closed and its topic was cleaned. The accepted baseline is dev8480d795d0cb09714deb9bdd4bd79f41bdc5b2a9. B119/#163 and B120/#164 remain a local loader-isolation milestone. After the31-test candidate and core1950 passed+1 existing skip/Flow273 passed, two new late-import tests in registered functions exposed a candidate regression. The current33-node suite is not accepted; the earlier full checks exclude these two tests and cannot accept this boundary. Private module lifetime must be fixed without narrowing supported behavior. See[record§192–197](../development/record.md). Related B items accumulate locally under20:56 cadence, with no immediate new PR. No scientific run or new numerical conclusion; the five Study protocols, prior Runs and scientific gates stay unchanged.


## Complete local loader checks, 2026-10-16

The current B119/#163 and B120/#164 implementation passed core1964 tests with1 existing platform skip and CPU Flow279 tests. All51 new nodes executed:45 unit and6 integration. Module lifetime/identity covers late imports and annotations; native System and compare readers preserve legacy helper state and relocated state without the original namespace cache. Runtime/foreign-cache identity, strict missing-class refusal and unchanged checkpoint input bytes are covered. Exact four-file source scope is recorded in[record§198–201](../development/record.md). Earlier candidate failures remain preserved. These local results cannot replace final-head/actual-dev CI, which has not run; no PR is pushed and163/164 remain open. The accepted baseline remains PR161/dev8480d79. Related B items accumulate locally into complete milestones under20:56 cadence, with few PRs. The five Study protocols, prior Runs, scientific negative evidence and numerical/recovery gates stay unchanged; no GPU, timing, download or new scientific experiment is performed.


## PR165 loader milestone review, 2026-10-16

After B119/#163 and B120/#164 entered the single PR165 milestone, pre-acceptance review found B121/#166: a recipe-local dynamic helper import succeeds on accepted dev but raises ModuleNotFoundError on the candidate. Ten red nodes (8 failures/2 controls) preceded9166cbd, which binds the dynamic entry to the existing owning Study namespace. Current core:1974 passed/1 existing skip; CPU Flow:279 passed; all61 new nodes executed. See[record§202](../development/record.md). Original and candidate failures remain preserved; all three items stay in one PR. New final-head/actual-dev CI is pending,163/164/166 remain open, and the accepted baseline is still PR161/dev8480d79. No scientific run, number or conclusion is added. The five Study plans, historical counterexamples, recovery/numerical gates and artifact layout stay unchanged; future scientific reruns still require a new version and original prerequisite gates.


## Accepted PR165 engineering baseline, 2026-10-16

The single B119/#163, B120/#164 and B121/#166 milestone is accepted through PR165: every actual matrix and aggregate passed on final head3a74728 and actual dev70a8750. All three issues are closed; the merged local/remote topic branch is removed with its commits preserved in dev. The accepted engineering baseline is now dev70a8750. Local core:1974 passed/1 existing skip; CPU Flow:279 passed; all61 new nodes executed. See[record§203](../development/record.md). Candidate failures and original red evidence remain preserved. This adds engineering evidence, not scientific experiments or conclusions. The five Study protocols, historical numbers/counterexamples, numerical/recovery prerequisite gates and file/artifact layout remain unchanged. Scientific reruns still require a new version and actual provenance; CI does not replace measurements. With the open-issue snapshot empty, review continues in the three correctness classes. Closeout documentation accumulates locally with the next related milestone rather than creating a separate documentation PR.


## Local B122 failure-evidence update, 2026-10-16 (not merged)

The accepted engineering baseline remains PR165/dev70a8750. B122/#167 confirms that terminal success-log IO refusal bypasses Runner failure finalization. Twelve red nodes (8 failures/4 normal controls) preceded761700f, which only indents the original record into the existing try boundary. Record timing, the process success-preservation exception and result/log formats remain unchanged. Runner49/core1974 passed+1 existing skip/CPU Flow291 passed; all12 new nodes executed. See[record§204–205](../development/record.md). Local results do not replace final-head/actual-dev CI;167 remains open, related work accumulates on the same local milestone branch without an immediate per-bug PR. No scientific experiment, number or conclusion is added. The five Study protocols, historical evidence, numerical/recovery prerequisites and artifact layout remain unchanged; scientific reruns still require a new version and actual provenance.
