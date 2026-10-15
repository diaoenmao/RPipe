> **RPipe research plan: verifiable contracts for research execution and evidence consistency**
> Version 1.0; updated 2026-10-16. The [Chinese plan](research_plan.md) is authoritative; this is its chapter-by-chapter English counterpart.
> Stage: plan revision and targeted preliminary literature review; no new scientific experiments under this plan.
> Implementation baseline: dev `ef6084681160a68588e0b2a359ab7cc07c398879`; the published reference is v0.4.1. Dev is not a release.
> This round uses no GPU, timing study, training or data download, and changes no file architecture or artifact layout.

# I. Inspiration and initial problem

## 1. Topic, origin and value

This is systems and applications research, informed by reproducibility methodology and a limited reproduction of historical implementations. RPipe is a source template and execution foundation for research projects, not a new classifier, optimizer or quantization method. Its object of study is the evidence chain from declarations through execution, persistence and read-only presentation to human conclusions.

Three observed failure families motivate it: failed writes replacing earlier evidence; damaged metadata breaking or misleading an entire display; and concurrent publication colliding or exposing incomplete text. The recent compare counterexample adds an important boundary: equal numbers alone do not establish correctness when both sides load the same checkpoint outside their Run. See the [artifact consistency design](../development/artifact_consistency_design.md), [development record](../development/record.md) and [#101](https://github.com/DreamSoul-AI/RPipe/issues/101). These are observations about this repository, not evidence that other frameworks share the defects.

The initial question is whether a small set of evidence-preservation, optional-read and atomic-publication rules can make results traceable, failures localizable and local corruption incapable of producing false success, under fixed computation and disk contracts. Reducing ambiguity in research evidence is the intended value. Novelty and external generalizability remain unproven.

## 2. Constraints and responsibility

- Enmao owns scope, resources and research direction. The Agent implements, searches, organizes evidence and prepares PRs. CI checks engineering contracts; it does not replace research judgment.
- This delivery is the Chinese plan, this complete English counterpart, literature evidence and applicable standards, integrated through a work PR into dev. Historical Study plans and results are not rewritten or presented as preregistrations of later lessons.
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
| D2 Short comparison evidence | Paired historical/current-chain Runs in main_probe; dataset × model pair | Original precision, seed, inputs, parameters, RNG and scheduling conditions; new source requires a new version and original gates | [Plan](../../studies/main_probe/docs/PLAN.md), [report](../../studies/main_probe/docs/STUDY_REPORT.md); read-only here |
| D3 Long-curve evidence | Cross-seed main_exp curves and each Run's own best evaluation | Preserve original training, interruption and new-identity records; do not relabel old provenance | [Plan](../../studies/main_exp/docs/PLAN.md), [report](../../studies/main_exp/docs/STUDY_REPORT.md); raw artifacts are not cloned |
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
| Conditional scientific metrics | Parameters/inputs/RNG/scheduling equivalence, loss/correct counts, curves/model ordering | Original Study definitions; no mixed-version aggregation | [main_probe gates](../../studies/main_probe/docs/PLAN.md), [main_exp gates](../../studies/main_exp/docs/PLAN.md) | Retain original gates; an unexecuted current baseline is not accepted |

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
| Two historical scientific Studies | Their PLAN/STUDY_REPORT and formal JSON | Historically accepted; current baseline not rerun | Only their recorded provenance and conditions, not current dev |
| Three correctness families | Approved design and linked red/green records | Existing repairs/regressions integrated into dev | Tested invariants, not absence of all bugs or scientific equivalence |
| PR98 / B086 | Record §§111–114; 56 corrected regression nodes | Final-head and actual-dev CI accepted | Text-cell structure for its enumerated cases |
| PR100 / B087 | Record §§113–115 and acceptance evidence; 24 nodes | Actual dev `187bb2e` accepted; ended remote deleted | Optional tracker decode boundary, not training-resume fallback |
| PR102 / B088 | Record §116 and acceptance evidence; 17 nodes, compare 56 | Actual dev `ef608468` accepted; ended remote deleted | Checkpoint ownership and comparison label structure, not unrestricted deserialization safety |
| This plan and review | This file, Chinese counterpart and literature ledger | Documentation only; no new scientific runs | Search/read scope and planned tests, not external superiority |

Negative results, old failures and canceled/pending checks remain in the development record and local evidence; `.tmp/` is not cloned. Missing original weights, exact sampler/resume equivalence outside supported contracts and universal external reproduction remain unresolved or unverified. Do not convert historical success or skipped tests into current acceptance.

# X. Artifacts, review and change history

The sole macro-plan authority is [research_plan.md](research_plan.md), with this English counterpart and [literature.md](literature.md). Do not duplicate it at `docs/research-plan.md`. Historical Study plans, scientific figures and formal results stay in their original Study; source standards stay in [standards/](standards/README.md).

Link each implementation batch to its RQ, real failing case, regression, PR and actual-dev acceptance. Engineering issues fixed in dev stay open until the next release; explorations #89/#90 are not closed merely as outdated. Delete an ended remote only after ancestry and actual CI checks; retain evidence and follow current authorization for local branches. This round does not merge main, tag, delete data or trigger experiments.

| **Date** | **Change** | **Reason and impact** |
|---|---|---|
| 2026-10-16 | Establish the template-based Chinese macro plan, full English counterpart and literature evidence/standard | Connect RQs, six dimensions, evidence levels and execution status; leave historical Study plans/results, runtime code and authorization unchanged |

Source provenance and actual reading limits are in [literature.md](literature.md). The two research-plan masters retain the bytes recorded in [standards/README](standards/README.md).
