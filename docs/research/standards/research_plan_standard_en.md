# Research Plan Standard — English translation

Translation of [the unchanged Chinese master](<研究计划规范 Research Plan Standard.md>), updated 2026-10-16. This translation is informational; the Chinese master governs. See [README](README.md) for source hashes, local adaptations and references that belong to the upstream DEM collection. This is a translation, not an independent verification of the cited research.

# I. Purpose and scope

A research plan explains the problem, the reach of existing evidence, the proposed validation, and which outcomes would change the current judgment. AI research considers six connected aspects: **literature review, data, models, algorithms, metrics and experimental design**, tied to explicit research questions and hypotheses.

This standard applies to literature reviews, theory, method development, reproduction, benchmarks and systems research. It can be adapted to other sciences. A model may be neural, statistical, mathematical or mechanistic; an algorithm may train, infer, solve or analyze.

Established: 2026-10-11. Updated: 2026-10-16. Status: recommended for adoption. Each project separately confirms its approach, budget, owner and deadlines.

Develop the six aspects as needed; combine or shorten them where appropriate. Theory can validate through proofs, counterexamples and complexity analysis. A literature study can stop at evidence comparison and gap analysis. Sources must be genuine, actual scope explicit, and conclusions traceable.

AI assistance may turn these ideas into prompts to detect omissions, record differences and arrange next steps. Choose steps, reproduction precision and recording detail according to the question; do not make every check a prerequisite for progress.

# II. From an idea to a plan

The basic cycle is **idea → literature review → research questions and hypotheses → joint planning of data, models, algorithms, metrics and experiments → execution and analysis → conclusions and new ideas**. Preliminary questions guide searches; review refines the formal question. Literature and experiments can produce new ideas.[6][7]

## 1. Idea

Record an idea, anomaly, practical difficulty or doubt about an explanation, with its source and trigger. State the possible problem or opportunity in one sentence, then a preliminary search question. Novelty and feasibility remain unverified.

## 2. Literature review

Check existing work, alternative explanations, strong baselines and evidence gaps around the preliminary question. Extract data, model, algorithm, metric and experimental conditions. Decide whether the idea is solved, worth continuing, needs adjustment or should be deferred; see Chapter III.

## 3. Questions and hypotheses

Specify the object, conditions, current limitation, expected improvement and scope. Ask an answerable question, for example: can post-processing improve probability calibration for a fixed model and test distribution while meeting accuracy and computational-cost constraints?

A hypothesis states a possible mechanism and observations that could refute it. Exploratory reviews may begin with questions and form hypotheses later. Support differences from the closest work with evidence; label unchecked novelty as unverified.

Hypothesis testing must state falsifiable predictions and uncertainty. Exploration may form hypotheses after evidence is collected; distinguish discovery from later independent testing.[8]

## 4. Connecting the six aspects

Literature supports the question and approach. Data defines the object and generalization scope; models represent it; algorithms operate on or solve models; metrics make the goal measurable; experiments organize comparisons and control alternative explanations. Cross-check the five technical arrangements and iterate with results.[9]

| **Aspect** | **Questions** | **Possible output** |
| --- | --- | --- |
| Literature | How is the problem defined? Which approaches, strong baselines, counterexamples and open questions exist? What supports this plan? | Located evidence table and rationale for the approach |
| Data | How is the object observed or represented? Sources, collection, cleaning, annotation, splits, biases and restrictions? | Data description, versions and split rules |
| Models | Which structure or assumptions represent the object? Inputs, outputs, parameters, state; fixed and variable parts? | Model definitions, versions and baselines |
| Algorithms | How to train, infer, search or solve? Steps, objectives, changes, cost and failure conditions? | Equations or pseudocode, algorithm baselines and configuration |
| Metrics | What counts as better? Measurement, units, direction, statistical conventions and decision criteria? | Primary metrics, necessary constraints and protocol |
| Experiments | Which comparison tests which hypothesis? What changes and what is controlled? How are alternative explanations excluded? | Controls, ablation matrix, budget and interpretation rules |

A model describes representation; an algorithm describes operations or solution. A model can use different algorithms and vice versa. Explain changes in data, models and algorithms separately so that effects can be attributed.

# III. Planning the six aspects

## 1. Literature review

Follow question → keywords and synonyms → search and citation tracing → screening → reading → comparison → gap. Record scope, information cutoff, search dates and inclusion/exclusion rationale. Search, archiving and reading depth follow the upstream Literature Review Standard (see the Chinese master's relative reference and README dependency note).

Extract data, models, algorithms, metrics and experimental design while reviewing core sources:

| **Field** | **Contents** |
| --- | --- |
| Problem and source | Question, paper or official implementation, version, section or figure/table location |
| Data and models | Dataset, split, preprocessing, model structure and size |
| Algorithms | Mechanism, training/inference steps and closest baselines |
| Metrics and experiments | Metric definition, controls, ablations, hardware, budget and comparison conditions |
| Evidence and use | Authors' results, actual reading scope, limitations and role in this plan |

Citations and publication venues help discover leads; relevance and evidence quality determine adoption. Label published numbers, your reproduced numbers and cross-paper extracts separately. Different conditions can inform an approach but cannot establish a fair ranking. Mark missing evidence for checking; finding no work does not prove a gap.[1][2]

## 2. Data

Record source, license or authorization, version, sample unit, size, distribution, annotation source/error, collection, cleaning, preprocessing and exclusion rules. For people or sensitive information, add applicable ethics, privacy and usage requirements.

Choose splits isolated by entity, time, place or another unit according to the generalization question. Respect split boundaries in cleaning, feature processing and parameter estimation; do not leak test information into training, selection or calibration. Record known duplicates, pretraining contamination and distribution biases, including what cannot be confirmed.

Prefer the explicit protocol of an established benchmark. Explain protocol changes and comparability. If data cannot enter Git, retain acquisition instructions, version identifiers, split manifests or verifiable summaries.

## 3. Models

Describe inputs/outputs, structure, parameter count, initialization or weights, version, assumptions and boundaries. Distinguish the model, training objective, inference procedure and surrounding system.

Choose baselines that answer the question, including useful simple baselines and the closest methods. Treat model size, pretraining data or weight-source changes as experimental factors. For theory and other sciences, specify variables, relations, assumptions and identifiability conditions.

## 4. Algorithms

Use equations or pseudocode for key steps. Identify the changed step and proposed mechanism relative to the baseline. Record objectives, hyperparameters, stopping conditions, randomness, state updates, time and space costs.

Reuse existing implementations and pin repository commits and configuration. Distinguish training loss from final evaluation metrics. Equal-budget quality comparisons fix budget; cost-to-quality comparisons fix quality. Both may be reported with separate definitions.

Prefer authors' official code; record repository, version/commit, license and necessary adaptations. Keep it in its official repository or use project src/ or reference/; no universal import layout or per-file hashing is required. Distinguish official, third-party and rewritten code honestly; a rewrite is not reproduction using official code.

## 5. Metrics

Choose primary metrics for the claim and necessary quality, resource or risk constraints. State meanings, direction and calculation conventions that affect comparison; add implementation versions and conditions for precise measurements. Give threshold rationale; unresolved values remain to be confirmed.

Locate key numbers: source figures/tables for published results, logs/results for measurements, calculation conventions for summaries. Separate planned targets, reported values and local measurements; unexecuted results remain unmeasured.

Performance measurement states timing scope, warm-up, synchronization, batch and concurrency conditions. Randomized experiments state repetitions and variability estimates; choose repetitions from expected effect, variance and budget, not one count for every project.

Use actual independent units in statistics; distinguish sampling variability from training randomness. Predefine methods and assumptions for intervals, tests or sample-size estimates where needed, and assess practical meaning. Correlated samples are not independent repetitions; one best run does not establish stability.

## 6. Experimental design

Tie every experiment to a question or hypothesis. Record treatment, control, fixed conditions, evaluation data, metrics, repetition, cost and interpretation rules. Choose relevant types, rather than every combination:

| **Type** | **Purpose** |
| --- | --- |
| Baseline reproduction | Prefer official code and broadly reproduce central results under similar key conditions; understand performance/gaps and establish a credible comparison |
| Main comparison | Test the full method under explicit comparable conditions |
| Ablation or mechanism | Test components and alternatives under comparable coverage, intervention locations and protocols; begin with simple settings |
| Robustness and boundaries | Test the reach across data, models, parameters or environments |
| Failure analysis | Explain degradation, anomalies, ineffective outcomes and triggers |

Method development usually starts by understanding/reproducing the original method. Reproduction establishes main behavior, trends and implementation credibility, not exact agreement of every number or rerunning the whole paper. Investigate differences large enough to change judgments; do not call a method ineffective based on an unreproduced baseline. General reproducibility factors appear in [9].

### 1. Baselines and official reproduction

Use the following as needed. Define representative results and broadly acceptable differences; specify exact tolerances and repetitions only when numerical alignment matters to the conclusion.

| **Step** | **Action and record** |
| --- | --- |
| Find implementation | Check paper, project page and authors' repository. If no official code is found, state searched sources and identify a trusted third-party or own implementation |
| Prepare and align | Record code version and adaptations; prefer official configuration; check data, model, evaluation script, coverage and intervention location, and comparison-relevant differences |
| Small checks and reproduction | Check execution, then representative results. Begin with authors' weights and evaluation scripts if useful; train or run method generation as required; record actual scope |
| Compare and arrange next steps | Record main numbers, trends, source files and differences. Continue when behavior broadly agrees and key differences are explained; investigate consequential gaps, or state resource limits and follow-up |

Describe code execution, weight-based reproduction and training/method reproduction separately. Evaluating author weights does not reproduce training; a project that does not study training need not retrain the full model for that reason.

### 2. Ablation coverage and numerical traceability

Ablate key components while keeping coverage, intervention location and protocol comparable; explain differences. In quantization, V/O-only 56 layers and the full 196-layer main table have different conditions, so their numerical difference is not automatically an algorithm effect.

Choose training, affine, scaling, rotation or other components according to the question; no exhaustive cross-product is required. Start simply with minimal original-weight changes, for example no scaling, then add key parts. Keep ineffective/degraded outcomes, distinguish targets from measurements, and locate source files.

Once a credible baseline exists, compare your method. Tune/select on development data; use the final test set for predetermined evaluation. Retain exploration and post-hoc changes; do not present metrics chosen after seeing results as prespecified. Explain multiple-comparison handling when material. Observational research states confounding and identification assumptions; correlation is not a causal conclusion.

### 3. Experiment selection and optimization

Distinguish hypothesis/mechanism testing from finding better configurations, then select tools as needed:

| **Method** | **Suitable problems and planning details** |
| --- | --- |
| Design of Experiments (DOE) | Multi-factor comparisons: randomization, blocking, full or fractional factorial designs; main/interactions, factors, levels and confounding limits |
| Operations Research (OR) | Frame selection, resource allocation or scheduling as decisions: variables, objective, budget, constraints and stop rules; simulation may use optimization, ranking/selection and replication-budget allocation.[10] |
| Bayesian Optimization (BO) | Expensive, non-differentiable black-box goals such as hyperparameters or simulation configurations: search space, initial trials, surrogate, acquisition, noise, cost and stopping; choose the next evaluation from observations.[11] |
| Bayesian Experimental Design | Choose experiments by expected utility for identification, information or decisions: prior, observation model, utility and design constraints.[12] |

OR supplies decision/optimization tools; BO finds good objective values; Bayesian design defines utility from the research purpose. Configuration search and mechanism testing need different criteria. Keep complete adaptive-search trajectories, budget/fairness conventions and independent evaluation or a predetermined validation stage; the tuning optimum alone is not final evidence.[9][11]

# IV. A short guide from plan to execution

1. **Record the idea:** source, idea and initial question; fill the [template](research_plan_template_en.md).
2. **Review literature:** extract the five technical aspects and check existing solutions, baselines, disputes and gaps.
3. **Set questions/hypotheses:** scope, mechanism and falsifiable criteria; continue, adjust or defer.
4. **Plan method/validation:** jointly design data, models, algorithms, metrics and experiments, comparisons, budget and interpretation; use DOE, OR or BO as needed.
5. **Small checks/baseline reproduction:** prefer official code, check execution, then representative behavior and differences. Develop your method after establishing credibility; investigate clear anomalies first.[3][9]
6. **Execute/analyze:** keep runs, search trajectories and failures; analyze by planned rules and record protocol-change effects.
7. **Conclude/find new questions:** say which hypotheses are supported, refuted or unresolved, with boundaries and next ideas.

Plans may iterate. Date important changes and explain their reason and affected experiments. Before execution, answer why, how to compare/judge and what resources are needed. Afterwards, trace what actually happened, its evidence and generalization limits.

# V. Use for different research types

| **Type** | **Use of the six aspects** |
| --- | --- |
| Literature | Evidence set, comparing data/models/algorithms/metrics/validation; evidence maps, disputes and recommendations; actual work stops at review |
| Theory | Objects, distributions/instances, mathematical models/assumptions and solution; theorem conditions, error/complexity bounds; proofs, counterexamples and necessary numerical checks |
| Methods | Broad official-code reproduction before improvements; explain differences, ablate key components and do not attribute failure to methods under a clearly unreproduced baseline |
| Reproduction | Prefer official code, main numbers/trends and gaps; distinguish weights, training/method and third-party implementations and actual coverage |
| Data/benchmarks | Sampling, annotation, splits, metric validity and baseline coverage; state what evaluation represents |
| Systems/applications | Task quality alongside latency, throughput, resource use and cost; fixed load, hardware and end-to-end boundaries |

# VI. Worked example

This is a teaching proposal, not an approved project or measured result.

**Idea:** a classifier can be accurate while making high-confidence errors; explore low-cost probability adjustment. First review the phenomenon, solutions and evaluation limits.

**Question/hypothesis:** for a fixed classifier, can temperature scaling improve test-distribution probability quality while meeting accuracy constraints? It adjusts probability scale. No stable primary improvement, or improvement limited to fitting data, fails to support the hypothesis.

| **Aspect** | **Example plan** |
| --- | --- |
| Literature | Check calibration, temperature scaling and evaluation limits; record calibration sets, models, metrics and controls. The upstream calibration outline is referenced in the Chinese master; see README for its unavailable DEM-only dependency |
| Data | Fixed training/calibration/test splits; fit temperature only on calibration. Dataset version, size and isolation unit await project selection |
| Model | Fixed classifier/weights; save raw predictions; no classifier parameter updates |
| Algorithm | Compare raw and temperature-scaled output with fixed fit objective, search range and stop rules |
| Metrics | Test NLL primary, accuracy constraint, Brier and fixed-bin ECE supplementary; set improvement criteria and statistics before formal execution |
| Experiments | Paired same-test-sample comparisons, calibration-size changes and failure cases; independent training repetitions if claiming robustness across training randomness |

Fill data, parameters, budget and thresholds before formal experiments. Conclusions cover measured models/distributions only; cross-task claims require independent evidence.

# VII. Use in a GitHub repository

## 1. research-plan skill

The [research-plan skill](https://github.com/DreamSoul-AI/knowledge/tree/main/skills/research-plan) turns this workflow into reusable agent guidance, with a [Chinese guide/template](https://github.com/DreamSoul-AI/knowledge/blob/main/skills/research-plan/references/README_zh.md).

For first installation, send Codex:

```text
$skill-installer
Please install the skill in https://github.com/DreamSoul-AI/knowledge/tree/main/skills/research-plan.
```

Invoke from the next turn, for example:

```text
Use $research-plan to organize this idea into a research plan: ...
First review literature, then specify questions/hypotheses and jointly plan data, models, algorithms, metrics and experiments.
Mark unknown settings to be confirmed. Plan only; do not train or publish.
```

For existing plans, request checks of evidence, six-aspect connections and design. The standard/template explains the approach; the skill supplies AI reminders. Decide whether to investigate, record limits or continue based on effects on conclusions, not whether every box is filled. The project confirms data, budget, ownership and judgments.

## 2. Plans and execution in repositories

Save the filled plan as docs/research-plan.md and link it from README. Existing repositories retain their structure; this table describes responsibilities, not mandatory new directories. RPipe's adaptation in [research.md](../../development/research.md) preserves macro plans and Study PLAN separation.

| **Contents** | **Suggested location/responsibility** |
| --- | --- |
| Idea, literature, questions/hypotheses, six aspects | docs/research-plan.md as current-plan entry |
| Evidence/references | Existing docs; docs/literature.md if large |
| Data/splits | Existing modules and descriptions; no private raw data in Git |
| Models/algorithms | Existing modules, interfaces and layers |
| Configuration/metrics | Existing configuration, evaluation and necessary tests/tools |
| Evidence/reports | Logs, raw metrics, artifact locations and analysis under project artifact rules |

Each implementation task links a planned question or experiment. Issues state inputs, scope, validation and output; PRs link issues, actual validation and limits. Maintain planned and implemented states separately.

Ideas may begin as exploration issues with source and preliminary question. After review, put selected questions in the plan and link implementation/experiment issues. OR/BO objectives, constraints, budgets and trial trajectories fit existing configs/logs; no new research framework is required.

Each formal run records at least experiment ID, commit, actual configuration and uncommitted changes, data/splits/model versions, dependencies/hardware, randomness, command, state, raw results and artifacts. Keep negative results and failure reasons. A seed alone does not guarantee reproducibility.

Repository maintainers may adopt this AGENTS.md instruction within scope:

> Begin research by recording the idea's source and reviewing literature to specify questions/hypotheses. Before implementation, read the current plan and related code and check data, models, algorithms, metrics and design. Reuse implementations; start with small checks and necessary baselines. Formal runs record versions, configuration, environment, search trajectory and raw results, retaining failures/negative results. Separate literature conclusions, hypotheses and execution states; report boundaries and new questions.

# VIII. References and actual use

The original standard synthesizes these sources; its six-aspect minimum and template are recommended conventions.

| **No.** | **Basis and scope used by the original author** |
| --- | --- |
| 1 | [knowledge literature-search](https://github.com/DreamSoul-AI/knowledge/blob/f721a1c3b6693f702464707f0fd97c6b7c1112c6/skills/literature-search/SKILL.md): question, search, screening, citation tracing and gaps |
| 2 | [knowledge paper-analysis](https://github.com/DreamSoul-AI/knowledge/blob/f721a1c3b6693f702464707f0fd97c6b7c1112c6/skills/paper-analysis/SKILL.md): separate claims, methods, data, baselines, metrics, experiments and limits |
| 3 | [knowledge pytorch-project](https://github.com/DreamSoul-AI/knowledge/blob/f721a1c3b6693f702464707f0fd97c6b7c1112c6/skills/pytorch-project/SKILL.md): reproducible data/evaluation and checks before formal training |
| 4 | Upstream Literature Review Standard, Chapters VI/VIII: depth, classification, comparison and evidence state; DEM-only relative dependency retained in Chinese master |
| 5 | Upstream General Research Standard, Chapters II/IV/VI: scope, traceability and validation records; DEM-only relative dependency retained in Chinese master |
| 6 | Ackermann and AGA Research Committee (2024), [From the idea to the research question and hypothesis](https://doi.org/10.1007/s00142-024-00682-x), Arthroskopie 37:232–237: idea/review, question/hypothesis sections; clinical rules not transplanted into AI |
| 7 | Cohen and Howe (1988), [How Evaluation Guides AI Research: The Message Still Counts More than the Medium](https://paulrcohen.github.io/papers/952-949-1-PB.pdf), AI Magazine 9(4): stages, evaluation and feedback throughout research |
| 8 | Forde and Paganini (2019), [The Scientific Method in the Science of Machine Learning](https://arxiv.org/abs/1904.10922), methodology position paper: hypothesis testing, statistical/systematic uncertainty |
| 9 | Gundersen, Coakley, Kirkpatrick and Gil (2022), [Sources of Irreproducibility in Machine Learning: A Review](https://arxiv.org/html/2204.07610v2), Sections4–9: jointly plan design, algorithms, data, evaluation and documentation |
| 10 | Amaran, Sahinidis, Sharda and Bury (2016), [Simulation optimization: a review of algorithms and applications](https://doi.org/10.1007/s10479-015-2019-x), Annals of Operations Research 240:351–380: optimization, ranking/selection and budget allocation |
| 11 | Frazier (2018), [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811): surrogates, acquisition and expensive black-box sequential search |
| 12 | Chaloner and Verdinelli (1995), [Bayesian Experimental Design: A Review](https://doi.org/10.1214/ss/1177009939), Statistical Science10(3):273–304: decision theory and expected utility |

The original author checked GitHub on 2026-10-11 at commit f721a1c3b6693f702464707f0fd97c6b7c1112c6. Old tutorials provide background; skill bodies and current project rules govern adoption. Historical summaries of citations, venues or resources do not directly judge quality.

The original author's literature verification date is 2026-10-14, with sections/abstracts read as listed. Sources support the components and connections; the six-aspect division and entry order are this standard's organizational convention. These statements describe upstream verification, not a new review by this translator.
