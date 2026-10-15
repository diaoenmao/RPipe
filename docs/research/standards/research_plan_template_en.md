# Research Plan Template — English translation

Informational translation of [the unchanged Chinese master](<研究计划模板 Research Plan Template.md>), updated 2026-10-16. The Chinese master governs. See [README](README.md) for provenance and RPipe placement rules.

# I. Idea and preliminary question

Copy this template into the project's topic directory or repository. Combine, shorten or omit inapplicable sections. Keep unknowns as **to be confirmed** and separate plans from execution. AI may use prompts to spot omissions; whether to add material depends on its effect on research judgments, not on completing every field.

Template updated: 2026-10-16.

- Topic: to fill.
- Type: literature / theory / methods / reproduction / data and benchmarks / systems and applications.
- Current state and update date: to fill.
- Idea, observation or practical difficulty: to fill.
- Source, trigger and existing records: to fill.
- Preliminary question and expected value: to fill; novelty and feasibility await checking.
- Known constraints, owner, deadline and resource budget: to be confirmed.

# II. Literature review

- Search scope, keywords and inclusion/exclusion rules: to fill.
- Information cutoff and actual search dates: to fill.
- Evidence map, candidate approaches, disputes and gaps: to fill.
- Rationale for selected approach and baselines: to fill.

| **Source and location** | **Question and reading scope** | **Data and models** | **Algorithms** | **Metrics and conditions** | **Conclusions, limits and use here** |
| --- | --- | --- | --- | --- | --- |
| To fill | To fill | To fill | To fill | To fill | To fill; distinguish authors' conclusions from your judgment |

- Decision on the idea: solved / continue / adjust / defer; give the evidence.

# III. Questions and hypotheses

- Object, conditions and core question: to fill.
- Current limitation and closest work: to fill, with sources.
- Hypothesis, mechanism and falsifiable predictions: to fill; exploration may begin with questions.
- Contribution goal and boundaries: to fill.
- Criteria for support, refutation or insufficient evidence: to fill.
- Six-aspect connection: how literature supports the approach and how data, models, algorithms, metrics and experiments jointly answer the question.

# IV. Data

- Source, license/authorization, version and acquisition: to fill.
- Observation/sample unit, size, distribution, annotation and bias: to fill.
- Cleaning, preprocessing, exclusions and quality checks: to fill.
- Training/development/calibration/test splits and isolation unit: as applicable.
- Leakage, duplicates and contamination checks: to fill.
- Data, split and nonpublic-artifact locations: to fill.

# V. Models

- Inputs/outputs and structure or mathematical definition: to fill.
- Parameters, state, assumptions and boundaries: to fill.
- Initialization, weight source, version and license: as applicable.
- Simple baselines and closest work: to fill.
- Fixed parts and experimentally varied parts: to fill.

# VI. Algorithms

- Steps, equations or pseudocode: to fill.
- Specific baseline changes and mechanism hypothesis: to fill.
- Objective, hyperparameters, tuning and stopping: to fill.
- Randomness, state, cost and failure conditions: to fill.
- Reused implementation, code version and necessary tasks: to fill.
- Official, third-party and own implementation: source, version, location and adaptations to fill. Prefer official code; identify rewrites honestly.

# VII. Metrics

| **Role/name** | **Definition, unit, direction** | **Data and aggregation** | **Implementation and conditions** | **Criterion and basis** |
| --- | --- | --- | --- | --- |
| Primary: to fill | To fill | To fill | To fill | To be confirmed |
| Constraint/supplement: to fill | To fill | To fill | To fill | To be confirmed |

Locate key numbers; separate targets, authors' reported values and local measurements. Make originals/results and calculation conventions retrievable. Unexecuted results remain **unmeasured**.

# VIII. Experiments and execution order

| **Experiment/question** | **Method and control** | **Varied/fixed factors** | **Data, metrics, repetitions** | **Budget and interpretation** |
| --- | --- | --- | --- | --- |
| Baseline reproduction: to fill | Preferred official implementation and scope: to fill | Key conditions/differences: to fill | Representative results and files: to fill | Broad agreement and next steps: to be confirmed |
| Main comparison: to fill | To fill | To fill | To fill | Is the baseline credible enough: to be confirmed |
| Ablation/boundary/alternative: as needed | Key components: to fill | Comparable coverage, intervention location and protocol: to fill | Controls, metrics and source files: to fill | Simple starting point and subsequent changes: to fill |

## 1. Baseline reproduction and next steps

Prefer official code; check execution, then broadly reproduce major numbers/trends to establish a credible baseline. Fill as needed; no requirement to rerun the entire paper. Investigate major differences that affect judgment first.

| **Item** | **Contents** |
| --- | --- |
| Implementation | Preferred official repository, version/commit, license and adaptations. If absent, record the search and alternative |
| Scope | Weight evaluation / training / method-generation; actual planned work. Weight evaluation alone is not reproduced training |
| Conditions | Data, model, evaluation, coverage/location and comparison-relevant differences |
| Targets/differences | Representative results, files and broadly acceptable difference; refine tolerances when exact alignment matters |
| Actual results | Command, config, logs and files; unexecuted remains unmeasured |
| Judgment/follow-up | Broad agreement, gaps/causes, readiness for own comparison or need to investigate |

| **Goal and conditions** | **Target and basis** | **Author result and file** | **Local result and file** | **Difference/judgment** |
| --- | --- | --- | --- | --- |
| To fill | To fill; exact tolerance as needed | To fill | Unmeasured | To assess |

## 2. Ablation coverage and components

- Coverage, intervention location, protocol and differences: to fill; do not attribute different-coverage outcomes directly to algorithms.
- Training, affine, scaling, rotation or other key components chosen for the question: to fill; no exhaustive combination requirement.
- Simple starting point and changes: to fill; prefer minimal original-weight modifications, for example no scaling first.
- Source files/locations for each result, aggregate and difference: to fill; separate targets and measurements.

## 3. Execution and optimization

- Small checks, representative baseline and own comparison: to fill.
- Tuning versus final testing: to fill.
- Sample size, repetitions and statistical rationale: to fill.
- Confounders, alternatives and validity limits: to fill.
- Order, resources and continue/stop conditions: to fill.
- Alternative validation for literature/theory: as needed.

Add only methods used:

- DOE: factors/levels, randomization/blocking, interactions and confounding limits.
- OR: decision variables, objective, constraints and solution for selection/allocation/scheduling.
- BO: search space, initial trials, surrogate, acquisition, noise, budget and stopping.
- Bayesian experimental design: prior, observation model, information/decision utility and constraints.
- Adaptive search/independent validation: full trajectory, fair comparison budget, independent evaluation of selected settings and statistical handling.

# IX. Execution records and conclusions

Keep code, data, models, configs, environment, commands and result locations needed to verify formal runs; detail depends on the question. Locate key numbers; fill results after execution, never as if planned numbers were measured. Explain anomalous baselines and limits; do not directly call the method ineffective.

| **Experiment/validation** | **Actual execution/evidence** | **Results/variation** | **Hypothesis judgment/boundaries** |
| --- | --- | --- | --- |
| Pending execution | To fill | Unmeasured | To assess |

- Failures, negative results and post-hoc analysis: to fill.
- Supported, refuted and unresolved conclusions: to fill.
- Scope, reproduction limits and next steps: to fill.
- New ideas/questions: to fill, identifying literature/experimental evidence.

# X. GitHub use and plan changes

Use [research-plan](https://github.com/DreamSoul-AI/knowledge/tree/main/skills/research-plan) to help fill/review this template if useful. Installation/invocation: [Standard, Chapter VII](research_plan_standard_en.md).

Example: “Use `$research-plan` to fill this plan from the idea and evidence below. Check the six aspects, controls and budget. Keep unknowns to be confirmed. Planning only.”

The general template suggests docs/research-plan.md linked from README and retains existing code/config layout. RPipe instead retains [macro plans and Study-specific PLAN files](../../development/research.md), not duplicate plans. Exploration issues capture ideas; implementation/experiment issues link questions, hypotheses and validation; PRs link tasks and actual results. Formal runs preserve versions/artifacts and complete adaptive-search trajectories.

| **Date** | **Change/reason** | **Affected hypotheses, experiments or results** |
| --- | --- | --- |
| To fill | To fill | To fill |

References: list papers, official code and standards actually used. Cite key facts nearby with versions and necessary section locations.

[Standard, Chapter VIII](research_plan_standard_en.md) contains the workflow/method sources on ideas, AI evaluation, reproducibility, simulation optimization, BO and Bayesian design. Each project adds the method sources it actually uses.
