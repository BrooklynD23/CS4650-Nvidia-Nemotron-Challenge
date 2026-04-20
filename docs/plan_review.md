# Expert Panel Review: plan_v0.1.md

**Document Reviewed**: `docs/plan_v0.1.md` (NVIDIA Nemotron Model Reasoning Challenge - Research & MVP Plan)
**Review Date**: 2026-04-19
**Overall Score**: **6.8 / 10**

---

## Panel Composition

| Expert | Domain | Methodology |
|--------|--------|-------------|
| **Karl Wiegers** | Requirements engineering | SMART criteria, testability analysis, stakeholder validation |
| **Michael Nygard** | Production systems & reliability | Failure mode analysis, operational excellence, circuit breaker patterns |
| **Gojko Adzic** | Specification by example | Given/When/Then scenarios, executable requirements, concrete examples |
| **Lisa Crispin** | Agile testing & quality | Risk-based testing, whole-team testing, quality attribute specification |

---

## Quality Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| **Requirements Clarity** | 7.5 / 10 | Good structure and phase decomposition; weak acceptance criteria and phase transition gates |
| **Completeness** | 7.0 / 10 | Broad coverage of all ML approaches; missing operational details and dependency mapping |
| **Testability** | 5.5 / 10 | Shallow verification plan with no concrete test cases or statistical significance |
| **Feasibility** | 6.0 / 10 | Unresolved blocking questions on compute/timeline; hardware now clarified |
| **Risk Management** | 4.5 / 10 | No failure modes, no rollback strategy, no cost caps, no OOM recovery |

---

## CRITICAL Issues (5)

### CR-1: No Phase Transition Criteria (Wiegers)
**Severity**: CRITICAL
**Location**: All phases (0-8)

The plan has 9 phases but no measurable gates defining when one phase is "done enough" to proceed to the next. When does baseline exploration end? What accuracy delta justifies investing compute in SFT vs. staying with prompting? Without gates, the student risks spending all time on early phases or skipping ahead prematurely.

**Recommendation**: Add a "Done When" subsection to each phase with specific, measurable criteria. Example: *Phase 1 is done when baseline accuracy is recorded on 3+ benchmarks AND a dummy submission is accepted on Kaggle.*

---

### CR-2: Blocking Open Questions Treated as Optional (Wiegers)
**Severity**: CRITICAL
**Location**: "Open Questions for Review" section (lines 480-486)

Open Questions #3-5 (compute budget, deadline, team size) are listed as afterthoughts, but they are **blocking requirements**. The plan's entire feasibility depends on their answers. A 48-hour H100 run costs ~$96 — without knowing if that's in budget, the plan is unexecutable. The timeline table is unanchored without a competition deadline.

**Recommendation**: Move these from "Open Questions" into a dedicated "Constraints & Resources" section at the top of the plan. Resolve them before any execution begins.

**Update**: User has since clarified compute resources (Colab Pro, RTX 3080, RTX 3060, HPC). Deadline and team size still need resolution.

---

### CR-3: No Rollback or Failure Recovery Strategy (Nygard)
**Severity**: CRITICAL
**Location**: Phases 4, 5, 6 (training phases)

What happens when a 48-hour training run produces a worse adapter? The plan defines no:
- Checkpoint frequency (save every N steps)
- Early stopping criteria (halt when val loss degrades)
- Kill conditions (what loss/reward values mean "abort")
- Recovery from Colab session preemption
- OOM recovery path (what to reduce when GPU memory overflows)

**Recommendation**: Add a "Failure Recovery" section defining checkpoint strategy, early stopping rules, Colab preemption recovery (Google Drive checkpoints), and an OOM batch size reduction ladder (256 -> 128 -> 64 -> 32).

---

### CR-4: No Concrete Smoke Test (Adzic)
**Severity**: CRITICAL
**Location**: Phase 1 and Verification Plan

The plan mentions "competition benchmark questions" but never shows a single example question, expected reasoning trace, or programmatic verification. The student has no concrete way to know if their pipeline actually works end-to-end without a tangible test case.

**Recommendation**: Add a smoke test subsection in Phase 1 with:
- One specific math problem (e.g., "What is 2^10 mod 7?")
- Expected reasoning trace with `<think>` tags
- Expected boxed answer: `\boxed{2}`
- Python verification script that extracts and checks the answer

---

### CR-5: No Validation Set Strategy (Crispin)
**Severity**: CRITICAL
**Location**: Phase 3 (Data Curation) and Phase 7 (Evaluation)

The plan mentions a "95/5 train/val split" for the Puzzle dataset but never defines how validation is used during training. There is no:
- Early stopping on validation loss
- Held-out test set reserved before any training
- Strategy for detecting overfitting to known benchmarks

The competition uses a "novel benchmark" — optimizing for MATH500 and AIME25 without testing generalization is risky.

**Recommendation**: Reserve 200 diverse problems as a held-out validation set before any training begins. Define 20 "golden" regression test problems that must always be answered correctly. Use pass@k metrics with confidence intervals.

---

## MAJOR Issues (9)

### MJ-1: No Dependency Mapping Between Phases (Wiegers)
**Location**: Execution Timeline

Phase 6 (GRPO) requires a trained SFT model from Phase 4, but this dependency is never explicitly stated. The plan reads as if phases can be executed independently when they cannot.

**Recommendation**: Add a dependency graph or explicit "Requires" field to each phase.

---

### MJ-2: "Explore ALL Approaches" vs. Resource Constraints (Wiegers)
**Location**: Context section and throughout

The stated goal to explore ALL approaches conflicts with finite compute, time, and energy. No prioritization framework exists for cutting scope if resources run short.

**Recommendation**: Add a scope-cutting priority list. Suggested drop order: Phase 6 (GRPO) -> Phase 5 (Synthetic Data) -> Phase 7.3 (Adapter Merging). Minimum viable submission: Phase 0-1 + Phase 3-4 (SFT only).

---

### MJ-3: Colab Session Limits Not Addressed (Nygard)
**Location**: Phase 4.4

Free/Pro Colab has session limits. Training runs exceeding session length will be terminated without warning. The plan assumes uninterrupted Colab availability.

**Recommendation**: Use Google Drive for checkpoint persistence. Design training scripts to resume from latest checkpoint. For long runs, prefer HPC cluster over Colab.

---

### MJ-4: No API Cost Caps for Synthetic Data Generation (Nygard)
**Location**: Phase 5

Phase 5 depends on external API calls to Claude/GPT-4/DeepSeek for synthetic data generation. No rate limits, cost caps, or fallback strategies are defined. A student could accidentally incur hundreds of dollars in API charges.

**Recommendation**: Set a hard cost cap ($20-50 per generation run). Implement dry-run mode that estimates cost before execution. Prefer open-weight models (DeepSeek-R1 on HuggingFace) over paid APIs where possible.

---

### MJ-5: Storage Requirements Unspecified (Nygard)
**Location**: Phase 0

The Llama-Nemotron dataset is 130GB. Combined with model weights (~16GB), adapters, and processed data, total storage could exceed 200GB. WSL2 on Windows often has constrained disk allocation.

**Recommendation**: Add storage requirements to Phase 0. Calculate total: ~150GB datasets + ~16GB model + ~5GB adapters/checkpoints. Verify WSL2 disk allocation or plan to use external storage.

---

### MJ-6: Ungrounded "Expected Gain" Estimates (Adzic)
**Location**: Phase 2, Section 2.2

The "Expected Gain" column lists values like "+5-10%", "+20%", "+15-20%" sourced from general LLM literature, not measured on Nemotron-3-Nano-4B. Presenting these as expected outcomes sets false expectations.

**Recommendation**: Rename column to "Literature Estimate (unvalidated on this model)". Add footnote explaining these are from research on larger models and may not transfer.

---

### MJ-7: Regex Bug in GRPO Reward Function (Adzic)
**Location**: Phase 6.2, `accuracy_reward` function (line ~346)

The regex `\\boxed\{(.+?)\}` uses a lazy match that fails on nested braces. For answers like `\boxed{\frac{1}{2}}`, it would extract `\frac{1` instead of `\frac{1}{2}`. This would silently produce incorrect reward signals during RL training, potentially training the model to avoid complex answers.

**Recommendation**: Replace with a brace-aware regex:
```python
# Handles one level of nesting (covers most math expressions)
match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', completion)
```

---

### MJ-8: No Regression Testing Across Benchmarks (Crispin)
**Location**: Phase 7

When applying GRPO after SFT, math accuracy may improve while science reasoning (GPQA) degrades. The plan has no strategy for detecting or handling capability regressions across different benchmark categories.

**Recommendation**: Evaluate on ALL benchmarks after each training phase. Flag any regression > 1% as requiring investigation. Define acceptable trade-off thresholds (e.g., "GPQA can drop 2% if AIME25 gains 5%").

---

### MJ-9: No Statistical Significance Testing (Crispin)
**Location**: Phase 7

AIME25 has ~30 problems. A 0.5% "improvement" is less than one question and is likely noise. Without confidence intervals, the student cannot distinguish real gains from random variance.

**Recommendation**: Use pass@k metrics (k=1, 5, 10) with multiple evaluation runs. Report confidence intervals. Define minimum detectable effect size for each benchmark based on eval set size.

---

## MINOR Issues (6)

| ID | Issue | Location | Recommendation |
|----|-------|----------|----------------|
| MN-1 | Loose version pinning (`torch>=2.2.0`) | Phase 0.2 | Pin to tested versions; Nemotron may require specific CUDA/torch combos |
| MN-2 | No reproducibility requirements | Throughout | Set random seeds (`torch.manual_seed(42)`, `transformers.set_seed(42)`) |
| MN-3 | WandB listed but not configured | Phase 0.2 | Add WandB project name, run naming convention, alert thresholds |
| MN-4 | No model weight backup strategy | Phase 4-6 | Tag and backup adapters to cloud storage after each successful run |
| MN-5 | Evaluation reproducibility undefined | Phase 7 | Use `temperature=0` for deterministic evaluation; document seed |
| MN-6 | Adapter Merging hand-waved | Phase 7.3 | Define concrete method (weight averaging, DARE, TIES) or defer explicitly |

---

## Strengths

The panel identified several notable strengths:

1. **Comprehensive breadth**: All major ML approaches covered (prompting, data curation, synthetic data, SFT, RL)
2. **Executable code examples**: Code snippets in Phases 1, 4, and 6 are concrete and copy-pasteable
3. **Well-sourced hyperparameters**: Training configs cite NVIDIA blog with specific values
4. **Framework comparison table**: Phase 4.3 provides useful decision matrix for framework selection
5. **Resource collection**: Key Resources table is comprehensive and well-organized
6. **Data format specification**: Phase 5.3 JSON example clearly shows expected structure
7. **Curriculum design ratios**: Phase 3.3 specifies testable ratios (60/20/15/5 domain mix)

---

## Prioritized Improvement Roadmap

### Immediate (before any execution)
1. Resolve blocking questions: compute budget (done), deadline, team size
2. Add phase transition gates with measurable criteria
3. Create one concrete smoke test (problem + expected output + verification)
4. Reserve held-out validation set before any training
5. Fix boxed regex bug in GRPO reward function

### Short-Term (during Phase 0-1)
6. Add failure recovery section (checkpoints, early stopping, OOM ladder)
7. Set API cost caps for synthetic data generation
8. Calculate and verify storage requirements
9. Relabel "Expected Gain" as "Literature Estimate"
10. Define scope-cutting priority order

### Long-Term (during Phase 4+)
11. Build regression test suite (20 golden problems)
12. Implement statistical significance framework (pass@k, confidence intervals)
13. Evaluate on out-of-distribution reasoning tasks
14. Define adapter versioning and backup strategy

---

## Expert Consensus

> *"The plan is well-researched and comprehensive in **breadth** but shallow in **depth** on execution details. The strongest sections are the code examples and hyperparameter tables — concrete and actionable. The weakest sections are verification, risk management, and phase transition criteria. For a capstone learning project, the multi-approach exploration is excellent; for competition performance, focus should narrow after Phase 2 baseline results reveal which approaches yield actual gains on this specific model."*

---

*Review conducted by Expert Specification Panel using Wiegers (Requirements), Nygard (Operations), Adzic (Testability), and Crispin (Testing) methodologies.*
