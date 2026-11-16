# Notebook sections

These files follow the numbered sections in
`legacy-code/bdb_wr_route_scaffold_refactored.ipynb`. Markdown is preserved as
comments, and `# %%` markers separate the original cells. The original notebook
is unchanged. Section 14 contains only checklist comments.

Run from the repository root:

```sh
python code-section/run_sections.py
python code-section/run_sections.py --through 7
```

The runner imports sections in numeric order. Every section explicitly imports its
own libraries and the variables/functions it needs from earlier sections. Imports
are cached by Python, so dependencies execute once per process. Importing a data
section runs its top-level loading/audit work; neural training and experiments
still require the existing explicit calls described in the TODOs.

You can also run a section directly, for example:

```sh
python code-section/02load_data.py
```

The existing numbered filenames are retained. Because Python's `from ... import`
syntax cannot use names starting with digits, sections use `import_module` and
explicit named bindings for their dependencies. Library imports are local to each
section. `pipeline_display.py` provides plain-text output for audit tables.

`DATA_DIR` resolves to the repository's `data` folder independently of the working
directory. Install the project requirements before running. Existing scaffold
TODOs and VERIFY comments remain, including unfinished baseline features and
neural experiment calls.

## File order

1. `01setup.py`
2. `02load_data.py`
3. `03data_audit.py`
4. `04cleaning_and_kinematics.py`
5. `05rep_tensors.py`
6. `06targets.py`
7. `07baselines.py`
8. `08dataset_and_batching.py`
9. `09model.py`
10. `10training.py`
11. `11cross_validation.py`
12. `12variations_and_ablations.py`
13. `13interpretation.py`
14. `14wrap_up_checklist.py`

## Original notebook overview

# DS340 Final Project Scaffold: Combine Drill Sequences -> NFL WR Route Performance

**Question.** Does a learned representation of a WR's Combine drill movement (LSTM encoders + pooling) predict NFL route performance (e.g. separation at pass forward) *beyond* the official Combine numbers and draft position?

**Pipeline.**
```
reps in each drill family -> [LSTM encoder(s)] -> one vector per rep
        -> [pooling: mean | max | learned-weight | self-attention] -> player vector
        -> (+ optional attributes) -> [MLP head] -> one prediction per route family
```

**Feature sets compared (same downstream evaluation for all):**
- **A**: official Combine numbers + height/weight (+ draft pick as a variation)
- **B**: hand-built kinematics from the tracking data (top speed, braking, turn sharpness, ...)
- **C**: learned (LSTM / attention model)

**How the sections map to your milestones**
- Sections 1-7 (setup, audit, cleaning, tensors, targets, baselines A/B) = your **mid-project TA milestone** (aim for ~Nov 17).
- Sections 8-11 = the neural model and grouped cross-validation.
- Sections 12-13 = variations/ablations and interpretation (figures for the paper + lightning talk).

**Conventions.**
- Anything marked `VERIFY` is a column name or value I inferred from public descriptions of the data. Check it against your actual files.
- Anything marked `TODO` is yours to complete.
- **AI statement (course requirement):** this scaffold was generated with Claude. The syllabus requires you to disclose AI use and comment AI-written code. Keep the `# [AI-SCAFFOLD]` tags on lines you did not write yourselves, and delete the tag on code you rewrite. Update your written AI statement to match what you actually kept.

---
**Decisions locked in so far (from the data audit; revisit only with evidence)**
- **Population:** WRs only, using premade `wr_only_*.csv` files (106 WRs have Combine tracking).
- **Target:** average `separation_at_pass_forward` per route family (`vertical`, `break`, `short`), adjusted for route type. FLAT and SCREEN are excluded.
- **Min routes per family:** `MIN_ROUTES = 20` (split-half reliability plateaus there; 15 and 25 are sensitivity checks). Note: this is a **count of routes**, not a distance.
- **Combine drills in the encoder:** the WR route drills (after merging name variants and mirroring left-side reps) plus the 40-yard dash. Left out: 3-cone and shuttle tracking (too sparse), ball tracking (about a third of attempts), `GAUNTLET_DRILL`, `OVER_SHOULDER_ADJUST`, and `CIRCUS` (2025 class only).
- **Channels:** `speed`, `acc_tan` (signed, along the path), `acc_cent` (signed, sideways). The sensor's `a` is the total acceleration magnitude, so it adds no new information. Raw channels are kept as an ablation.
