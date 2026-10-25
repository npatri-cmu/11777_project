# 11-777 Multimodal Reasoning: Research Idea 1

## Does the Model Look Where It Answers? Attention Grounding vs. Correctness in VLMs

This branch (`idea1`) contains the complete experimental code, analysis scripts, and midterm report draft for **Research Idea 1** of the 11-777 Midterm Project.

---

### 1. Repository Structure

```
├── rq1grounding/            # Core Python package
│   ├── attention.py         # Attention extraction and processing
│   ├── cli.py               # Argument parsers and utilities
│   ├── data.py              # V* benchmark and synthetic data loaders
│   ├── geometry.py          # Bounding box and patch intersection utilities
│   ├── hooks.py             # Residual norm recording and visual sink removal
│   ├── model.py             # Qwen3-VL-8B model loading and inference
│   ├── viz.py               # Matplotlib plotting helpers
│   └── vea.py               # Visual Evidence Alignment (VEA) probe
├── scripts/                 # Executable experiment scripts
│   ├── attention_vs_accuracy.py  # Core experiment: attention grounding vs. QA correctness
│   ├── p1_attention.py      # Probe 1: layerwise pointing & AUROC
│   ├── p2_knockout.py       # Probe 2: causal sliding-window knockout probe
│   ├── method_steer.py      # Method A: visual zoom & attention steering
│   ├── method_vea.py        # Method B: Visual Evidence Alignment benchmark
│   └── report.py            # Summary report generation
├── docs/                    # Detailed documentation and findings
│   ├── IDEA1_REPORT_OUTLINE.txt  # Writing outline and guidelines
│   └── FINDINGS.txt         # Full quantitative findings and statistical test logs
├── midterm_report/          # Midterm report LaTeX project (ICML 2026 format)
│   ├── report.tex           # Main paper source
│   ├── references.bib       # 14 verified bibliography citations
│   ├── report.pdf           # Compiled 8-page paper (with figures)
│   └── figs/                # Publication-grade figures
│       ├── accuracy_vs_attention.png  # Fig 1: Accuracy by peak location & AUROC terciles
│       ├── layers.png                 # Fig 2: Layerwise pointing trajectories
│       └── gallery.png                # Fig 3: 2x2 error decomposition qualitative gallery
├── requirements.txt         # Python dependencies
├── run_attention.sbatch     # SLURM script for core experiment
└── run_full.sbatch          # SLURM script for full pipeline
```

---

### 2. Key Empirical Findings

1. **Grounded attention yields high accuracy ($86.4\%$)**:
   - Across all 191 questions in $V^*$, when the model's text-to-image attention peak lands on the target object ($g=1$, $n=88$), accuracy reaches **$86.4\%$**.
   - When the peak lands elsewhere ($g=0$, $n=103$), accuracy drops to **$61.2\%$** (Fisher's exact test $p = 8.76 \times 10^{-5}$, odds ratio $3.90$).
2. **Robustness against Simpson's Paradox**:
   - Attribute questions ($n=115$): $88.3\%$ vs $61.8\%$ ($p = 0.0011$, $\text{OR} = 4.45$).
   - Relation questions ($n=76$): $82.1\%$ vs $60.4\%$ ($p = 0.0729$, $\text{OR} = 2.82$).
   - Category-stratified Cochran-Mantel-Haenszel (CMH) test: $\chi^2_{\text{MH}} = 14.25$, $p = 1.60 \times 10^{-4}$, pooled $\text{OR}_{\text{MH}} = 3.88$.
3. **Mechanistic Failure Taxonomy ($2 \times 2$)**:
   - $76.9\%$ ($40/52$) of errors exhibit attention peak outside the ground-truth box (localization bottleneck).
   - $23.1\%$ ($12/52$) of errors exhibit precise on-object grounding yet fail to predict correctly (multimodal binding/readout failure).
4. **Causal Reading Layers**:
   - Sliding-window knockout probe identified layers $[12, 13, 14, 15]$ as the critical reading window (log-prob drop $-0.309$, accuracy drop from $72.8\%$ to $63.4\%$).

---

### 3. How to Run

```bash
# 1. Environment setup
conda create -n rq1 python=3.11 -y
conda activate rq1
pip install -r requirements.txt

# 2. Run core experiment (single V100 GPU)
python scripts/attention_vs_accuracy.py \
    --model checkpoints/Qwen3-VL-8B-Instruct \
    --data_root data/vstar \
    --dtype float16 \
    --out outputs/attention_vs_accuracy

# 3. Or submit via SLURM
sbatch run_attention.sbatch
```
