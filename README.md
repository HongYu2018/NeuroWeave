# NeuroWeave

**Sparse neural graphs, neuronal complexity, and continual visual recognition without replaying old images**

NeuroWeave is a research implementation for studying how a small neural system can extend its recognition vocabulary while preserving useful computations. Rather than assuming that better learning always requires deeper predefined layers, more neurons, or more complex individual neurons, the project examines three components separately:

1. **The representational core:** a sparsely connected graph of nonlinear computational neurons, including both multi-compartment dendritic neurons and conventional point-neuron controls.
2. **The circuit adaptation mechanism:** optional local paths and soma-integrated branches that can be proposed, accepted, ablated, or reused when new classes are introduced.
3. **The recognition memory:** compact class-indexed feature prototypes that preserve access to earlier classes without replaying their raw training images.

This repository accompanies the research manuscript **“What Enables Neural Networks to Learn Without Replaying the Past? Neuron Complexity, Readout Interference and Sparse Circuit Adaptation”**, intended for the journal *Neural Networks*. The paper and supporting figures are included under [`paper/`](paper/). This public research release is not a general-purpose AI agent and does not establish biological equivalence with the human brain.

![NeuroWeave experimental architecture](paper/figures/F1_architecture.png)

*Figure: experimental decomposition of the shared sparse neural representation, incremental class enrolment, protected class memories, and inference-time competition. A vector version is available at [`paper/figures/F1_architecture.pdf`](paper/figures/F1_architecture.pdf).*

---

## Research questions and motivation

Consider a visual model trained to recognise a set of components. A previously unseen component is then introduced using only a few labelled images. A useful continual learner must classify **all** known components, not only the most recently introduced ones. This poses two different problems: the representation of earlier classes might change, and the decision mechanism might stop choosing older labels even when the representation is unchanged.

Our experiments investigate:

- **Stable representation and expandable capability.** Can a fixed population of sparse computational neurons support an expanding vocabulary of recognised classes through compact class memories?
- **Neuron versus circuit complexity.** When do nonlinear dendritic compartments produce a measurable benefit over simpler point neurons with comparable parameter budgets?
- **Recognition accessibility.** Can old-class predictions fail even when the underlying feature graph is frozen, and can a different readout preserve access to earlier classes?
- **Zero raw-image replay.** What happens when new categories must be learned without retraining on stored old-class images, and what knowledge must instead remain in feature or parameter statistics?

**Interpretation:** the code is designed to examine these questions, not to guarantee positive answers. In the available tests, point-neuron models are highly competitive; newly developed branches sometimes help and sometimes do not. The zero-replay audit shows a particularly clear distinction between retaining a representation and retaining correct access to old categories. More complicated neuron-level processing is not automatically better in these particular settings.

### Selected recorded observations

| Study | Recorded result | What it does and does not show |
| --- | --- | --- |
| COIL-100 class expansion (one seed, 36 classes) | Fixed dendritic prototypes: **83.80%** overall; fixed point prototypes: **85.11%**; parameter-matched adaptive point model: **86.27%** | Feasibility of expanding the class vocabulary with a fixed neural core and compact prototypes. Not evidence of general superiority of either neuron type. |
| Optical handwritten digits (five seeds, 10 classes) | Fixed dendritic prototypes: **73.46% ± 3.06**; approximately parameter-matched fixed point prototypes: **76.12% ± 2.07** | A second visual domain using the protected-core code; accuracy varies by seed and digit class order. |
| COIL-100 zero-replay readout audit (one seed) | Frozen graph + expanding trainable head: **5.38%** accuracy on old classes at the final stage; frozen dendritic prototypes: **82.55%** on old classes | Severe recognition interference is possible without changing the frozen feature graph. The methods differ in memory and readout, so this is not a single-factor architecture comparison. |
| Historical CIFAR-10 graph construction (two-seed aggregate) | Guided graph: **60.63%** ten-class accuracy; random graph: **60.37%** | A different CNN-to-graph protocol, with a frozen CNN encoder and task-specific readouts. **The original training source and per-seed logs are not included here.** |

All these numbers refer to the **archived recorded experiments**, not guaranteed outcomes of a new run on different hardware. Recompute the archived values with `scripts/verify_recorded_results.py` before citing them. See [`results/archived/DATA_PROVENANCE.md`](results/archived/DATA_PROVENANCE.md) and [`results/archived/CIFAR10_PROVENANCE.md`](results/archived/CIFAR10_PROVENANCE.md).

---

## Quick start: Windows (PowerShell)

**Start here if this is your first time running the code.** The project requires no Docker installation or PowerShell virtual-environment activation. Commands below assume you have extracted or cloned this repository and opened PowerShell in its **root directory**, where `README.md`, `requirements.txt`, `scripts/`, and `experiments/` are visible.

### 1. Check Python

Python **3.11** is recommended. Use a recent version of Git if you intend to publish or contribute code.

```powershell
py -3.11 --version
```

If this reports that Python 3.11 is unavailable, install it from <https://www.python.org/downloads/> and open a new PowerShell window. An installed Python with the `python` command can also be used in place of `py -3.11`.

### 2. Create one virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

The environment lives inside the repository, and you can call its Python executable directly. This avoids the common PowerShell `Activate.ps1` execution-policy error.

### 3. Install dependencies

For **CPU-based verification**, use:

```powershell
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an **NVIDIA GPU**, install the PyTorch build compatible with your GPU driver and CUDA runtime from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing `requirements.txt`. Do not assume that an arbitrary `pip install torch` command will install a CUDA-enabled wheel.

Check the environment:

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

CUDA is optional for the small smoke test and unit tests. The COIL-100 experiments are more practical on an NVIDIA GPU.

### 4. Run the automated tests

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_TESTS_WINDOWS.ps1
```

The script tests **both** experimental codebases and verifies selected numbers from the archived CSV files. In the release-preparation environment, the protected-core suite passed **25 tests** (two CUDA-only skips), and the readout-audit suite passed **51 tests** (four CUDA-only skips). These counts describe that environment, not a promise that all supported systems behave identically.

### 5. Run a small experiment without downloading data

A synthetic class-incremental experiment is a useful smoke test for all eight protected-core comparison methods. Run it with the `neuro_weave` module inside the **protected-core working directory**:

```powershell
Push-Location .\experiments\protected_core
try {
    & ..\..\.venv\Scripts\python.exe -m neuro_weave.assemblies_v21_run `
        --dataset synthetic --source-classes 6 --novel-classes 6 `
        --increment 2 --initial-shots 8 --new-shots 4 `
        --max-val 4 --image-size 8 --neurons 8 `
        --epochs 2 --steps 3 --branch-fanin 3 `
        --output results/readme_smoke --cpu
} finally {
    Pop-Location
}
```

This command was executed successfully as part of the README preparation. It uses **small, synthetic inputs**, not COIL-100, and is intended to verify operation rather than reproduce a paper result. Expect `summary.csv`, `configuration.json`, `per_class_accuracy.csv`, `development_events.csv`, `training_summary.json`, topology JSONs and model checkpoints (`.pt`) in:

`experiments/protected_core/results/readme_smoke/`

### 6. Run the main COIL-100 study

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COIL_WINDOWS.ps1
```

The driver attempts to acquire the public COIL-100 data on first use, or reuses images already present at:

`experiments/protected_core/data/coil100/images/`

Expected result folder:

`experiments/protected_core/results/coil_seed11/`

For new users, **run the tests and synthetic smoke test before this dataset experiment**.

---

## Linux / macOS setup

Run from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
```

For CUDA-based Linux use, replace the generic PyTorch install with the appropriate wheel from [pytorch.org](https://pytorch.org/get-started/locally/). Apple Silicon acceleration is not an explicitly validated configuration in this release.

Test both implementations and verify the records:

```bash
(cd experiments/protected_core && ../../.venv/bin/python -m pytest -q tests)
(cd experiments/readout_audit && ../../.venv/bin/python -m pytest -q tests)
.venv/bin/python scripts/verify_recorded_results.py
```

CPU synthetic smoke test:

```bash
(
  cd experiments/protected_core
  ../../.venv/bin/python -m neuro_weave.assemblies_v21_run \
    --dataset synthetic --source-classes 6 --novel-classes 6 \
    --increment 2 --initial-shots 8 --new-shots 4 \
    --max-val 4 --image-size 8 --neurons 8 --epochs 2 \
    --steps 3 --branch-fanin 3 --output results/readme_smoke --cpu
)
```

For the full COIL-100 or five-seed digit settings, see the independent commands below. Always run each module from its own experiment directory.

---

## Directory map: where the research lives

```text
NeuroWeave/
├── README.md                            <- start here
├── requirements.txt                     <- shared Python dependencies
├── experiments/
│   ├── protected_core/
│   │   ├── neuro_weave/
│   │   │   ├── model.py                 <- sparse DAG, dendritic and point neurons
│   │   │   ├── assemblies_v21.py        <- protected prototypes, branch/path enrolment
│   │   │   ├── assemblies_v21_run.py    <- class-incremental experiment driver
│   │   │   ├── continual.py             <- training stream and class schedule helpers
│   │   │   ├── data.py                  <- synthetic, COIL-100, digits, Pets, CIFAR-10
│   │   │   └── ...
│   │   ├── tests/
│   │   └── requirements.txt
│   └── readout_audit/
│       ├── neuro_weave/
│       │   ├── continual_v166.py        <- zero-replay audit experiment driver
│       │   ├── paper_controls.py        <- importance/prototype/growth diagnostics
│       │   ├── model.py                 <- graph models and readout controls
│       │   └── ...
│       ├── scripts/                     <- historical experiment launchers
│       └── tests/
├── scripts/
│   ├── RUN_TESTS_WINDOWS.ps1
│   ├── RUN_COIL_WINDOWS.ps1
│   ├── RUN_DIGITS_5SEEDS_WINDOWS.ps1
│   ├── RUN_ZERO_REPLAY_AUDIT_WINDOWS.ps1
│   ├── RUN_CIFAR10_PROTECTED_CORE_WINDOWS.ps1
│   └── verify_recorded_results.py
├── results/
│   └── archived/                        <- recorded measurements (no raw image data)
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── make_figures.py
│   ├── figures/
│   └── reproduced/
└── .github/workflows/tests.yml
```

**Important import rule:** The two experimental folders both contain a Python package called `neuro_weave`. They are **separate implementations** and should not be placed together on `PYTHONPATH`. Every script provided in the repository changes into the correct working directory before invoking Python. If you run a module yourself, `cd` into the appropriate `experiments/...` folder first. Neither implementation needs to be installed with `pip install -e .`.

---

## Understanding the architecture

### Representation training

A sparse directed acyclic graph is built with a fixed population of neurons, generally **32** in the main reported experiments. Unlike a conventional multilayer perceptron, its hidden units are connected through an ordered graph with no prescribed stack of fully connected hidden layers. Source-class images are used to train that representation.

Neurons can be instantiated as:

- **Dendritic units**, whose local nonlinear branches contribute to a somatic computation, including an interaction between branches.
- **Point-neuron controls**, which use a simpler nonlinear somatic transform with comparable source connectivity.
- **Parameter-matched point-neuron controls**, using a slightly different fan-in to approximately match the base parameter count. Parameter matching should **not** be interpreted as equality of FLOPs, latency or energy.

### Protected class prototypes

Once the shared representation has been trained, the protected-core experiments freeze it for subsequent class enrolment. A new class is introduced using four labelled training supports. Class-level feature prototypes and an associated path are stored. Earlier class-specific memories and paths remain unchanged.

All previously enrolled classes compete in the **same decision space** at inference. The evaluation does not supply a task identifier to restrict the candidate class set.

### Optional local circuit adaptation

An adaptive condition may propose a new soma-integrated local branch or reuse a previously available path. The branch parameters are fitted using **two** novel-class supports, while the other two supports are used for selection. These supports all come from the class's **training allocation**. Neither test queries nor old test images are used to accept a candidate branch.

Path acceptance and reuse counts are logged. Post-training branch ablation tests whether accepted branches have a measurable effect on prediction. **A counted reuse event alone does not demonstrate beneficial transfer to a later class.**

### A different experiment: readout-access interference

The zero-replay audit uses an expanding classifier instead of the protected class-prototype mechanism. It asks: can the score for a previously learned category become ineffective even if the underlying feature graph is frozen?

The audit separately measures accuracy on old classes when predictions are restricted to old labels, and accuracy on those same examples when all new labels compete. It also records the fraction of old examples assigned to new categories and representation drift between checkpoints. Because its training and readout differ from the protected-core study, **do not treat the two experimental families as a single controlled ablation**.

---

## Reproducing the reported experiments

### A. Verify archived results without training

This should be the first reproducibility command:

**Windows**

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_recorded_results.py
```

**Linux / macOS**

```bash
.venv/bin/python scripts/verify_recorded_results.py
```

The script checks selected row-level accuracy values and five-seed digit statistics against their archived CSV files. It **does not rerun network training**. These are different levels of reproducibility, and both are documented.

### B. COIL-100 protected-core experiment

Run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COIL_WINDOWS.ps1
```

Or on Linux / macOS, from the repository root:

```bash
(
  cd experiments/protected_core
  ../../.venv/bin/python -m neuro_weave.assemblies_v21_run \
    --dataset coil100 --seed 11 --source-classes 24 \
    --novel-classes 12 --increment 4 \
    --initial-shots 4 --new-shots 4 --max-val 18 \
    --image-size 16 --neurons 32 --epochs 30 --steps 20 \
    --branch-fanin 8 --temperature 12 --branch-gain 0.35 \
    --reuse 3 --accept-threshold 0.01 \
    --data-root data --output results/coil_seed11
)
```

Configuration: 24 initial classes, then 4 + 4 + 4 classes, with **four labelled training examples per new class**. COIL-100 viewpoints are partitioned by angle (see `data.py`), and the labelled class order is shuffled deterministically using seed 11.

Expected directory: `experiments/protected_core/results/coil_seed11/`.

The key files are `summary.csv`, `per_class_accuracy.csv`, `development_events.csv`, `configuration.json`, `training_summary.json`, `.pt` checkpoints and individual topology files. The initial training and incremental-stage results use a single class vocabulary with full-class prediction at evaluation time.

### C. Handwritten digits, five seeds

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_DIGITS_5SEEDS_WINDOWS.ps1
```

**Linux / macOS**

```bash
(
  cd experiments/protected_core
  for seed in 11 12 13 14 15; do
    ../../.venv/bin/python -m neuro_weave.assemblies_v21_run \
      --dataset digits --seed "$seed" --source-classes 6 --novel-classes 4 \
      --increment 2 --initial-shots 4 --new-shots 4 --max-val 12 \
      --image-size 16 --neurons 32 --epochs 30 --steps 20 \
      --branch-fanin 8 --temperature 12 --branch-gain 0.35 \
      --reuse 3 --accept-threshold 0.01 \
      --data-root data --output "results/digits_seed${seed}" --cpu
  done
)
```

This dataset is bundled with scikit-learn through `sklearn.datasets.load_digits`. It comprises **1,797** images of digits 0 through 9, originally at 8×8 pixels. The research loader uses a class-stratified split, source classes, and four labelled examples for each new class. See `data.py` for the exact item selection.

Output folders are `experiments/protected_core/results/digits_seed11/` through `digits_seed15/`. The archived study reports means and sample standard deviations over seeds 11–15.

### D. COIL-100 zero-replay readout audit

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_ZERO_REPLAY_AUDIT_WINDOWS.ps1
```

**Linux / macOS**

```bash
(
  cd experiments/readout_audit
  ../../.venv/bin/python -m neuro_weave.continual_v166 \
    --dataset coil100 --seed 11 --source-classes 24 --novel-classes 12 \
    --increment 4 --fewshot 4 --max-train 4 --max-val 18 \
    --memory 0 --neurons 32 --branches 4 --fanin 16 \
    --initial-synapses 0.55 --epochs 20 --adapt-epochs 22 \
    --branch-trial-epochs 3 --output results/zero_replay_audit_seed11
)
```

The audit uses **zero replayed old training images** (`--memory 0`). However, it may retain feature prototypes and online importance statistics. This is not a completely memoryless system.

Default zero-replay branch selection uses **only current new-class training supports** and excludes historical validation images. Avoid enabling `--allow-historical-validation` if you want to preserve this restriction.

Output folder: `experiments/readout_audit/results/zero_replay_audit_seed11/`. The experiment records `continual_metrics.csv`, `old_new_competition_diagnostics.csv`, `experience_accuracy_matrix.csv`, per-class and development-event records, checkpoints, and a run summary.

### E. CIFAR-10: distinguish the two protocols

The repository contains a **new official-data loader and a runnable protected-core CIFAR-10 protocol**:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_CIFAR10_PROTECTED_CORE_WINDOWS.ps1
```

It downloads the official CIFAR-10 Python archive from the University of Toronto and checks its published MD5. The experiment enrols classes incrementally using only four labelled examples per new class and evaluates a selected held-out subset of the official test data.

**This command was not run to produce the paper's historical CIFAR-10 table.** That table comes from a different CNN-to-graph architecture using a frozen 48-dimensional CNN encoder and task-specific outputs. Its original source code and per-seed logs are not present in this repository. Only the transcribed two-seed aggregate and a provenance explanation are available. These protocols **must not be described as equivalent or merged in statistical analysis**.

See [`CIFAR10_PROVENANCE.md`](results/archived/CIFAR10_PROVENANCE.md) for the exact boundary of the available evidence.

---

## Datasets and where to place them

| Dataset | How it is obtained | Where the code expects it | Purpose |
| --- | --- | --- | --- |
| Synthetic images | Created in memory | No files needed | Setup / mechanism smoke test only |
| COIL-100 | Official public dataset, downloaded by loader | `experiments/protected_core/data/coil100/images/` or `experiments/readout_audit/data/coil100/images/` | Object identity recognition across viewpoints |
| Optical digits | `scikit-learn` built-in `load_digits` | No manual files needed | Independent class-incremental digit study |
| CIFAR-10 | [University of Toronto archive](https://www.cs.toronto.edu/~kriz/cifar.html) with MD5 check | `experiments/protected_core/data/downloads/cifar-10-python.tar.gz` | Optional **unexecuted** protected-core protocol |
| Oxford-IIIT Pet | Public Oxford dataset loader | `experiments/protected_core/data/pets/` or respective audit project | Optional pilot; not a main reported result here |

If COIL-100 was downloaded for an earlier NeuroWeave project, copy the existing `coil100/images` folder into the relevant experiment's `data/` directory. It should contain approximately **7,200 PNG images** named like `obj1__0.png`. The audit implementation has its own data root, so either copy the images into its `data/coil100/images` folder or supply a shared `--data-root` through a manual invocation. Data are **not bundled** with this repository and remain subject to their original licences and terms.

The COIL-100 split in the protected-core loader assigns angles with `angle % 20 == 0` to training, `angle % 20 == 10` to validation, and other angles to query evaluation. This viewpoint split is **research-defined**, not an official standard class-incremental benchmark split. The digits split uses 64 train, 28 validation and 48 query candidates per class before the study's few-shot restrictions. The CIFAR-10 loader derives fit and validation examples from official training batches and query examples from the official test batch.

---

## Understanding the output files

After a protected-core experiment, the most useful outputs are:

| File | Interpretation |
| --- | --- |
| `configuration.json` | Arguments, dataset class IDs, selected compute device; essential for reproducibility |
| `summary.csv` | Overall, old-class and latest-class accuracy at each stage for each model |
| `per_class_accuracy.csv` | Accuracy for each original category at each stage |
| `development_events.csv` | Candidate branch acceptance, reuse records and selection events |
| `training_summary.json` | Initial model training durations; timing depends on system hardware |
| `*_topology.json` | Accepted computation paths and structural metadata |
| `*.pt` | Research checkpoints for the learned representation and enrolled memory |

`summary.csv` includes `fixed_prototype_ablation_delta`, defined as the difference between the model's accuracy with branches active and its **own** accuracy with branches disabled. That is *not* identical to the improvement over an independently trained fixed-prototype baseline. A positive ablation result alone does not establish beneficial transfer across classes.

For readout-audit experiments, also inspect:

| Field | Meaning |
| --- | --- |
| `old_only_accuracy` | Old-class accuracy when prediction candidates are restricted to old labels |
| `old_full_accuracy` | Accuracy on those same old-class examples when **all** currently learned labels compete |
| `old_predictions_as_new_fraction` | Fraction of old-class examples labelled as a newly introduced category |
| `graph_representation_RMSE` | Feature drift between checkpoints; zero for a completely frozen graph |
| `continual_metrics.csv` | Overall, old, new and forgetting metrics for all tested readout/training policies |

For the precise columns and units, inspect the CSV headers and their writer functions in `assemblies_v21_run.py` and `continual_v166.py`. Percentages in manuscript tables are often derived from CSV values stored as proportions in `[0,1]`.

---

## Regenerate the paper and confirm claims

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_recorded_results.py
.\.venv\Scripts\python.exe .\paper\make_figures.py
```

The first command checks the recorded headline results. The second regenerates the plots from archived CSVs and produces vector PDFs in `paper/figures/`. For the architecture drawing specifically:

```powershell
.\.venv\Scripts\python.exe .\paper\figures\generate_architecture.py
```

The manuscript is in `paper/main.tex` with references in `paper/references.bib`. To build it locally, install a TeX distribution with `pdflatex` and `bibtex`, and then execute from the `paper/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

You may also upload the separate journal Overleaf project ZIP directly to [Overleaf](https://www.overleaf.com/). The reproducibility scripts **do not** train the models when regenerating plots from the archive.

---

## Reproducibility, limitations, and scientific integrity

**Levels of reproducibility included here:**

1. **Code execution:** unit and integration tests check available model functionality and dataset interfaces.
2. **Recorded-result verification:** archived CSVs can be independently parsed to reproduce the reported summary numbers.
3. **Training reruns:** the protected-core COIL-100 and digits implementations, plus the COIL zero-replay readout audit, can be rerun from their supplied source under the specified protocols. Results may vary with hardware, package versions, and nondeterministic training kernels.
4. **Historical CIFAR-10:** only aggregate records and protocol information are available. **Exact training reproduction is not possible from this release** until its original CNN-to-graph implementation and seed logs are recovered.

Additional qualifications:

- A constant neuron population is **not** a constant-memory system: class prototypes and some path parameters are added as the set of known classes grows.
- No old *images* are replayed in the zero-replay protocols, but feature prototypes and parameter-importance statistics may be stored.
- The protected-core representation and previous class paths are frozen **by design**. The system does not demonstrate self-learned global synaptic consolidation.
- The architecture uses a sparse, ordered directed acyclic graph; no experiment proves universally optimal self-organisation.
- Point-neuron controls are approximately parameter-matched, not strictly matched by FLOPs or measured energy.
- Some learned branches influence predictions; the same model with branches ablated can perform worse or better. Reuse counts alone are not proof of positive cross-task transfer.
- COIL-100 has one principal protected-core seed; digits has five seeds; the zero-replay audit is a separate protocol. **Do not pool them as though they were repeated trials of the same algorithm.**
- The current archive does not include a full dependency lockfile from every recorded training environment.

See [`results/archived/DATA_PROVENANCE.md`](results/archived/DATA_PROVENANCE.md), [`paper/REFERENCE_AUDIT.md`](paper/REFERENCE_AUDIT.md), and [`paper/FIGURE_REGENERATION.md`](paper/FIGURE_REGENERATION.md) for additional provenance details.

---

## Troubleshooting

<details>
<summary><strong>PowerShell says that running scripts is disabled.</strong></summary>

Use the explicit command `powershell -ExecutionPolicy Bypass -File .\scripts\RUN_TESTS_WINDOWS.ps1`. This bypasses the restriction **for that PowerShell process only**. Alternatively, run the Python commands directly using `.\.venv\Scripts\python.exe`. You do not need to change your machine-wide policy or activate the virtual environment.

</details>

<details>
<summary><strong>No module named <code>neuro_weave</code>.</strong></summary>

Run the `python -m neuro_weave...` command from either `experiments/protected_core` or `experiments/readout_audit`, as appropriate. Both have a `neuro_weave` package. Running them from the repository root without changing directory causes imports to fail or resolve incorrectly.

</details>

<details>
<summary><strong>CUDA is unavailable, or PyTorch reports a driver mismatch.</strong></summary>

Run `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. If CUDA is false, check the NVIDIA driver and install a PyTorch build selected for your driver and platform from [the official installation page](https://pytorch.org/get-started/locally/). The synthetic command supports `--cpu`; large experiments will be slower on CPU.

</details>

<details>
<summary><strong>COIL-100 download fails, times out, or gives an incomplete archive.</strong></summary>

Try downloading from the official Columbia University host on a network that allows access, then place the extracted `objN__angle.png` images in `experiments/protected_core/data/coil100/images/` (and the audit folder if running that experiment). Check that approximately 7,200 image files are present. Do not change the split code to accommodate missing files without recording the change.

</details>

<details>
<summary><strong>My results do not match the paper exactly.</strong></summary>

First run `scripts/verify_recorded_results.py` to verify the archived CSVs independently of the training process. Next compare the `configuration.json` in the new output folder with the archived study, especially the dataset, seed, class order, shots, input size, train/validation/query split and model settings. CUDA kernels, PyTorch versions and timing can vary. Do not overwrite the original archived results with new runs.

</details>

<details>
<summary><strong>Import or checkpoint loading errors occur.</strong></summary>

Keep each checkpoint with the experimental codebase that created it; the two projects are not interchangeable. Only load `.pt` checkpoints obtained from a trusted source. Inspect the saved `configuration.json` for model shape and class order rather than assuming a checkpoint belongs to another training configuration.

</details>

---

## Publishing or contributing to the research repository

The ZIP is a **repository source snapshot**; it does not itself create a public GitHub repository. Once the authors have approved the code, dataset-use obligations, institutional permissions and licence, you can initialise and push it:

```powershell
git init
git add .
git commit -m "Release NeuroWeave research source and archived results"
git branch -M main
# Create an empty repository on github.com, then replace this example URL.
git remote add origin https://github.com/YOUR-ORGANISATION/YOUR-REPOSITORY.git
git push -u origin main
```

Before making the repository public:

- Confirm whether an open-source `LICENSE` can be granted. **No licence has been assigned automatically.**
- Add a `CITATION.cff` only after author names, a valid paper title, year and identifier have been agreed. [`CITATION_TEMPLATE.md`](CITATION_TEMPLATE.md) is a placeholder, not a confirmed citation.
- Remove any private paths, credentials or unpublished data that were added locally after download.
- Add the actual repository link and, if applicable, a versioned archive DOI to the article's Data and Code Availability section.
- Use Issues or Pull Requests for reproductions, bug reports and code improvements. Include the Python version, operating system, PyTorch/CUDA version, command and the relevant error trace.

### Acknowledging and citing the work

When citing the manuscript, use the final published bibliographic details once available. Until then, refer to the repository and manuscript by their provisional title without inventing a DOI, publication year or venue acceptance. Do not redistribute COIL-100, Oxford-IIIT Pet or CIFAR-10 archives as part of the repository.

---

## Research integrity note

This repository deliberately includes results where simpler point neurons performed as well as or better than their dendritic counterparts. That is a meaningful part of the scientific record. It also preserves the distinction between **representation persistence**, **recognition-access interference**, **class-prototype memory**, and **beneficial circuit adaptation**, which cannot be inferred from a single overall accuracy figure.

**Questions and contributions:** open a GitHub Issue once the public repository is created. Please provide sufficient information to reproduce the observation, including the exact script, configuration and saved results.
