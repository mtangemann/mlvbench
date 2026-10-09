# Figure Ground Shape Cues

**You are looking at the latest development version of mlvbench. Switch to [v1.0.0](https://github.com/mtangemann/mlvbench/releases/tag/v1.0.0) for the code used in the arXiv preprint.**

## Usage
```bash
# Local debug run
python experiment.py --condition ... --model ... [--output-path output/debug]

# Launch all jobs on SLURM
# Re-running this script will launch all jobs that are not done yet or not yet submitted
python launch.py output/<workspace-name> [--config configs/default.yaml]

# List status for each job
python launch.py output/<workspace-name> --status

# Cancel all running and queued jobs
python launch.py output/<workspace-name> --cancel
```


## Output structure
Each experiment run creates a *living workspace* under `output/`: models, evaluations, and reports can be added incrementally during development. Once an experiment is finalized, a clean run can be generated from scratch if desired.

```
output/
└── 2026-08-04_first_run/
    ├── README.md                       # experiment description and logbook
    ├── results/
    │   ├── timm_deit3_base_patch16_224.fb_in22k_ft_in1k/
    │   │   ├── pretrained/
    │   │   │   ├── convexity/
    │   │   │   │   ├── probe/                      # probe training job, incl. val results
    │   │   │   │   └── evaluations/                # outputs of evaluation jobs
    │   │   │   │       ├── convexity_test/         # one directory per condition and test set
    │   │   │   │       ├── convexity_test_reversed/
    │   │   │   │       └── convexity_test_split/
    │   │   │   ├── natural/
    │   │   │   └── ...
    │   │   ├── random_0/               # randomly initialized model with seed 0
    │   │   ├── random_1/
    │   │   └── ...
    │   ├── timm_vit_base_patch16_dinov3.lvd1689m/
    │   └── ...
    └── reports/                        # reports that aggregate and visualize results
```
