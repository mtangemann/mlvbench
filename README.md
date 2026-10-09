# MLV-Bench: Human vs. Machine Mid-Level Vision

This repository contains the code for the paper "[Vision Transformers Learn Gestalt-Like Figure-Ground Cues from Natural Images](https://arxiv.org/abs/2607.08932)" and provides reusable components for testing mid-level representations in Vision Transformers.

**You are looking at the latest development version of mlvbench. Switch to [v1.0.0](https://github.com/mtangemann/mlvbench/releases/tag/v1.0.0) for the latest stable release.**


## Prerequisites
Clone this repository and install the Python dependencies:

```bash
git clone https://github.com/mtangemann/mlvbench
cd mlvbench
uv sync
```


## Probing figure-ground organization in Vision Transformers
All experiment code is provided in [`experiments/figure_ground_shape_cues`](experiments/figure_ground_shape_cues).

```bash
cd experiments/figure_ground_shape_cues

# Train probes for an individual model and condition. Recommend for debugging and
# testing the setup.
python experiment.py --condition natural --model timm/vit_small_patch16_dinov3.lvd1689m

# Use the `launch.py` script to train probes for all networks and conditions. This
# supports submitting parallel jobs on a Slurm cluster, but you might have to adapt the
# Slurm settings for your cluster.
python launch.py path/to/output
```

## MLV-Bench
Several of the components provided by MLV-Bench are are generic and can be reused for
different experiments, including the *feature extraction* and *probe fitting*. This
components are contained in the core [`mlvbench`](mlvbench) library.

```python
from mlvbench.models import build_model
from mlvbench.trainer import ProbeTrainer

model = build_model("timm/vit_base_patch16_224.orig_in21k")


# --- Feature Extraction ---

images = ...  # torch uint8 Tensor with shape (B, 3, H, W)
features = model.forward_features(images)

features.patch("block.11")                  # Patch features with shape (B, N, C)
features.patch("block.11", format="BHWC")   # or reshaped to a spatial feature map
features.cls("block.11")                    # CLS token


# --- Probe Fitting ---

data_module = MyDataModule(...)

trainer = ProbeTrainer(
    task: "binary_segmentation",
    probe: "linear",
    data_module: data_module,
    batch_size: 256,
    max_steps: 1000,
)
trainer.fit(model)

# Access the fitted probes via `model.probes`
print(f"Fitted {len(model.probes)} probes to the following layers:")
for probe in model.probes:
    print(probe.layer)

# Evaluate probes
test_loader = data_module.test_loader(batch_size=256)
trainer.evaluate(model, test_loader, "output/example/evaluation")
```

Have a look at the [API documentation](https://mtangemann.github.io/mlvbench) for more
information.


## Citation
Please cite our paper if you use MLV-Bench for your experiments.

```bibtex
@misc{tangemann2026figureground,
      title={Vision Transformers Learn Gestalt-Like Figure-Ground Cues from Natural Images}, 
      author={Matthias Tangemann and Benjamin Lo and Zygmunt Pizlo and Kaleem Siddiqi and Dirk B. Walther and Sven Dickinson},
      year={2026},
      eprint={2607.08932},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.08932}, 
}
```

Please also make sure to cite the original authors for all models that you use. Each
model provides a link to more information as `model.url`.
