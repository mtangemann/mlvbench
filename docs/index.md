# MLV-Bench

MLV-Bench provides common building blocks for evaluating mid-level representations in
Vision Transformers (ViTs).

**Installation.** Clone the repository and install it's dependencies to get started.

```bash
git clone https://github.com/mtangemann/mlvbench
cd mlvbench
uv sync
```

**Feature Extraction.** The foundation of MLV-Bench is a unified API for feature extraction from various ViT models. This allows implementing experiments once, and then running them across a wide range of models:

```python
from mlvbench.models import build_model

model = build_model("timm/vit_base_patch16_224.orig_in21k")

images = ...  # torch uint8 Tensor with shape (B, 3, H, W)
features = model.forward_features(images)

features.patch("block.11")                  # Patch features with shape (B, N, C)
features.patch("block.11", format="BHWC")   # or reshaped to a spatial feature map
features.cls("block.11")                    # CLS token
```

Have a look at the [`mlvbench.models`](API/mlvbench.models/) module and the [list of supported models](models) for further information.


**Probing.** Wo provide a high-level API for fitting probes to readout information from intermediate model layers:

```python
from mlvbench.trainer import ProbeTrainer

data_module = MyDataModule(...)

trainer = ProbeTrainer(
    task: "binary_segmentation",
    probe: "linear",
    data_module: data_module,
    batch_size: 256,
    max_steps: 1000,
)

model = build_model("timm/vit_base_patch16_siglip_224.v2_webli")
trainer.fit(model)

# Access the fitted probes via `model.probes`
print(f"Fitted {len(model.probes)} probes to the following layers:")
for probe in model.probes:
    print(probe.layer)

# Evaluate probes
test_loader = data_module.test_loader(batch_size=256)
trainer.evaluate(model, test_loader, "output/example/evaluation")
```

The documentation for [`mlvbench.trainer`](API/mlvbench.trainer) provides more information for how to customize the probe fitting.


**And more.** We're actively working on expanding MLV-Bench. Have a look at the API docs to see what is already there, and stay tuned for further improvements in the future.
