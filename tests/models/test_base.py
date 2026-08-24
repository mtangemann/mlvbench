"""Tests for the model base class."""

import pytest
import torch

from mlvbench.models import Features, Model, TokenLayout, load_model
from mlvbench.probes import LinearProbe, build_probe


class _StubModel(Model):
    """Minimal model with deterministic features for testing attached probes."""

    _NUM_TOKENS = 4

    def __init__(self, name: str = "stub/model", embed_dim: int = 8):
        super().__init__()
        self._name = name
        self._embed_dim = embed_dim
        # A parameter so that `Model.device` works and `.to()` has an effect.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_layers(self) -> int:
        return 2

    @property
    def layer_names(self) -> list[str]:
        return ["block.0", "block.1"]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def input_size(self) -> int:
        return 8

    @property
    def url(self) -> str:
        return "https://example.com/stub"

    @property
    def token_layout(self) -> TokenLayout:
        # Four patch tokens on a 2x2 grid, no prefix tokens.
        return TokenLayout(prefix=(), grid_size=None)

    def forward_features(self, images, layers=None):
        """Return features deterministically derived from the input images."""
        layers = layers or self.layer_names
        batch_size = images.shape[0]
        size = self._NUM_TOKENS * self._embed_dim
        base = images.reshape(batch_size, -1)[:, :size]
        base = base.reshape(batch_size, self._NUM_TOKENS, self._embed_dim)
        tokens = {layer: base + index for index, layer in enumerate(layers)}
        # Four patch tokens on a 2x2 grid, no prefix tokens.
        return Features(tokens, TokenLayout(grid_size=(2, 2)))


def _make_probe(model, layer="block.0", lr=1e-3, weight_decay=0.0, step=10):
    return build_probe(
        "linear",
        layer,
        model.embed_dim,
        metadata={"lr": lr, "weight_decay": weight_decay, "step": step},
    )


def _make_probed_model() -> Model:
    model = _StubModel(embed_dim=8)
    model.probes = [_make_probe(model)]
    return model


def test_forward_probes_combines_features_and_probes():
    """forward_probes applies one probe per attached entry and aligns predictions."""
    model = _make_probed_model()
    images = torch.randn(2, 3, 8, 8)

    predictions = model.forward_probes(images)

    assert len(predictions) == len(model.probes)
    # LinearProbe maps (B, N, C) -> (B, N, 1).
    assert predictions[0].shape == (2, _StubModel._NUM_TOKENS, 1)


def test_from_checkpoint_roundtrip(tmp_path, monkeypatch):
    """A saved model reloads with identical probe weights and metadata."""
    model = _make_probed_model()
    path = tmp_path / "model.pt"
    model.save(path)

    monkeypatch.setattr(
        "mlvbench.models.build_model",
        lambda name,
        precision="auto",
        device="auto",
        compile=False,
        pretrained=True,
        seed=None: _StubModel(
            name=name, embed_dim=8
        ),
    )
    loaded = load_model(path, device="cpu")

    images = torch.randn(3, 3, 8, 8)
    expected = model.forward_probes(images)
    actual = loaded.forward_probes(images)

    assert len(actual) == len(expected)
    assert torch.allclose(actual[0], expected[0])

    assert loaded.probes[0].layer == "block.0"
    assert loaded.probes[0].metadata == {"lr": 1e-3, "weight_decay": 0.0, "step": 10}
    assert loaded.probes[0].name == "linear"


def test_checkpoint_records_version_metadata(tmp_path):
    """save() stamps the format version and the creating mlvbench version."""
    from mlvbench.models._base import CHECKPOINT_FORMAT_VERSION

    model = _make_probed_model()
    path = tmp_path / "model.pt"
    model.save(path)

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["mlvbench_checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert isinstance(checkpoint["mlvbench_version"], str)


def test_load_rejects_incompatible_format_version(tmp_path):
    """load_model() fails with a clear error on an unsupported format version."""
    model = _make_probed_model()
    path = tmp_path / "model.pt"
    model.save(path)

    # Simulate a checkpoint written by an incompatible (e.g. older) mlvbench.
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["mlvbench_checkpoint_format_version"] = 0
    checkpoint["mlvbench_version"] = "0.0.0-old"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="format version"):
        load_model(path, device="cpu")


@pytest.mark.parametrize("include_weights", [False, True])
def test_from_checkpoint_restores_weights(tmp_path, monkeypatch, include_weights):
    """`include_weights` controls whether backbone weights are restored on load."""
    model = _StubModel(embed_dim=8)
    with torch.no_grad():
        model.dummy.fill_(5.0)

    path = tmp_path / "model.pt"
    model.save(path, include_weights=include_weights)

    # build_model rebuilds the backbone with fresh (zero) weights, as it would
    # re-download a pretrained checkpoint.
    monkeypatch.setattr(
        "mlvbench.models.build_model",
        lambda name,
        precision="auto",
        device="auto",
        compile=False,
        pretrained=True,
        seed=None: _StubModel(
            name=name, embed_dim=8
        ),
    )
    loaded = load_model(path, device="cpu")

    if include_weights:
        assert loaded.dummy.item() == 5.0  # restored from the saved state_dict
    else:
        assert loaded.dummy.item() == 0.0  # left at the freshly built value


class _CustomProbe(LinearProbe):
    """A custom probe type identified by its own name."""

    name = "custom_linear"


def test_custom_probe_roundtrip(tmp_path, monkeypatch):
    """A custom probe type is restored by supplying `probe_types` at load time."""
    model = _StubModel(embed_dim=8)
    probe = _CustomProbe(
        "block.0",
        model.embed_dim,
        metadata={"lr": 1e-3, "weight_decay": 0.0, "step": 10},
    )
    model.probes = [probe]

    path = tmp_path / "model.pt"
    model.save(path)

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["probes"][0]["name"] == "custom_linear"

    monkeypatch.setattr(
        "mlvbench.models.build_model",
        lambda name,
        precision="auto",
        device="auto",
        compile=False,
        pretrained=True,
        seed=None: _StubModel(
            name=name, embed_dim=8
        ),
    )

    with pytest.raises(ValueError, match="Unknown probe type"):
        load_model(path, device="cpu")

    loaded = load_model(
        path, device="cpu", probe_types={"custom_linear": _CustomProbe}
    )
    assert len(loaded.probes) == 1
    assert isinstance(loaded.probes[0], _CustomProbe)
    assert loaded.probes[0].layer == "block.0"
    assert loaded.probes[0].metadata["step"] == 10
