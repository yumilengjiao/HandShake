"""Forward-only checks for explicit experimental variants; no neural training."""

import json

import pytest
import torch

from algorithm.bundle import architecture_for, deployment_architecture, runtime_contract
from algorithm.model_config import MODEL_VARIANTS, build_model
from algorithm.training.config import TrainConfig
from shared.classes import vocabulary_for


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_motion_initialization_preserves_original_baseline_logits():
    torch.manual_seed(42)
    baseline = build_model("tcn").eval()
    torch.manual_seed(42)
    motion = build_model("tcn", variant="motion").eval()
    x = torch.randn(2, 32, 94)
    with torch.inference_mode():
        torch.testing.assert_close(motion(x), baseline(x), rtol=0, atol=0)
    assert motion.receptive_field == 31
    assert sum(p.numel() for p in motion.parameters()) == 70090


@pytest.mark.parametrize("variant", ["motion", "motion_dropout35"])
def test_motion_branch_is_causal_and_matches_recomputed_windows(variant):
    model = build_model("tcn", variant=variant).eval()
    # Exercise the branch, not just its zero-initialized baseline output.
    torch.nn.init.normal_(model.motion_classifier.weight, std=0.1)
    x = torch.randn(2, 49, 94)
    x[..., 73:94] = 1
    changed = x.clone()
    changed[:, 21:] = 100 * torch.randn_like(changed[:, 21:])
    with torch.inference_mode():
        history = model.forward_all(x)
        perturbed = model.forward_all(changed)
        torch.testing.assert_close(
            history[:, :21], perturbed[:, :21], rtol=0, atol=1e-6
        )
        assert not torch.allclose(history[:, 21:], perturbed[:, 21:])
        for endpoint in (31, 39, 48):
            torch.testing.assert_close(
                model(x[:, endpoint - 31 : endpoint + 1]),
                history[:, endpoint],
                rtol=1e-5,
                atol=1e-6,
            )


def test_motion_means_do_not_dilute_constant_hand_geometry_with_padding():
    model = build_model("tcn", variant="motion").eval()
    x = torch.zeros(1, 32, 94)
    x[:, 25:, :69] = 3
    x[:, 25:, 73:94] = 1
    contrasts, present = model.motion_contrasts(x)
    torch.testing.assert_close(
        contrasts, torch.zeros_like(contrasts), rtol=0, atol=1e-6
    )
    assert present.sum() == 7
    with torch.inference_mode():
        model.motion_classifier.bias.fill_(9)
        blank = torch.zeros_like(x)
        expected = super(type(model), model).forward_all(blank)
        torch.testing.assert_close(model.forward_all(blank), expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", MODEL_VARIANTS)
def test_variant_metadata_and_runtime_modules_are_explicit(variant):
    vocabulary = vocabulary_for()
    config = TrainConfig(model_variant=variant)
    assert config.model_variant == variant
    architecture = architecture_for("tcn", vocabulary, variant)
    assert deployment_architecture("tcn", vocabulary, variant)["receptive_field"] == 31
    model = build_model("tcn", variant=variant)
    expected_dropout = 0.35 if "dropout35" in variant else 0.2
    assert architecture["dropout"] == model.blocks[0].dropout.p == expected_dropout
    assert ("algorithm.models.motion_tcn" in runtime_contract(variant)) == (
        "motion" in variant
    )
    if variant == "baseline":
        assert "model_variant" not in architecture
        assert "motion_branch" not in architecture
    else:
        assert architecture["model_variant"] == variant
    json.dumps(architecture, allow_nan=False)


@pytest.mark.parametrize(
    "variant", ["motion", "dropout35", "motion_dropout35", "unknown"]
)
def test_nonbaseline_mlp_and_unknown_variants_are_rejected(variant):
    with pytest.raises(ValueError):
        build_model("mlp", variant=variant)
    with pytest.raises(ValueError):
        TrainConfig(model_type="mlp", model_variant=variant)
