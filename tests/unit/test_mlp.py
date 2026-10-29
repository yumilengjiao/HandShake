import pytest

torch = pytest.importorskip("torch")
from torch import nn

from algorithm.models import GestureMLP, masked_cross_entropy
from shared.classes import load_classes


@pytest.mark.parametrize("batch_size", [1, 4])
def test_mlp_shapes_finite_cpu_loss_and_gradients(batch_size):
    torch.manual_seed(7)
    model = GestureMLP()
    x = torch.randn(batch_size, 94)
    logits = model(x)
    assert logits.shape == (batch_size, load_classes().class_count)
    loss = masked_cross_entropy(
        logits, torch.arange(batch_size) % model.num_classes, torch.ones(batch_size, dtype=torch.bool)
    )
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mlp_architecture_count_and_raw_logits():
    model = GestureMLP()
    assert sum(p.numel() for p in model.parameters()) == 20_741
    assert [(layer.in_features, layer.out_features) for layer in model.network if isinstance(layer, nn.Linear)] == [
        (94, 128), (128, 64), (64, 5)
    ]
    assert [layer.p for layer in model.network if isinstance(layer, nn.Dropout)] == [0.2, 0.2]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[-1].bias.copy_(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert torch.equal(model(torch.zeros(1, 94)), torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]))


def test_mlp_eval_is_reproducible_and_zero_mask_features_are_finite():
    model = GestureMLP().eval()
    x = torch.zeros(3, 94)
    assert torch.equal(model(x), model(x))
    assert torch.isfinite(model(x)).all()
    assert model.class_tokens == load_classes().tokens


@pytest.mark.parametrize("x", [torch.zeros(94), torch.zeros(2, 32, 94), torch.zeros(1, 93), torch.zeros(0, 94), torch.ones(1, 94, dtype=torch.long)])
def test_mlp_rejects_bad_dimensions_or_dtype(x):
    with pytest.raises(ValueError):
        GestureMLP()(x)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_mlp_rejects_nonfinite_features(bad):
    x = torch.zeros(1, 94)
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        GestureMLP()(x)


@pytest.mark.parametrize("kwargs", [{"num_classes": 4}, {"num_classes": 6}, {"input_dim": 0}, {"dropout": 1.0}, {"dropout": float("nan")}, {"hidden_dims": (64,)}, {"vocabulary": object()}])
def test_mlp_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GestureMLP(**kwargs)


def test_mlp_configurable_input_width_and_hidden_widths():
    vocab = load_classes()
    model = GestureMLP(input_dim=12, hidden_dims=(8, 4), num_classes=vocab.class_count, vocabulary=vocab)
    assert model(torch.zeros(2, 12)).shape == (2, vocab.class_count)
