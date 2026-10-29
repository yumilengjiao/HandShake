import pytest

torch = pytest.importorskip("torch")
from torch import nn

from algorithm.models import CausalConv1d, GestureTCN, masked_cross_entropy
from shared.classes import load_classes


@pytest.fixture(autouse=True, scope="module")
def limit_cpu_threads():
    # Tiny diagnostic tensors are slower with large OpenMP thread pools.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_default_tcn_structure_capacity_and_receptive_field():
    model = GestureTCN()
    assert model.receptive_field == 31
    assert model.window_size == 32
    assert sum(p.numel() for p in model.parameters()) == 56_325
    assert sum(p.numel() for p in model.input_projection.parameters()) == 6_080
    assert sum(p.numel() for p in model.blocks.parameters()) == 49_920
    assert sum(p.numel() for p in model.classifier.parameters()) == 325
    assert len(model.blocks) == 4
    assert model.class_tokens == load_classes().tokens
    for block, dilation in zip(model.blocks, (1, 2, 4, 8)):
        assert sum(isinstance(module, nn.Conv1d) for module in block.modules()) == 1
        assert block.conv.padding == (0,)
        assert block.conv.stride == (1,)
        assert block.conv.dilation == (dilation,)
        assert block.conv.kernel_size == (3,)
        assert block.conv.left_padding == 2 * dilation
        assert block.norm.normalized_shape == (64,)
        assert block.dropout.p == 0.2


def test_causal_conv_uses_only_current_and_past_with_left_padding():
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=2, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0)
    x = torch.arange(1.0, 8.0).view(1, 1, -1)
    torch.testing.assert_close(conv(x), torch.tensor([[[1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 15.0]]]))


@pytest.mark.parametrize("batch_size", [1, 3])
def test_tcn_cpu_forward_backward_and_batch_shape(batch_size):
    torch.manual_seed(71)
    model = GestureTCN()
    x = torch.randn(batch_size, 32, 94)
    logits = model(x)
    assert logits.shape == (batch_size, 5)
    assert model.forward_all(x).shape == (batch_size, 32, 5)
    loss = masked_cross_entropy(
        logits, torch.arange(batch_size) % model.num_classes, torch.ones(batch_size, dtype=torch.bool)
    )
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("cut", [0, 9, 30])
def test_future_perturbations_cannot_change_present_or_past(cut):
    torch.manual_seed(73)
    model = GestureTCN().eval()
    x = torch.randn(2, 32, 94)
    changed = x.clone()
    changed[:, cut + 1 :, :] = 100 * torch.randn_like(changed[:, cut + 1 :, :])
    with torch.no_grad():
        original = model.forward_all(x)
        perturbed = model.forward_all(changed)
    torch.testing.assert_close(original[:, : cut + 1], perturbed[:, : cut + 1], rtol=0, atol=1e-6)
    assert not torch.allclose(original[:, cut + 1 :], perturbed[:, cut + 1 :])


def test_receptive_field_includes_earliest_allowed_frame_only():
    torch.manual_seed(81)
    model = GestureTCN().eval()
    x = torch.randn(1, 32, 94, requires_grad=True)
    model(x).square().sum().backward()
    assert x.grad is not None
    # At t=31 the 31-frame receptive field is exactly [1,31].
    assert torch.count_nonzero(x.grad[:, 0]) == 0
    assert x.grad[:, 1].abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_full_history_and_recomputed_window_endpoints_agree():
    torch.manual_seed(83)
    model = GestureTCN().eval()
    x = torch.randn(2, 49, 94)
    with torch.no_grad():
        history = model.forward_all(x)
        for endpoint in (31, 39, 48):
            window = x[:, endpoint - 31 : endpoint + 1]
            torch.testing.assert_close(model(window), history[:, endpoint], rtol=1e-5, atol=1e-6)


def test_batch_samples_do_not_change_other_samples_outputs():
    torch.manual_seed(85)
    model = GestureTCN().eval()
    x = torch.randn(1, 32, 94)
    joined = torch.cat((x, 100 * torch.randn_like(x)), dim=0)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(joined)[:1], rtol=1e-5, atol=1e-6)


def test_eval_reproducible_raw_logits_zero_mask_and_short_prefix():
    model = GestureTCN().eval()
    x = torch.zeros(2, 32, 94)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
        assert torch.isfinite(model(x)).all()
        assert model.forward_all(x[:, :1]).shape == (2, 1, 5)
        torch.testing.assert_close(model(x), model.forward_all(x)[:, -1])
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.arange(-2.0, 3.0))
        assert torch.equal(model(x), torch.arange(-2.0, 3.0).repeat(2, 1))


@pytest.mark.parametrize("x", [torch.zeros(1, 94), torch.zeros(1, 32, 93), torch.zeros(0, 32, 94), torch.zeros(1, 0, 94), torch.zeros(1, 32, 94, dtype=torch.long)])
def test_tcn_rejects_invalid_input_shape_or_dtype(x):
    with pytest.raises(ValueError):
        GestureTCN().forward_all(x)


@pytest.mark.parametrize("length", [1, 30, 31, 33])
def test_regular_forward_requires_full_window(length):
    with pytest.raises(ValueError, match="exactly 32"):
        GestureTCN()(torch.zeros(1, length, 94))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_tcn_rejects_nonfinite_input(value):
    x = torch.zeros(1, 32, 94)
    x[0, 3, 7] = value
    with pytest.raises(ValueError, match="finite"):
        GestureTCN()(x)


@pytest.mark.parametrize("kwargs", [{"num_classes": 4}, {"num_classes": 6}, {"input_dim": 0}, {"channels": 0}, {"dilations": ()}, {"dilations": (0, 2)}, {"dropout": float("inf")}, {"window_size": 30}, {"kernel_size": 0}, {"vocabulary": object()}])
def test_tcn_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GestureTCN(**kwargs)


def test_configurable_tcn_width_dilations_and_window():
    model = GestureTCN(input_dim=12, channels=8, dilations=(1, 2), dropout=0, window_size=8)
    assert model.receptive_field == 7
    assert model(torch.zeros(2, 8, 12)).shape == (2, 5)
