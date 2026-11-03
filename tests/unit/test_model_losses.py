import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from algorithm.models import masked_cross_entropy


def test_masked_ce_matches_only_supervised_rows_and_gradients():
    logits = torch.tensor([[1.0, -2, 3, 1, 0], [5.0, 4, 3, 2, 1], [0.0, 2, 1, -1, -2]], requires_grad=True)
    targets = torch.tensor([2, -1, 1])
    valid = torch.tensor([True, False, True])
    loss = masked_cross_entropy(logits, targets, valid)
    assert loss is not None
    torch.testing.assert_close(loss, F.cross_entropy(logits[[0, 2]], targets[[0, 2]]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[1], torch.zeros(5))
    assert logits.grad[[0, 2]].abs().sum() > 0


def test_all_invalid_batch_returns_skip_signal_without_fake_zero_loss():
    logits = torch.randn(2, 5, requires_grad=True)
    assert masked_cross_entropy(logits, torch.full((2,), -1), torch.zeros(2, dtype=torch.bool)) is None
    assert logits.grad is None


def test_sequence_diagnostic_loss_supports_matching_masks():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[0, 1, -1], [4, -1, 2]])
    valid = targets >= 0
    loss = masked_cross_entropy(logits, targets, valid)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~valid]) == 0


@pytest.mark.parametrize("bad_label", [-1, 5, 99])
def test_valid_label_out_of_vocabulary_is_rejected(bad_label):
    with pytest.raises(ValueError, match="vocabulary"):
        masked_cross_entropy(torch.zeros(1, 5), torch.tensor([bad_label]), torch.tensor([True]))


def test_invalid_label_sentinel_is_not_classified_as_background():
    logits = torch.randn(2, 5)
    loss = masked_cross_entropy(logits, torch.tensor([-999, 0]), torch.tensor([False, True]))
    torch.testing.assert_close(loss, F.cross_entropy(logits[1:], torch.tensor([0])))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_rejected_even_on_invalid_rows(bad):
    logits = torch.zeros(2, 5)
    logits[0, 2] = bad
    with pytest.raises(ValueError, match="finite"):
        masked_cross_entropy(logits, torch.tensor([-1, 0]), torch.tensor([False, True]))


@pytest.mark.parametrize("logits,targets,valid", [
    (torch.zeros(5), torch.tensor(0), torch.tensor(True)),
    (torch.zeros(2, 5), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, dtype=torch.bool)),
    (torch.zeros(2, 5), torch.zeros(2, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool)),
    (torch.zeros(0, 5), torch.zeros(0, dtype=torch.long), torch.ones(0, dtype=torch.bool)),
    (torch.zeros(2, 5), torch.zeros(2), torch.ones(2, dtype=torch.bool)),
    (torch.zeros(2, 5), torch.zeros(2, dtype=torch.long), torch.ones(2)),
    (torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)),
])
def test_loss_rejects_shape_and_dtype_mistakes(logits, targets, valid):
    with pytest.raises(ValueError):
        masked_cross_entropy(logits, targets, valid)


def test_large_half_precision_logits_use_finite_loss_reduction():
    logits = torch.tensor([[60000, -60000, 0, 1, 2]], dtype=torch.float16, requires_grad=True)
    loss = masked_cross_entropy(logits, torch.tensor([1]), torch.tensor([True]))
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_weighted_ce_uses_fixed_weights_not_batch_renormalization():
    logits = torch.tensor([[1., -2, 3], [5., 4, 3], [0., 2, 1]], requires_grad=True)
    targets = torch.tensor([2, -1, 1])
    valid = targets >= 0
    weights = torch.tensor([0., .5, 2.])
    loss = masked_cross_entropy(logits, targets, valid, class_weights=weights)
    expected = (F.cross_entropy(logits[valid], targets[valid], reduction="none") * weights[targets[valid]]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros(3))
    single = masked_cross_entropy(logits[:1], targets[:1], valid[:1], class_weights=weights)
    torch.testing.assert_close(single, 2 * F.cross_entropy(logits[:1], targets[:1]))


def test_weighted_diagnostic_sequence_and_half_reduction():
    logits = torch.tensor([[[60000., -60000.], [0., 1.]]], dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([[1, -1]])
    weights = torch.tensor([.5, 2.])
    loss = masked_cross_entropy(logits, targets, targets >= 0, class_weights=weights)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad[:, 1]) == 0
    assert masked_cross_entropy(logits, targets, torch.zeros_like(targets, dtype=torch.bool), class_weights=weights) is None


@pytest.mark.parametrize("weights", [
    torch.ones(2), torch.ones(3, 1), torch.ones(3, dtype=torch.long),
    torch.tensor([1., -1, 1]), torch.zeros(3), torch.tensor([1., float("nan"), 1]),
    torch.tensor([1., float("inf"), 1]), torch.ones(3, requires_grad=True), [1., 1., 1.],
])
def test_invalid_class_weights_rejected(weights):
    with pytest.raises(ValueError, match="class_weights"):
        masked_cross_entropy(torch.zeros(1, 3), torch.tensor([0]), torch.tensor([True]), class_weights=weights)


def test_supervised_class_missing_from_weight_counts_is_rejected():
    with pytest.raises(ValueError, match="positive class weights"):
        masked_cross_entropy(torch.zeros(1, 3), torch.tensor([0]), torch.tensor([True]), class_weights=torch.tensor([0., 1, 1]))
