import math

import pytest

torch = pytest.importorskip("torch")

from motion_capture import confidence_map, mixture_of_laplace_nll


def test_nll_matches_equation_for_zero_error():
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros_like(pred)
    alpha = torch.tensor([[[[0.25]]]])
    beta1 = 0.0
    beta2 = math.log(2.0)

    actual = mixture_of_laplace_nll(
        pred, target, alpha, beta1, beta2, reduction="none"
    )
    # 0.25/(2*1) + 0.75/(2*2) = 0.3125。
    expected = torch.full_like(actual, -math.log(0.3125))
    torch.testing.assert_close(actual, expected)


def test_default_reduction_averages_two_flow_coordinates():
    pred = torch.tensor([[[[1.0]], [[2.0]]]])
    target = torch.zeros_like(pred)
    # alpha=1 且 beta1=0 时退化为 scale=1 的单 Laplace。
    actual = mixture_of_laplace_nll(pred, target, 1.0, 0.0, 2.0)
    expected = torch.tensor(math.log(2.0) + 1.5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)


def test_valid_mask_excludes_invalid_pixels():
    pred = torch.tensor([[[[0.0, 100.0]], [[0.0, 100.0]]]])
    target = torch.zeros_like(pred)
    valid = torch.tensor([[[[True, False]]]])
    actual = mixture_of_laplace_nll(
        pred, target, 1.0, 0.0, 2.0, valid_mask=valid
    )
    torch.testing.assert_close(actual, torch.tensor(math.log(2.0)), atol=2e-6, rtol=0)


def test_extreme_error_and_logits_have_finite_gradients():
    pred = torch.full((1, 2, 2, 2), 1e4, requires_grad=True)
    target = torch.zeros_like(pred)
    alpha_logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    beta2 = torch.full((1, 1, 2, 2), 5.0, requires_grad=True)
    loss = mixture_of_laplace_nll(
        pred,
        target,
        alpha_logits,
        0.0,
        beta2,
        alpha_is_logits=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(alpha_logits.grad).all()
    assert torch.isfinite(beta2.grad).all()


def test_confidence_map_raw_value_and_minmax_order():
    alpha = torch.tensor([[[[0.25]]]])
    raw = confidence_map(alpha, 0.0, math.log(2.0), normalization="none")
    torch.testing.assert_close(raw, torch.tensor([[[[0.3125]]]]))

    alpha_varying = torch.tensor([[[[0.1, 0.9]]]])
    normalized = confidence_map(alpha_varying, 0.0, math.log(4.0))
    assert normalized.shape == (1, 1, 1, 2)
    torch.testing.assert_close(normalized[0, 0, 0], torch.tensor([0.0, 1.0]))

