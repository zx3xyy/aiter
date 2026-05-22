# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib

import aiter
import pytest
import torch
from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_moe_fake
from aiter.ops.shuffle import shuffle_weight


def _make_bf16_case(token_num=16, hidden_dim=128, inter_dim=128, expert_num=4, topk=2):
    hidden_states = torch.randn(
        (token_num, hidden_dim), dtype=dtypes.bf16, device="cuda"
    )
    w1 = torch.randn(
        (expert_num, inter_dim, hidden_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn(
        (expert_num, hidden_dim, inter_dim), dtype=dtypes.bf16, device="cuda"
    )
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 2], [3, 0]] * (token_num // 4),
        dtype=dtypes.i32,
        device="cuda",
    )
    topk_weight = torch.rand((token_num, topk), dtype=dtypes.fp32, device="cuda")
    return hidden_states, w1, w2, topk_weight, topk_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_returns_unweighted_route_outputs():
    torch.manual_seed(0)
    token_num = 256
    hidden_dim = 128
    topk = 2
    dtype = dtypes.bf16

    hidden_states, w1, w2, topk_weight, topk_ids = _make_bf16_case(
        token_num=token_num, hidden_dim=hidden_dim, topk=topk
    )

    w1_aiter = shuffle_weight(w1, layout=(16, 16))
    w2_aiter = shuffle_weight(w2, layout=(16, 16))
    out = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        do_finalize=False,
    )
    out_with_different_weights = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight.flip(1),
        topk_ids,
        activation=aiter.ActivationType.Silu,
        do_finalize=False,
    )

    assert out.shape == (token_num, topk, hidden_dim)
    assert out.dtype == dtype
    torch.testing.assert_close(out, out_with_different_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_ignores_weights_with_doweight_stage1():
    torch.manual_seed(2)
    hidden_states, w1, w2, topk_weight, topk_ids = _make_bf16_case(token_num=256)
    w1_aiter = shuffle_weight(w1, layout=(16, 16))
    w2_aiter = shuffle_weight(w2, layout=(16, 16))

    out = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=True,
        do_finalize=False,
    )
    out_with_different_weights = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight.flip(1),
        topk_ids,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=True,
        do_finalize=False,
    )

    torch.testing.assert_close(out, out_with_different_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_weighted_sum_matches_finalized_output():
    torch.manual_seed(1)
    token_num = 256
    hidden_dim = 128
    inter_dim = 128
    expert_num = 4
    topk = 2
    dtype = dtypes.bf16

    hidden_states = torch.randn((token_num, hidden_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((expert_num, inter_dim, hidden_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((expert_num, hidden_dim, inter_dim), dtype=dtype, device="cuda")
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 2], [3, 0]] * (token_num // 4),
        dtype=dtypes.i32,
        device="cuda",
    )
    topk_weight = torch.rand((token_num, topk), dtype=dtypes.fp32, device="cuda")

    w1_aiter = shuffle_weight(w1, layout=(16, 16))
    w2_aiter = shuffle_weight(w2, layout=(16, 16))
    route_outputs = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        do_finalize=False,
    )

    combined_output = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
    )
    manual_combined = (route_outputs.float() * topk_weight[..., None]).sum(dim=1)
    diff = manual_combined - combined_output.float()

    assert diff.norm() / combined_output.float().norm() < 0.02
    assert diff.abs().max() < 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_respects_expert_mask():
    torch.manual_seed(3)
    hidden_states, w1, w2, topk_weight, topk_ids = _make_bf16_case()
    expert_mask = torch.tensor([1, 1, 1, 0], dtype=dtypes.i32, device="cuda")
    w1_aiter = shuffle_weight(w1[:3], layout=(16, 16))
    w2_aiter = shuffle_weight(w2[:3], layout=(16, 16))

    out = fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=aiter.ActivationType.Silu,
        do_finalize=False,
    )

    masked_route = topk_ids == 3
    assert torch.count_nonzero(out[masked_route]) == 0
    assert torch.count_nonzero(out[~masked_route]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_does_not_reenter_fused_moe(monkeypatch):
    torch.manual_seed(0)
    token_num = 16
    hidden_dim = 128
    inter_dim = 128
    expert_num = 4
    topk = 2
    dtype = dtypes.bf16

    hidden_states = torch.randn((token_num, hidden_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((expert_num, inter_dim, hidden_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((expert_num, hidden_dim, inter_dim), dtype=dtype, device="cuda")
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 2], [3, 0]] * (token_num // 4),
        dtype=dtypes.i32,
        device="cuda",
    )
    topk_weight = torch.rand((token_num, topk), dtype=dtypes.fp32, device="cuda")

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    original_fused_moe_ = fused_moe_module.fused_moe_
    call_count = 0

    def wrapped_fused_moe_(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_fused_moe_(*args, **kwargs)

    monkeypatch.setattr(fused_moe_module, "fused_moe_", wrapped_fused_moe_)

    w1_aiter = shuffle_weight(w1, layout=(16, 16))
    w2_aiter = shuffle_weight(w2, layout=(16, 16))
    fused_moe_module.fused_moe(
        hidden_states,
        w1_aiter,
        w2_aiter,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        do_finalize=False,
    )

    assert call_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_handles_zero_tokens():
    hidden_states = torch.empty((0, 128), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn((4, 128, 128), dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn((4, 128, 128), dtype=dtypes.bf16, device="cuda")
    topk_ids = torch.empty((0, 2), dtype=dtypes.i32, device="cuda")
    topk_weight = torch.empty((0, 2), dtype=dtypes.fp32, device="cuda")

    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        do_finalize=False,
    )

    assert out.shape == (0, 2, 128)
    assert out.dtype == dtypes.bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_moe_do_finalize_false_rejects_num_local_tokens():
    hidden_states, w1, w2, topk_weight, topk_ids = _make_bf16_case(token_num=4)
    num_local_tokens = torch.tensor([4], dtype=dtypes.i32, device="cuda")

    with pytest.raises(NotImplementedError, match="do_finalize=False"):
        fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            num_local_tokens=num_local_tokens,
            do_finalize=False,
        )


def test_fused_moe_do_finalize_false_rejects_quantized_path():
    hidden_states = torch.empty((4, 128), dtype=dtypes.bf16)
    w1 = torch.empty((4, 128, 128), dtype=dtypes.bf16)
    w2 = torch.empty((4, 128, 128), dtype=dtypes.bf16)
    topk_ids = torch.empty((4, 2), dtype=dtypes.i32)
    topk_weight = torch.empty((4, 2), dtype=dtypes.fp32)

    with pytest.raises(NotImplementedError, match="QuantType.No"):
        fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant_type=aiter.QuantType.per_1x32,
            do_finalize=False,
        )


def test_fused_moe_fake_rejects_do_finalize_false_num_local_tokens():
    hidden_states = torch.empty((4, 128), dtype=dtypes.bf16)
    w1 = torch.empty((4, 128, 128), dtype=dtypes.bf16)
    w2 = torch.empty((4, 128, 128), dtype=dtypes.bf16)
    topk_ids = torch.empty((4, 2), dtype=dtypes.i32)
    topk_weight = torch.empty((4, 2), dtype=dtypes.fp32)
    num_local_tokens = torch.tensor([4], dtype=dtypes.i32)

    with pytest.raises(NotImplementedError, match="do_finalize=False"):
        fused_moe_fake(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            num_local_tokens=num_local_tokens,
            do_finalize=False,
        )


def test_fused_moe_fake_rejects_do_finalize_false_quantized_path():
    hidden_states = torch.empty((4, 128), dtype=dtypes.bf16)
    w1 = torch.empty((4, 128, 128), dtype=dtypes.bf16)
    w2 = torch.empty((4, 128, 128), dtype=dtypes.bf16)
    topk_ids = torch.empty((4, 2), dtype=dtypes.i32)
    topk_weight = torch.empty((4, 2), dtype=dtypes.fp32)

    with pytest.raises(NotImplementedError, match="QuantType.No"):
        fused_moe_fake(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant_type=aiter.QuantType.per_1x32.value,
            do_finalize=False,
        )
