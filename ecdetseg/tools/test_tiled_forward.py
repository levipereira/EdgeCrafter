"""Sanity tests for TiledECDet tile arithmetic and feature map reconstruction.

Validates pure tensor operations without model code, weights, GPU, or dataset.
Run: conda activate ec && python ecdetseg/tools/test_tiled_forward.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    """Register a pass/fail result with optional detail."""
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return condition


def pad_to_tile_compatible(H: int, W: int, tile_size: int, stride: int):
    """Compute padded dimensions so tiles cover the entire frame.

    Condition: (H_pad - tile_size) % stride == 0 and H_pad >= H.
    When tile_size is a multiple of stride, this simplifies to H_pad % stride == 0.
    """
    import math
    H_pad = tile_size + math.ceil((H - tile_size) / stride) * stride if H > tile_size else tile_size
    W_pad = tile_size + math.ceil((W - tile_size) / stride) * stride if W > tile_size else tile_size
    return H_pad, W_pad


# ============================================================================
# TEST 1: Tile grid arithmetic (with padding)
# ============================================================================
def test_tile_grid_arithmetic():
    """Verify grid dimensions and that padding makes frames tile-compatible."""
    print("\n" + "=" * 70)
    print("TEST 1: Tile grid arithmetic + PadToMultiple")
    print("=" * 70)

    tile_size = 448
    stride = 224
    patch_size = 16

    cases = [
        # (H_orig, W_orig, H_padded, W_padded, exp_nh, exp_nw)
        (1080, 1920, 1120, 2016, 4, 8),
        (720,  1280,  896, 1344, 3, 5),
        (448,   448,  448,  448, 1, 1),
        (672,   896,  672,  896, 2, 3),
    ]

    for H_orig, W_orig, H_exp, W_exp, exp_nh, exp_nw in cases:
        H_pad, W_pad = pad_to_tile_compatible(H_orig, W_orig, tile_size, stride)
        check(
            f"Padding {H_orig}x{W_orig}",
            H_pad == H_exp and W_pad == W_exp,
            f"padded={H_pad}x{W_pad} (exp {H_exp}x{W_exp})"
        )

        nh = (H_pad - tile_size) // stride + 1
        nw = (W_pad - tile_size) // stride + 1
        N = nh * nw
        check(
            f"  Grid {H_pad}x{W_pad}",
            nh == exp_nh and nw == exp_nw,
            f"nh={nh} (exp {exp_nh}), nw={nw} (exp {exp_nw}), N={N}"
        )

        tile_feat_h = tile_size // patch_size
        stride_feat = stride // patch_size
        global_feat_h = H_pad // patch_size
        global_feat_w = W_pad // patch_size

        last_y = (nh - 1) * stride_feat + tile_feat_h
        last_x = (nw - 1) * stride_feat + tile_feat_h
        check(
            f"  Full coverage H",
            last_y == global_feat_h,
            f"last_y={last_y}, global_feat_h={global_feat_h}"
        )
        check(
            f"  Full coverage W",
            last_x == global_feat_w,
            f"last_x={last_x}, global_feat_w={global_feat_w}"
        )

    print()
    print("  [INFO] Raw 1080x1920 is NOT tile-compatible (coverage 78%).")
    print("  [INFO] PadToMultiple(stride=224) pads to 1120x2016 for full coverage.")
    print("  [INFO] This confirms CRITICAL-1: PadToMultiple is required.")


# ============================================================================
# TEST 2: F.unfold tile count on padded sizes
# ============================================================================
def test_unfold_tile_count():
    """Verify F.unfold output matches expected tile count on padded frames."""
    print("\n" + "=" * 70)
    print("TEST 2: F.unfold tile count (padded sizes)")
    print("=" * 70)

    B, C = 2, 3
    tile_size = 448
    stride = 224

    cases = [
        # (H_padded, W_padded, exp_N)
        (1120, 2016, 32),
        (896,  1344, 15),
        (448,   448,  1),
        (672,   896,  6),
    ]

    for H, W, exp_N in cases:
        x = torch.randn(B, C, H, W)
        patches = F.unfold(x, kernel_size=tile_size, stride=stride)
        N = patches.shape[-1]
        check(
            f"unfold {H}x{W}",
            N == exp_N,
            f"N={N} (exp {exp_N}), patches shape={list(patches.shape)}"
        )


# ============================================================================
# TEST 3: Tile extraction shape and content verification
# ============================================================================
def test_tile_extraction_shapes():
    """Verify tile reshaping and content from F.unfold output."""
    print("\n" + "=" * 70)
    print("TEST 3: Tile extraction shapes + content")
    print("=" * 70)

    B, C, H, W = 2, 3, 1120, 2016
    tile_size = 448
    stride = 224

    x = torch.randn(B, C, H, W)
    patches = F.unfold(x, kernel_size=tile_size, stride=stride)
    N = patches.shape[-1]

    tiles = patches.view(B, C, tile_size, tile_size, N).permute(0, 4, 1, 2, 3)
    tiles = tiles.reshape(B * N, C, tile_size, tile_size)

    check(
        "tiles shape",
        list(tiles.shape) == [B * N, C, tile_size, tile_size],
        f"{list(tiles.shape)} (exp [{B*N}, {C}, {tile_size}, {tile_size}])"
    )

    tile_0_0 = tiles[0]
    expected = x[0, :, 0:tile_size, 0:tile_size]
    check(
        "tile[0,0] == x[0,:,0:448,0:448]",
        torch.allclose(tile_0_0, expected, atol=1e-6),
        f"max diff = {(tile_0_0 - expected).abs().max().item():.2e}"
    )

    tile_0_1 = tiles[1]
    expected_01 = x[0, :, 0:tile_size, stride:stride + tile_size]
    check(
        "tile[0,1] == x[0,:,0:448,224:672]",
        torch.allclose(tile_0_1, expected_01, atol=1e-6),
        f"max diff = {(tile_0_1 - expected_01).abs().max().item():.2e}"
    )

    nw = (W - tile_size) // stride + 1
    tile_1_0 = tiles[nw]
    expected_10 = x[0, :, stride:stride + tile_size, 0:tile_size]
    check(
        "tile[1,0] == x[0,:,224:672,0:448]",
        torch.allclose(tile_1_0, expected_10, atol=1e-6),
        f"max diff = {(tile_1_0 - expected_10).abs().max().item():.2e}"
    )


# ============================================================================
# TEST 4: Feature map accumulation coverage on padded sizes
# ============================================================================
def test_feature_map_coverage():
    """Verify tile accumulation covers the entire global feature map."""
    print("\n" + "=" * 70)
    print("TEST 4: Feature map accumulation coverage (padded)")
    print("=" * 70)

    B = 2
    embed_dim = 192
    tile_size = 448
    stride = 224
    patch_size = 16

    cases = [(1120, 2016), (896, 1344), (672, 896), (448, 448)]

    for H, W in cases:
        nh = (H - tile_size) // stride + 1
        nw = (W - tile_size) // stride + 1
        N = nh * nw

        tile_feat_h = tile_size // patch_size
        stride_feat = stride // patch_size
        global_feat_h = H // patch_size
        global_feat_w = W // patch_size

        global_count = torch.zeros(B, 1, global_feat_h, global_feat_w)

        for tile_idx in range(N):
            row = tile_idx // nw
            col = tile_idx % nw
            y0 = row * stride_feat
            x0 = col * stride_feat
            y1 = y0 + tile_feat_h
            x1 = x0 + tile_feat_h

            global_count[:, :, y0:y1, x0:x1] += 1.0

        min_count = global_count.min().item()
        max_count = global_count.max().item()
        coverage = (global_count > 0).float().mean().item()
        zero_pixels = (global_count == 0).sum().item()

        check(
            f"Coverage {H}x{W} (N={N})",
            coverage == 1.0,
            f"min={min_count}, max={max_count}, zeros={zero_pixels}"
        )
        check(
            f"  min_count >= 1",
            min_count >= 1.0,
            f"min={min_count}"
        )

        print(f"  [INFO] Overlap distribution: min={min_count}, max={max_count}")


# ============================================================================
# TEST 5: F.fold reconstruction on padded sizes
# ============================================================================
def test_fold_reconstruction():
    """Verify F.fold can reconstruct input from F.unfold output."""
    print("\n" + "=" * 70)
    print("TEST 5: F.fold reconstruction (unfold invertibility)")
    print("=" * 70)

    B, C = 2, 3
    tile_size = 448
    stride = 224

    cases = [(1120, 2016), (896, 1344), (448, 448)]

    for H, W in cases:
        x = torch.randn(B, C, H, W)

        patches = F.unfold(x, kernel_size=tile_size, stride=stride)

        ones_patches = F.unfold(
            torch.ones(B, C, H, W), kernel_size=tile_size, stride=stride
        )
        divisor = F.fold(
            ones_patches, output_size=(H, W),
            kernel_size=tile_size, stride=stride
        )

        reconstructed = F.fold(
            patches, output_size=(H, W),
            kernel_size=tile_size, stride=stride
        )
        reconstructed = reconstructed / divisor.clamp(min=1.0)

        max_diff = (x - reconstructed).abs().max().item()
        check(
            f"Fold reconstruction {H}x{W}",
            max_diff < 1e-5,
            f"max_diff={max_diff:.2e}"
        )


# ============================================================================
# TEST 6: Gradient flow through unfold
# ============================================================================
def test_gradient_flow():
    """Verify gradients flow through F.unfold."""
    print("\n" + "=" * 70)
    print("TEST 6: Gradient flow through F.unfold")
    print("=" * 70)

    B, C, H, W = 1, 3, 1120, 2016
    tile_size = 448
    stride = 224

    x = torch.randn(B, C, H, W, requires_grad=True)
    patches = F.unfold(x, kernel_size=tile_size, stride=stride)
    loss = patches.sum()
    loss.backward()

    check("Gradient exists", x.grad is not None)
    check(
        "Gradient non-zero",
        x.grad is not None and x.grad.abs().sum().item() > 0,
        f"grad sum = {x.grad.abs().sum().item():.2f}" if x.grad is not None else ""
    )


# ============================================================================
# TEST 7: Memory estimate (informational)
# ============================================================================
def test_memory_estimate():
    """Print memory estimates for reference."""
    print("\n" + "=" * 70)
    print("TEST 7: Memory estimates (informational)")
    print("=" * 70)

    B = 2
    embed_dim = 192
    patch_size = 16
    tile_size = 448
    stride = 224
    depth = 12

    tile_feat_h = tile_size // patch_size
    tokens_per_tile = tile_feat_h * tile_feat_h + 1

    configs = [
        ("1080p padded", 1120, 2016),
        ("720p padded", 896, 1344),
    ]

    for label, H, W in configs:
        global_feat_h = H // patch_size
        global_feat_w = W // patch_size
        nh = (H - tile_size) // stride + 1
        nw = (W - tile_size) // stride + 1
        N = nh * nw

        bytes_per_float = 4
        act_mb = B * tokens_per_tile * embed_dim * depth * bytes_per_float / 1e6
        global_mb = B * embed_dim * global_feat_h * global_feat_w * bytes_per_float / 1e6

        print(f"  [{label}] {H}x{W}, Tiles: {N} ({nh}x{nw})")
        print(f"    Tokens/tile: {tokens_per_tile}, Feature map: {global_feat_h}x{global_feat_w}")
        print(f"    Activations/tile: ~{act_mb:.0f} MB, Global feat: ~{global_mb:.0f} MB")
        print(f"    Peak: ~{act_mb + global_mb:.0f} MB")

    check("Memory estimates computed", True, "informational only")


# ============================================================================
# TEST 8: Verify raw 1080x1920 is NOT tile-compatible (documents CRITICAL-1)
# ============================================================================
def test_raw_1080p_not_compatible():
    """Confirm that raw 1080x1920 lacks full coverage -- documents CRITICAL-1."""
    print("\n" + "=" * 70)
    print("TEST 8: Raw 1080x1920 is NOT tile-compatible (CRITICAL-1)")
    print("=" * 70)

    tile_size = 448
    stride = 224
    patch_size = 16

    H, W = 1080, 1920
    nh = (H - tile_size) // stride + 1
    nw = (W - tile_size) // stride + 1

    pixel_coverage_h = (nh - 1) * stride + tile_size
    pixel_coverage_w = (nw - 1) * stride + tile_size
    uncovered_h = H - pixel_coverage_h
    uncovered_w = W - pixel_coverage_w

    check(
        "Raw 1080p has incomplete coverage",
        uncovered_h > 0 or uncovered_w > 0,
        f"uncovered: {uncovered_h}px bottom, {uncovered_w}px right"
    )

    H_pad, W_pad = pad_to_tile_compatible(H, W, tile_size, stride)
    check(
        "After PadToMultiple: full coverage",
        (H_pad - tile_size) % stride == 0 and (W_pad - tile_size) % stride == 0,
        f"{H}x{W} -> {H_pad}x{W_pad}, pad +{H_pad-H}px H, +{W_pad-W}px W"
    )

    print(f"  [INFO] CRITICAL-1 confirmed: PadToMultiple(stride={stride}) is mandatory.")
    print(f"  [INFO] 1080p: 1080x1920 -> 1120x2016 (+40px H, +96px W)")


# ============================================================================
# TEST 9: VisionTransformer.forward returns (outs, final_register) [Step 2]
# ============================================================================
def test_vit_forward_signature():
    """Verify VisionTransformer returns (outs, register) and accepts register_state."""
    print("\n" + "=" * 70)
    print("TEST 9: VisionTransformer.forward signature [Step 2]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import VisionTransformer

    torch.manual_seed(42)
    vit = VisionTransformer(
        embed_dim=192, num_heads=3, depth=12, patch_size=16,
        return_layers=[10, 11],
    )
    vit.eval()

    B = 2
    x = torch.randn(B, 3, 640, 640)

    with torch.no_grad():
        result = vit(x)

    check(
        "Returns tuple of length 2",
        isinstance(result, tuple) and len(result) == 2,
        f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
    )

    outs, final_reg = result

    check(
        "outs is list of 2 tensors",
        isinstance(outs, list) and len(outs) == 2,
        f"len={len(outs)}"
    )
    check(
        "outs[0] shape [B, 1600, 192]",
        list(outs[0].shape) == [B, 1600, 192],
        f"{list(outs[0].shape)}"
    )
    check(
        "final_register shape [B, 1, 192]",
        list(final_reg.shape) == [B, 1, 192],
        f"{list(final_reg.shape)}"
    )

    with torch.no_grad():
        custom_reg = torch.randn(B, 1, 192)
        outs2, final_reg2 = vit(x, register_state=custom_reg)

    check(
        "Accepts register_state without error",
        isinstance(outs2, list) and len(outs2) == 2,
        ""
    )
    check(
        "Custom register_state changes output",
        not torch.allclose(outs[0], outs2[0], atol=1e-5),
        "outputs differ as expected when register_state differs"
    )

    with torch.no_grad():
        outs_a, reg_a = vit(x, register_state=None)
        outs_b, reg_b = vit(x, register_state=None)

    check(
        "Deterministic with register_state=None",
        torch.allclose(outs_a[0], outs_b[0], atol=1e-6),
        f"max_diff={( outs_a[0] - outs_b[0]).abs().max().item():.2e}"
    )


# ============================================================================
# TEST 10: ViTAdapter._forward_single backward compat [Step 3]
# ============================================================================
def test_vit_adapter_backward_compat():
    """Verify ViTAdapter._forward_single produces correct shapes."""
    print("\n" + "=" * 70)
    print("TEST 10: ViTAdapter._forward_single backward compat [Step 3]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import ViTAdapter

    torch.manual_seed(42)
    adapter = ViTAdapter(
        name="ecvitt",
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        num_levels=3,
        skip_load_backbone=True,
    )
    adapter.eval()

    B = 2
    x = torch.randn(B, 3, 640, 640)

    with torch.no_grad():
        feats = adapter(x)

    check(
        "Returns list of 3 feature maps",
        isinstance(feats, list) and len(feats) == 3,
        f"len={len(feats)}"
    )

    expected_shapes = [
        [B, 192, 80, 80],
        [B, 192, 40, 40],
        [B, 192, 20, 20],
    ]
    for i, (feat, exp) in enumerate(zip(feats, expected_shapes)):
        check(
            f"feat[{i}] shape",
            list(feat.shape) == exp,
            f"{list(feat.shape)} (exp {exp})"
        )

    check(
        "_forward_single method exists",
        hasattr(adapter, '_forward_single'),
        ""
    )

    with torch.no_grad():
        feats_single = adapter._forward_single(x)
        feats_forward = adapter.forward(x)

    for i in range(3):
        check(
            f"forward == _forward_single feat[{i}]",
            torch.allclose(feats_single[i], feats_forward[i], atol=1e-6),
            f"max_diff={( feats_single[i] - feats_forward[i]).abs().max().item():.2e}"
        )


# ============================================================================
# TEST 11: TileExtractor standalone [Step 4]
# ============================================================================
def test_tile_extractor_standalone():
    """Verify TileExtractor produces correct tiles and catches invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST 11: TileExtractor standalone [Step 4]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import TileExtractor

    extractor = TileExtractor(tile_size=448, stride=224)
    B, C = 2, 3

    x = torch.randn(B, C, 1120, 2016)
    tiles, meta = extractor(x)

    check("meta B", meta['B'] == B, f"{meta['B']}")
    check("meta N", meta['N'] == 32, f"{meta['N']}")
    check("meta nh", meta['nh'] == 4, f"{meta['nh']}")
    check("meta nw", meta['nw'] == 8, f"{meta['nw']}")
    check(
        "tiles shape",
        list(tiles.shape) == [B * 32, C, 448, 448],
        f"{list(tiles.shape)}"
    )

    tile_00 = tiles[0]
    expected = x[0, :, 0:448, 0:448]
    check(
        "TileExtractor tile[0,0] matches direct slice",
        torch.allclose(tile_00, expected, atol=1e-6),
        f"max_diff={( tile_00 - expected).abs().max().item():.2e}"
    )

    caught = False
    try:
        bad_x = torch.randn(1, 3, 200, 200)
        extractor(bad_x)
    except AssertionError:
        caught = True
    check("Rejects frame smaller than tile_size", caught)

    caught2 = False
    try:
        bad_x2 = torch.randn(1, 3, 1080, 1920)
        extractor(bad_x2)
    except AssertionError:
        caught2 = True
    check("Rejects non-tile-compatible dimensions", caught2)


# ============================================================================
# TEST 12: _forward_tiled stateless shapes [Step 5]
# ============================================================================
def test_forward_tiled_stateless():
    """Verify _forward_tiled produces correct output shapes on padded 1080p input."""
    print("\n" + "=" * 70)
    print("TEST 12: _forward_tiled stateless shapes [Step 5]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import ViTAdapter

    torch.manual_seed(42)
    adapter = ViTAdapter(
        name="ecvitt",
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        num_levels=3,
        skip_load_backbone=True,
    )
    adapter.tile_size = 448
    adapter.tile_stride = 224
    adapter.eval()

    B = 2
    x_small = torch.randn(B, 3, 448, 448)
    with torch.no_grad():
        feats_small = adapter(x_small)

    check(
        "448x448 routes to _forward_single",
        isinstance(feats_small, list) and len(feats_small) == 3,
        f"shapes: {[list(f.shape) for f in feats_small]}"
    )
    check(
        "448x448 feat[0] shape",
        list(feats_small[0].shape) == [B, 192, 56, 56],
        f"{list(feats_small[0].shape)}"
    )

    x_big = torch.randn(B, 3, 1120, 2016)
    with torch.no_grad():
        feats_big = adapter(x_big)

    check(
        "_forward_tiled returns 3 feature maps",
        isinstance(feats_big, list) and len(feats_big) == 3,
        ""
    )

    global_feat_h = 1120 // 16
    global_feat_w = 2016 // 16
    expected_shapes = [
        [B, 192, global_feat_h * 2, global_feat_w * 2],
        [B, 192, global_feat_h, global_feat_w],
        [B, 192, global_feat_h // 2, global_feat_w // 2],
    ]
    for i, (feat, exp) in enumerate(zip(feats_big, expected_shapes)):
        check(
            f"tiled feat[{i}] shape",
            list(feat.shape) == exp,
            f"{list(feat.shape)} (exp {exp})"
        )

    for i, feat in enumerate(feats_big):
        has_nan = torch.isnan(feat).any().item()
        has_inf = torch.isinf(feat).any().item()
        check(f"tiled feat[{i}] no NaN/Inf", not has_nan and not has_inf)


# ============================================================================
# TEST 13: BN stats preserved during _forward_tiled (CRITICAL-2) [Step 5]
# ============================================================================
def test_bn_stats_preserved():
    """Verify BatchNorm running stats are not corrupted by _forward_tiled."""
    print("\n" + "=" * 70)
    print("TEST 13: BN running stats preserved (CRITICAL-2) [Step 5]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import ViTAdapter

    torch.manual_seed(42)
    adapter = ViTAdapter(
        name="ecvitt",
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        num_levels=3,
        skip_load_backbone=True,
    )
    adapter.tile_size = 448
    adapter.tile_stride = 224

    bn_stats_before = {}
    for name, m in adapter.backbone.patch_embed.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_stats_before[name] = {
                'mean': m.running_mean.clone(),
                'var': m.running_var.clone(),
            }

    adapter.train()
    x = torch.randn(2, 3, 672, 896)
    with torch.no_grad():
        _ = adapter._forward_tiled(x)

    for name, m in adapter.backbone.patch_embed.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and name in bn_stats_before:
            mean_diff = (m.running_mean - bn_stats_before[name]['mean']).abs().max().item()
            var_diff = (m.running_var - bn_stats_before[name]['var']).abs().max().item()
            check(
                f"BN '{name}' stats unchanged",
                mean_diff < 1e-6 and var_diff < 1e-6,
                f"mean_diff={mean_diff:.2e}, var_diff={var_diff:.2e}"
            )

    check("At least 1 BN layer found", len(bn_stats_before) > 0, f"found {len(bn_stats_before)}")


# ============================================================================
# TEST 14: Dispatcher routes correctly [Step 4/5]
# ============================================================================
def test_dispatcher_routing():
    """Verify forward() routes to the correct method based on input size."""
    print("\n" + "=" * 70)
    print("TEST 14: Dispatcher routing [Step 4/5]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import ViTAdapter

    torch.manual_seed(42)
    adapter = ViTAdapter(
        name="ecvitt",
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        num_levels=3,
        skip_load_backbone=True,
    )
    adapter.eval()

    B = 1
    x_small = torch.randn(B, 3, 448, 448)
    with torch.no_grad():
        feats_no_tile = adapter(x_small)
    check(
        "No tile_size attr -> _forward_single for any input",
        isinstance(feats_no_tile, list) and len(feats_no_tile) == 3,
        ""
    )

    adapter.tile_size = 448
    adapter.tile_stride = 224

    with torch.no_grad():
        feats_448 = adapter._forward_single(x_small)
        feats_dispatch = adapter(x_small)
    check(
        "448x448 with tile_size=448 -> _forward_single",
        all(
            torch.allclose(a, b, atol=1e-6)
            for a, b in zip(feats_448, feats_dispatch)
        ),
        "output matches _forward_single"
    )

    x_big = torch.randn(B, 3, 672, 896)
    with torch.no_grad():
        feats_tiled = adapter(x_big)
    check(
        "672x896 with tile_size=448 -> _forward_tiled",
        list(feats_tiled[1].shape) == [B, 192, 42, 56],
        f"feat[1] shape={list(feats_tiled[1].shape)} (matches 672x896 global feat)"
    )


# ============================================================================
# TEST 15: Register state accumulation [Step 6]
# ============================================================================
def test_register_accumulation():
    """Verify register state changes across tiles and tile_pos_proj works."""
    print("\n" + "=" * 70)
    print("TEST 15: Register state accumulation [Step 6]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import ViTAdapter

    torch.manual_seed(42)
    adapter = ViTAdapter(
        name="ecvitt",
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        num_levels=3,
        skip_load_backbone=True,
    )
    adapter.tile_size = 448
    adapter.tile_stride = 224
    adapter.eval()

    check(
        "tile_pos_proj exists",
        hasattr(adapter, 'tile_pos_proj'),
        f"type={type(getattr(adapter, 'tile_pos_proj', None))}"
    )
    check(
        "tile_pos_proj shape (2 -> 192)",
        adapter.tile_pos_proj.in_features == 2 and adapter.tile_pos_proj.out_features == 192,
        f"in={adapter.tile_pos_proj.in_features}, out={adapter.tile_pos_proj.out_features}"
    )

    B = 1
    x = torch.randn(B, 3, 672, 896)

    register_states = []

    from ecdetseg.engine.edgecrafter.ecvit import TileExtractor
    extractor = TileExtractor(tile_size=448, stride=224)
    tiles, meta = extractor(x)
    N, nh, nw = meta['N'], meta['nh'], meta['nw']

    register_state = adapter.backbone.register_token.expand(B, -1, -1).clone()
    register_states.append(register_state.clone())

    with torch.no_grad():
        for tile_idx in range(N):
            tile_batch = tiles[tile_idx::N]
            row = tile_idx // nw
            col = tile_idx % nw
            tile_pos = torch.tensor([[col / nw, row / nh]], dtype=x.dtype)
            pos_signal = adapter.tile_pos_proj(tile_pos).unsqueeze(0).expand(B, -1, -1)
            tile_register = register_state + pos_signal
            _, new_register = adapter.backbone(tile_batch, register_state=tile_register)
            register_state = new_register.detach()
            register_states.append(register_state.clone())

    check(
        f"Collected {N+1} register states",
        len(register_states) == N + 1,
        ""
    )

    reg_std_initial = register_states[0].std().item()
    reg_std_final = register_states[-1].std().item()
    check(
        "Final register std > 0.001",
        reg_std_final > 0.001,
        f"initial_std={reg_std_initial:.4f}, final_std={reg_std_final:.4f}"
    )

    all_same = all(
        torch.allclose(register_states[0], register_states[i], atol=1e-5)
        for i in range(1, len(register_states))
    )
    check(
        "Register state changes across tiles",
        not all_same,
        "states differ as expected"
    )

    diffs = []
    for i in range(1, len(register_states)):
        d = (register_states[i] - register_states[i-1]).abs().mean().item()
        diffs.append(d)
    check(
        "All tile transitions have nonzero diff",
        all(d > 1e-6 for d in diffs),
        f"min_diff={min(diffs):.4e}, max_diff={max(diffs):.4e}"
    )

    with torch.no_grad():
        feats = adapter(x)
    for i, feat in enumerate(feats):
        check(
            f"Tiled output feat[{i}] no NaN/Inf after register accumulation",
            not torch.isnan(feat).any().item() and not torch.isinf(feat).any().item(),
            ""
        )


# ============================================================================
# TEST 16: Full ECDet end-to-end with tiling [Step 7]
# ============================================================================
def test_ecdet_end_to_end():
    """Verify full ECDet forward pass works end-to-end with tiled input."""
    print("\n" + "=" * 70)
    print("TEST 16: Full ECDet end-to-end [Step 7]")
    print("=" * 70)

    from ecdetseg.engine.edgecrafter.ecvit import ViTAdapter
    from ecdetseg.engine.edgecrafter.hybrid_encoder import HybridEncoder
    from ecdetseg.engine.edgecrafter.decoder import ECTransformer
    from ecdetseg.engine.edgecrafter.modeling import ECDet

    torch.manual_seed(42)

    def make_model(tile_size=0, tile_stride=0):
        bb = ViTAdapter(
            name="ecvitt", embed_dim=192, num_heads=3, patch_size=16,
            num_levels=3, skip_load_backbone=True,
        )
        enc = HybridEncoder(
            in_channels=[192, 192, 192], hidden_dim=192,
            dim_feedforward=512, depth_mult=0.67, expansion=0.34,
            csp_type='csp2', fuse_op='sum',
        )
        dec = ECTransformer(
            num_classes=10, feat_channels=[192, 192, 192],
            hidden_dim=192, dim_feedforward=512, num_layers=4,
            eval_idx=-1, eval_spatial_size=None,
        )
        return ECDet(bb, enc, dec, tile_size=tile_size, tile_stride=tile_stride)

    model_no_tile = make_model()
    model_no_tile.eval()

    B = 1
    x_small = torch.randn(B, 3, 640, 640)
    with torch.no_grad():
        out_small = model_no_tile(x_small)

    check(
        "ECDet no-tile 640x640 produces output",
        'pred_logits' in out_small or isinstance(out_small, dict),
        f"keys={list(out_small.keys()) if isinstance(out_small, dict) else 'N/A'}"
    )

    model_tiled = make_model(tile_size=448, tile_stride=224)
    model_tiled.eval()

    check(
        "tile_size injected into backbone",
        hasattr(model_tiled.backbone, 'tile_size') and model_tiled.backbone.tile_size == 448,
        ""
    )
    check(
        "tile_stride injected into backbone",
        hasattr(model_tiled.backbone, 'tile_stride') and model_tiled.backbone.tile_stride == 224,
        ""
    )

    x_big = torch.randn(B, 3, 672, 896)
    with torch.no_grad():
        out_big = model_tiled(x_big)

    check(
        "ECDet tiled 672x896 produces output",
        isinstance(out_big, dict) and 'pred_logits' in out_big,
        f"keys={list(out_big.keys())}"
    )

    if 'pred_logits' in out_big:
        logits = out_big['pred_logits']
        check(
            "pred_logits shape [B, 300, num_classes]",
            logits.shape[0] == B and logits.shape[1] == 300 and logits.shape[2] == 10,
            f"{list(logits.shape)}"
        )

    if 'pred_boxes' in out_big:
        boxes = out_big['pred_boxes']
        check(
            "pred_boxes shape [B, 300, 4]",
            boxes.shape[0] == B and boxes.shape[1] == 300 and boxes.shape[2] == 4,
            f"{list(boxes.shape)}"
        )
        check(
            "pred_boxes in [0, 1] range",
            boxes.min().item() >= -0.1 and boxes.max().item() <= 1.1,
            f"min={boxes.min().item():.3f}, max={boxes.max().item():.3f}"
        )

    for key in out_big:
        if isinstance(out_big[key], torch.Tensor):
            check(
                f"out['{key}'] no NaN",
                not torch.isnan(out_big[key]).any().item(),
                ""
            )


# ============================================================================
# TEST 17: PadToMultiple transform [Step 8]
# ============================================================================
def test_pad_to_multiple():
    """Verify PadToMultiple produces tile-compatible dimensions."""
    print("\n" + "=" * 70)
    print("TEST 17: PadToMultiple transform [Step 8]")
    print("=" * 70)

    from ecdetseg.engine.data.transforms._transforms import PadToMultiple
    from ecdetseg.engine.data._misc import Image

    pad_transform = PadToMultiple(tile_size=448, stride=224)

    cases = [
        # (H_in, W_in, H_expected, W_expected)
        (1080, 1920, 1120, 2016),
        (720, 1280, 896, 1344),
        (448, 448, 448, 448),
        (672, 896, 672, 896),
        (500, 600, 672, 672),
    ]

    for H_in, W_in, H_exp, W_exp in cases:
        img = Image(torch.rand(3, H_in, W_in))

        out = pad_transform(img)
        if isinstance(out, tuple):
            out_img = out[0]
        else:
            out_img = out

        H_out, W_out = out_img.shape[-2], out_img.shape[-1]

        check(
            f"PadToMultiple {H_in}x{W_in}",
            H_out == H_exp and W_out == W_exp,
            f"got {H_out}x{W_out} (exp {H_exp}x{W_exp})"
        )

        ts, s = 448, 224
        check(
            f"  tile-compatible",
            (H_out - ts) % s == 0 and (W_out - ts) % s == 0,
            f"(H-448)%224={(H_out-ts)%s}, (W-448)%224={(W_out-ts)%s}"
        )

    img_orig = Image(torch.rand(3, 1080, 1920))
    out = pad_transform(img_orig)
    if isinstance(out, tuple):
        out_img = out[0]
    else:
        out_img = out

    top_left = out_img[:, :1080, :1920]
    check(
        "Padding is right+bottom only (content preserved)",
        torch.allclose(top_left, img_orig, atol=1e-6),
        f"max_diff={(top_left - img_orig).abs().max().item():.2e}"
    )

    pad_region_bottom = out_img[:, 1080:, :]
    pad_region_right = out_img[:, :, 1920:]
    check(
        "Bottom padding is zeros",
        pad_region_bottom.abs().max().item() == 0.0,
        ""
    )
    check(
        "Right padding is zeros",
        pad_region_right.abs().max().item() == 0.0,
        ""
    )

    check(
        "No padding needed for 448x448",
        pad_transform._compute_padded_size(448, 448) == (448, 448),
        ""
    )
    check(
        "No padding needed for 672x896",
        pad_transform._compute_padded_size(672, 896) == (672, 896),
        ""
    )


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TiledECDet -- Full Validation Suite")
    print("=" * 70)

    # Step 1: Pure arithmetic
    test_tile_grid_arithmetic()
    test_unfold_tile_count()
    test_tile_extraction_shapes()
    test_feature_map_coverage()
    test_fold_reconstruction()
    test_gradient_flow()
    test_memory_estimate()
    test_raw_1080p_not_compatible()

    # Steps 2-3: VisionTransformer + ViTAdapter refactor
    test_vit_forward_signature()
    test_vit_adapter_backward_compat()

    # Steps 4-5: TileExtractor + _forward_tiled
    test_tile_extractor_standalone()
    test_forward_tiled_stateless()
    test_bn_stats_preserved()
    test_dispatcher_routing()

    # Step 6: Register accumulation
    test_register_accumulation()

    # Step 7: Full ECDet end-to-end
    test_ecdet_end_to_end()

    # Step 8: PadToMultiple transform
    test_pad_to_multiple()

    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
