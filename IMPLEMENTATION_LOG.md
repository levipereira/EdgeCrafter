# TiledECDet — Implementation Log

Step-by-step implementation record. Each step was independently verified
before moving to the next.

**Environment:** PyTorch 2.11.0+cu130, CUDA available, conda `ec`
**Test suite:** `ecdetseg/tools/test_tiled_forward.py` — 112 tests, all passing.

---

## Step 0 — Setup [DONE]

- Copied `KNOWN_ISSUES.md`, `DECISIONS.md`, `TILED_ECDET.md` from `GPT/` to repo root.
- Verified conda `ec` env: PyTorch 2.11.0+cu130, CUDA available.

---

## Step 1 — Tile arithmetic validation [DONE]

**Key finding:** Original docs assumed 1080x1920 was tile-compatible with
tile_size=448, stride=224, producing 3x7=21 tiles. This is WRONG.

The actual tile coverage for raw 1080x1920 is only 78% (184px uncovered
bottom, 128px uncovered right). The frame MUST be padded to 1120x2016,
giving 4x8=32 tiles with 100% coverage.

**Corrected tile counts:**
- 1080x1920 -> pad to 1120x2016: 4x8 = **32 tiles** (not 21)
- 720x1280 -> pad to 896x1344: 3x5 = 15 tiles
- 448x448: 1x1 = 1 tile (no padding needed)

**Tests:** 8 arithmetic tests, all pass.

---

## Step 2 — VisionTransformer.forward signature [DONE]

**Change:** `forward(self, x)` -> `forward(self, x, register_state=None)`
**Return:** `outs` -> `(outs, final_register)`

With `register_state=None`, output is numerically identical to original.
Custom `register_state` correctly alters output.

**File:** `ecdetseg/engine/edgecrafter/ecvit.py`
**Tests:** 7 tests, all pass.

---

## Step 3 — ViTAdapter forward split [DONE]

**Change:** `forward()` -> `forward()` dispatcher + `_forward_single()`
Updated `self.backbone(x)` call to unpack `(return_layers, _)`.

**Verified:** `_forward_single` output is bitwise identical to original `forward`.

**File:** `ecdetseg/engine/edgecrafter/ecvit.py`
**Tests:** 8 tests, all pass.

---

## Step 4 — TileExtractor [DONE]

**New class:** `TileExtractor(nn.Module)` using `F.unfold`.
Validates tile-compatibility of input dimensions.
Rejects frames smaller than tile_size or non-compatible dimensions.

**File:** `ecdetseg/engine/edgecrafter/ecvit.py`
**Tests:** 8 tests, all pass.

---

## Step 5 — _forward_tiled stateless [DONE]

**New method:** `ViTAdapter._forward_tiled(x)` with sequential tile loop.
Initially stateless (no register accumulation) — internal SAHI equivalent.

**CRITICAL-2 fix included:** BatchNorm in patch_embed forced to eval() during
tile loop. Verified: BN running stats drift = 0.00 across 3 BN layers.

**Dispatcher:** `forward()` routes to `_forward_tiled` when input > tile_size.

**Output shapes for 1120x2016 input:**
- feat[0]: [B, 192, 140, 252] (stride 8)
- feat[1]: [B, 192, 70, 126] (stride 16)
- feat[2]: [B, 192, 35, 63] (stride 32)

**File:** `ecdetseg/engine/edgecrafter/ecvit.py`
**Tests:** 12 tests, all pass.

---

## Step 6 — Register accumulation [DONE]

**Changes:**
- Register state passed between tiles with `detach()` (truncated BPTT)
- `tile_pos_proj = nn.Linear(2, 192)` added to ViTAdapter (576 params)
- Normalized tile position `(col/nw, row/nh)` injected into register before each tile

**Verified:**
- `register_state.std()` = 3.57 after all tiles (well above 0.001 threshold)
- All tile transitions have nonzero diff (min=0.45, max=0.55)
- No NaN/Inf in output

**File:** `ecdetseg/engine/edgecrafter/ecvit.py`
**Tests:** 9 tests, all pass.

---

## Step 7 — ECDet end-to-end [DONE]

**Change:** `ECDet.__init__` accepts `tile_size` and `tile_stride`, injects
into backbone.

**Verified full pipeline:**
- 640x640 standard path: pred_logits [1, 300, 10], pred_boxes [1, 300, 4]
- 672x896 tiled path: same output structure, boxes in [0.013, 0.982] range
- No NaN anywhere

**File:** `ecdetseg/engine/edgecrafter/modeling.py`
**Tests:** 9 tests, all pass.

---

## Step 8 — PadToMultiple transform [DONE]

**New class:** `PadToMultiple(nn.Module)` — pads right+bottom only.
Must be placed after `ConvertPILImage` in the transform pipeline.

**Verified:**
- Correct dimensions for 5 test cases (1080p, 720p, 448, 672, 500)
- Content preserved in top-left region (max_diff = 0.00)
- Padding regions are zeros
- All padded sizes are tile-compatible

**File:** `ecdetseg/engine/data/transforms/_transforms.py`
**Tests:** 14 tests, all pass.

---

## Step 9 — Config YAML + training [PENDING]

**Requires:** VisDrone 2019 DET dataset and pretrained ECDet-S weights.
**Config:** `ecdetseg/configs/ecdet/ecdet_s_visdrone_tiled.yml`

---

## Summary of modified files

| File | Changes |
|------|---------|
| `ecdetseg/engine/edgecrafter/ecvit.py` | TileExtractor, VisionTransformer.forward signature, ViTAdapter._forward_single/_forward_tiled/forward dispatcher, tile_pos_proj |
| `ecdetseg/engine/edgecrafter/modeling.py` | ECDet.__init__ accepts tile_size/tile_stride |
| `ecdetseg/engine/data/transforms/_transforms.py` | PadToMultiple class |
| `ecdetseg/tools/test_tiled_forward.py` | 112 validation tests |

## New parameters added

| Parameter | Location | Shape | Count |
|-----------|----------|-------|-------|
| `tile_pos_proj.weight` | `ViTAdapter` | [192, 2] | 384 |
| `tile_pos_proj.bias` | `ViTAdapter` | [192] | 192 |
| **Total** | | | **576** |
