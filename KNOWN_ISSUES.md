# Known Issues — TiledECDet

All problems discovered during implementation must be logged here immediately,
in the same session the problem is found. Never defer.

Format: ID, severity, title, root cause, affected files, fix status, fix or workaround.

Severity: **CRITICAL** (blocks training) | **SERIOUS** (trains but converges poorly) | **MANAGEABLE** (efficiency or quality impact, does not block)

---

## CRITICAL-1: DataLoader crashes without Resize

**Status:** Open — fix designed, not yet implemented
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
`BatchImageCollateFunction.__call__` (engine/data/dataloader.py line 142):
```python
images = torch.cat([x[0][None] for x in items], dim=0)
```
Requires all frames in the batch to have identical shapes. VisDrone 2019 DET frames
have variable sizes (typically 1920×1080 but not guaranteed). Removing `Resize→640`
from the transform pipeline causes an immediate crash before the first forward pass.

**Affected files:**
- `engine/data/dataloader.py` line 142 — do not modify (collate logic is correct)
- `engine/data/transforms/_transforms.py` — add fix here
- `configs/ecdet/ecdet_s_visdrone_tiled.yml` — use fix as first transform

**Fix:**
Implement `PadToMultiple(tile_size, stride)` transform. Pads frame to the nearest size
where `(H - tile_size) % stride == 0`. When `tile_size` is a multiple of `stride`
(448 = 2*224), this simplifies to `H % stride == 0`. Pad right and bottom only — never
centered. Normalized bbox coordinates remain valid without adjustment because right/bottom
padding adds space after objects, not before them.

**Corrected tile counts (after padding):**
- 1080x1920 -> 1120x2016: 4x8 = 32 tiles (not 3x7 = 21 as originally estimated)
- 720x1280 -> 896x1344: 3x5 = 15 tiles
- 448x448: 1x1 = 1 tile (no padding needed)

---

## CRITICAL-2: BatchNorm in PatchEmbed corrupts pretrained weights

**Status:** Open — fix designed, not yet implemented
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
`ConvPyramidPatchEmbed` uses `ConvNormLayer_fuse` (ecvit.py lines 151-172, imported
from hybrid_encoder.py) which contains `nn.BatchNorm2d`. When `_forward_tiled` runs
with B=2 frames and 21 sequential tiles, the BN layer sees only 2 samples per forward
call instead of a full batch. With `model.train()` active, BN updates its running mean
and variance after every tile, progressively overwriting the COCO-pretrained statistics.
By the end of the first epoch, the PatchEmbed BN running stats are completely corrupted.

**Affected files:**
- `engine/edgecrafter/ecvit.py` — `ConvPyramidPatchEmbed` (lines 151-172)
- `engine/edgecrafter/hybrid_encoder.py` — `ConvNormLayer_fuse` (lines 25-47, source of BN)

**Fix:**
In `ViTAdapter._forward_tiled`, freeze all BN layers in `patch_embed` before the tile loop:
```python
for m in self.backbone.patch_embed.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
```
This preserves pretrained running stats without freezing the conv weights themselves.
The BN in `HybridEncoder` CSP blocks is unaffected — those blocks receive the fully
reassembled global feature map, one forward per batch item, not per tile.

---

## CRITICAL-3: RoPE has no global position awareness

**Status:** Open — fix designed, not yet implemented
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
`RopePositionEmbedding` with `normalize_coords="separate"` (ecvit.py lines 82-84)
computes position coordinates as:
```python
coords_h = torch.arange(0.5, H) / H   # always in [0,1] relative to tile height
coords_w = torch.arange(0.5, W) / W   # always in [0,1] relative to tile width
```
Token at position (0,0) of every tile receives RoPE coordinates `[-1, -1]` regardless
of where that tile originates in the full frame. The backbone cannot distinguish a tile
at frame grid position (row=0, col=0) from one at (row=2, col=6). Without positional
context, the register token accumulation has no spatial ordering — all tiles look locally
identical to the model.

**Affected files:**
- `engine/edgecrafter/ecvit.py` — `RopePositionEmbedding.forward` (lines 73-108)
- `engine/edgecrafter/ecvit.py` — `ViTAdapter.__init__` and `_forward_tiled` (add fix here)

**Fix:**
Add `tile_pos_proj = nn.Linear(2, embed_dim)` to `ViTAdapter.__init__`.
Before each tile's backbone call in `_forward_tiled`:
```python
tile_pos = torch.tensor([[col / nw, row / nh]], device=x.device, dtype=x.dtype)
register_state = register_state + self.tile_pos_proj(tile_pos).unsqueeze(0)
```
Cost: 384 new parameters (2 × 192). No backbone weights changed. See DEC-004.

---

## SERIOUS-1: Decoder sees 42k anchors vs 1.6k from COCO training

**Status:** Open — mitigation designed (two-phase fine-tune), not yet implemented
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
`MSDeformableAttention` sampling offsets (decoder.py lines 139-163) were initialized
and trained with COCO feature maps of ~1,600 tokens (40×40 at stride 8, producing
~1,600 + ~400 + ~100 = ~2,100 total anchors across 3 levels).

Tiled 1080p mode produces feature maps of approximately:
- Level 0 (stride 8):  120×67 = 8,040 tokens
- Level 1 (stride 16):  60×34 = 2,040 tokens
- Level 2 (stride 32):  30×17 =   510 tokens
- Total: ~10,590 anchors

The deformable attention offset scale and initialization distribution learned on COCO
does not transfer directly. Training backbone and decoder simultaneously from the start
creates competing gradients — decoder loss is dominated by the scale mismatch.

**Affected files:**
- `engine/edgecrafter/decoder.py` — `MSDeformableAttention._reset_parameters` (lines 150-163)
- `configs/ecdet/ecdet_s_visdrone_tiled.yml` — two-phase training schedule

**Mitigation:**
Two-phase fine-tune (see DEC-006): Phase 1 (epochs 1-20) backbone frozen, only
decoder and encoder train. Phase 2 (epochs 21-50) full training with backbone LR
50× lower than decoder. See `DECISIONS.md` DEC-006.

---

## SERIOUS-2: Simple average in tile overlap creates feature discontinuities

**Status:** Open — Gaussian fix designed, simple average implemented as Phase 1
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
In `_forward_tiled`, overlapping tile regions accumulate feature sums divided by a
count tensor (simple average). An object whose bounding box falls on a tile border
receives a feature vector that is the arithmetic mean of two independent backbone
passes — each with a different surrounding context (different neighboring content in
each tile). This feature discontinuity is invisible to the loss function but creates
inconsistency in the feature space that the decoder must learn to compensate for.

**Affected files:**
- `engine/edgecrafter/ecvit.py` — `ViTAdapter._forward_tiled` reassemble block

**Fix:**
Gaussian-weighted accumulation:
```
weight(i, j) = exp(-((i - cy)^2 + (j - cx)^2) / (2 * sigma^2))
cy = tile_feat_h // 2,  cx = tile_feat_w // 2,  sigma = tile_feat_h / 3
```
Replace the uniform `count += 1.0` with `count += weight`.
Objects fully inside the tile center (weight ≈ 1.0) dominate that region's feature.
Border objects receive low weight from each contributing tile. See DEC-005.

---

## SERIOUS-3: Register token capacity is insufficient for 21 tiles

**Status:** Open — diagnostic defined, no structural fix planned yet
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
Each tile produces 784 tokens × 192 embedding dimensions of information.
The register token compresses this through a single attention pass into 1 × 192 dims.
Compression ratio is 784:1 per tile, approximately 16,464:1 for the full 21-tile sequence.
The register token can only retain coarse global scene statistics — altitude appearance,
surface type, traffic density — not object-level spatial features useful for detection.
If the register collapses to a near-constant vector, the implementation degenerates to
stateless internal SAHI, losing the main benefit of the proposed architecture.

**Affected files:**
- `engine/edgecrafter/ecvit.py` — `VisionTransformer.forward` (register state passing)
- `engine/edgecrafter/ecvit.py` — `ViTAdapter._forward_tiled` (accumulation loop)

**Diagnostic:**
Monitor `register_state.std()` per epoch. If `std < 0.001` consistently, accumulation
has collapsed. Log this value in `outputs/logs/` during training.

**Potential future fix:**
Expand register to K tokens (e.g., K=4 or K=8) for higher capacity. Requires changing
`register_token` shape from `[1, 1, 192]` to `[1, K, 192]` and updating all downstream
handling. Document as a new `DECISIONS.md` entry before implementing.

---

## MANAGEABLE-1: Training throughput is approximately 21× slower per step

**Status:** Accepted — no fix planned
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
Sequential tile loop: 21 tiles per 1080p frame processed one at a time. Each tile
requires a full ViT forward pass (~177 MB activations for B=2 across 12 blocks).
Wall-clock time per training step scales linearly with N=21 compared to single-frame mode.

**Mitigation:**
Enable `use_amp: True` in config. Use gradient accumulation to simulate `total_batch_size=32`
with actual `total_batch_size=4`. GPU utilization per tile remains high — the bottleneck
is the number of sequential kernel launches, not GPU efficiency per launch.

---

## MANAGEABLE-2: Aspect ratio mismatch between COCO training and 1080p feature maps

**Status:** Accepted — resolves naturally during fine-tune
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
COCO training used 640×640 inputs producing square feature maps (aspect ratio 1.0).
Tiled 1080p mode produces feature maps of 120×67 at stride 8 (aspect ratio 1.79).
The anchor grid in `_generate_anchors` uses `grid_size=0.05`, producing square anchors
that are suboptimal for the landscape aspect ratio initially.

**Mitigation:**
Fine-tune on VisDrone re-calibrates anchor regression to the landscape aspect ratio
naturally. No code change required.

---

## MANAGEABLE-3: Mosaic and Mixup are incompatible with variable-size tiled frames

**Status:** Resolved by configuration
**Repository:** `https://github.com/levipereira/EdgeCrafter`

**Root cause:**
`Mosaic` augmentation and `Mixup` in `BatchImageCollateFunction` were designed for
fixed 640×640 frames with uniform spatial layout. They are not compatible with
the variable-size, padding-based input used in tiled mode.

**Fix:**
Tiled config (`ecdet_s_visdrone_tiled.yml`) sets `mosaic_prob: 0.0` and `mixup_prob: 0.0`.
These augmentations are excluded from the tiled training pipeline by configuration.
