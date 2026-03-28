# TiledECDet — Project Context

## What this project is

A modification of **ECDet-S** (EdgeCrafter, Intellindust AI Lab, 2026) to detect small
objects in high-resolution aerial frames (1080p) without losing information through
downscaling.

**Core hypothesis:** processing a frame through sequential internal tiles while
accumulating global context in the ViT register token allows a compact model (~10M params)
to learn detection as if it were seeing the full high-resolution frame — without depending
on external SAHI preprocessing.

**Target application:** vehicle detection in UAV aerial imagery (VisDrone 2019 DET /
UAVDT datasets) for the Garver project — an aerial traffic perception system.

**Author fork:** `https://github.com/levipereira/EdgeCrafter`
**Upstream reference:** `https://github.com/Intellindust-AI-Lab/EdgeCrafter`

---

## Repository structure

```
EdgeCrafter/
├── .cursor/
│   └── rules/
│       ├── cursor_rules_tiled_ecdet.mdc      # always-on global rules
│       └── cursor_skills_tiled_ecdet.mdc     # glob-activated task skills
├── TILED_ECDET.md                            # this file — full project context
├── KNOWN_ISSUES.md                           # problem log (update immediately on discovery)
├── DECISIONS.md                              # architectural decision log
└── ecdetseg/
    ├── engine/
    │   ├── edgecrafter/
    │   │   ├── ecvit.py          # ← MODIFY: TileExtractor, VisionTransformer, ViTAdapter
    │   │   ├── hybrid_encoder.py # do not modify
    │   │   ├── decoder.py        # do not modify
    │   │   ├── modeling.py       # ← MODIFY: ECDet.__init__ only
    │   │   └── postprocessor.py  # do not modify
    │   ├── data/
    │   │   ├── dataloader.py     # do not modify
    │   │   └── transforms/
    │   │       └── _transforms.py # ← MODIFY: add PadToMultiple
    │   └── solver/
    │       └── ec_engine.py      # do not modify
    ├── configs/
    │   └── ecdet/
    │       ├── ecdet_s.yml                    # original config — do not modify
    │       └── ecdet_s_visdrone_tiled.yml     # ← CREATE
    └── tools/
        └── test_tiled_forward.py             # ← CREATE
```

---

## Current architecture — exact code behavior

### Full forward pass

```
input [B, 3, 640, 640]          ← resize done EXTERNALLY before model
    ↓
ViTAdapter.forward(x)
    ↓ ConvPyramidPatchEmbed       → stride=16, output [B, 192, 40, 40]
    ↓ VisionTransformer           → 12 blocks, RoPE 2D, fixed register_token
    ↓ fuse return_layers [10,11]  → mean, reshape [B, 192, 40, 40]
    ↓ interpolate ×2, ×1, ×0.5   → 3 feature maps: [B,192,80,80] [40,40] [20,20]
    ↓
HybridEncoder(proj_feats)        → FPN top-down + PAN bottom-up
    ↓ 3 levels [B, 256, H, W]
    ↓
ECTransformer(feats, targets)    → 300 DETR-style queries, deformable attention
    ↓ pred_logits, pred_boxes (normalized [0,1])
    ↓
PostProcessor                    → boxes × orig_target_sizes → absolute coords
```

### VisionTransformer.forward (ecvit.py line 326) — current code

```python
def forward(self, x):
    outs = []
    x_embed = self.patch_embed(x)
    _, _, H, W = x_embed.shape                           # H=W=40 for 640×640 input
    x_embed = x_embed.flatten(2).transpose(1, 2)         # [B, 1600, 192]
    register_token = self.register_token.expand(B, -1, -1)  # [B, 1, 192] — fixed param
    x = torch.cat((register_token, x_embed), dim=1)      # [B, 1601, 192]
    rope_sincos = self.rope_embed(H=H, W=W)
    for i, blk in enumerate(self.blocks):
        x = blk(x, rope_sincos=rope_sincos)
        if i in self.return_layers:                      # [10, 11]
            outs.append(x[:, 1:])                       # excludes register from output
    return outs
```

### ViTAdapter.forward (ecvit.py line 475) — current code

```python
def forward(self, x):
    H_c = x.shape[2] // self.patch_size  # 40
    W_c = x.shape[3] // self.patch_size  # 40
    bs  = x.shape[0]
    return_layers = self.backbone(x)
    fused = torch.mean(torch.stack(return_layers), dim=0)
    fused = fused.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)
    proj_feats = []
    for i in range(self.num_levels):          # 3 levels
        scale = 2 ** (1 - i)                 # 2.0, 1.0, 0.5
        feat = F.interpolate(fused, [int(H_c*scale), int(W_c*scale)], mode='bilinear')
        proj_feats.append(feat)
    proj_feats = [proj(f) for proj, f in zip(self.projector, proj_feats)]
    return proj_feats
```

### RoPE — why it is tile-blind (CRITICAL-3)

```python
# normalize_coords="separate" — coords always in [0,1] relative to tile dimensions
coords_h = torch.arange(0.5, H) / H
coords_w = torch.arange(0.5, W) / W
# token (0,0) of tile at grid position (row=0,col=0) gets the same RoPE as
# token (0,0) of tile at grid position (row=2,col=6)
```

### ECTransformer anchor generation — why decoder needs no modification

```python
# training time: always dynamic, uses actual feature map spatial_shapes
if self.training or self.eval_spatial_size is None:
    anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
# inference with eval_spatial_size set: uses cached anchors (must remain unset for tiled)
else:
    anchors = self.anchors
```

---

## Target architecture — TiledECDet

```
input [B, 3, 1920, 1080]         ← full resolution frame, no external resize
    ↓
ViTAdapter.forward(x)             ← dispatcher: routes to _forward_tiled
    ↓
    TileExtractor (F.unfold)
        → 21 tiles [B*21, 3, 448, 448]  (3×7 grid, 50% overlap)
    ↓
    for tile_idx in range(21):
        tile_pos_proj([col/nw, row/nh]) → added to register_state (CRITICAL-3 fix)
        backbone(tile, register_state) → (features, new_register)
        register_state = new_register.detach()  ← truncated BPTT
        accumulate tile features → global_feat[y0:y1, x0:x1]
    ↓
    global_feat /= global_count          ← average overlapping regions
    → 3 feature maps: [B,192,134,160] [67,120] [33,60]  (proportional to 1080p)
    ↓
HybridEncoder(proj_feats)                ← unchanged, larger feature maps
    ↓
ECTransformer(feats, targets)            ← unchanged, dynamic anchors for new shapes
    ↓ pred_boxes normalized [0,1] in 1080p frame space
    ↓
PostProcessor(orig_target_sizes=[1920,1080]) ← unchanged
    → absolute coordinates in 1080p space
```

---

## Tile arithmetic for 1920×1080

```
tile_size = 448, stride = 224, patch_size = 16

Frame:   1920 × 1080
Grid:    nh = (1080-448)//224 + 1 = 3
         nw = (1920-448)//224 + 1 = 7
         N  = 3 × 7 = 21 tiles

Feature space (÷16):
  tile:    28 × 28 tokens
  global: 120 × 67 tokens = 8,040 tokens at stride-8 level
  vs COCO: 40 × 40 = 1,600 tokens

Decoder anchor count:
  level 0 (stride 8):   120×67   = 8,040
  level 1 (stride 16):   60×34   = 2,040
  level 2 (stride 32):   30×17   =   510
  total: 10,590 anchors  (vs ~1,600 from COCO training)
```

---

## Known problems

Full details in `KNOWN_ISSUES.md`. Summary:

**CRITICAL-1:** `BatchImageCollateFunction` crashes without Resize — frames have variable
sizes. Fix: `PadToMultiple(stride=224)` as first transform.

**CRITICAL-2:** `BatchNorm2d` in `ConvPyramidPatchEmbed` corrupts pretrained running stats
when processing tiles with small B. Fix: `m.eval()` on all BN in `patch_embed` inside
`_forward_tiled`.

**CRITICAL-3:** `RopePositionEmbedding` with `normalize_coords="separate"` is tile-relative
only — no global position awareness. Fix: `tile_pos_proj = nn.Linear(2, 192)` injects
normalized tile grid coordinates into register state before each tile.

**SERIOUS-1:** Decoder sees ~10,590 anchors vs ~1,600 from COCO. Fix: two-phase fine-tune,
Phase 1 decoder-only (backbone frozen), Phase 2 full training with backbone LR 50× lower.

**SERIOUS-2:** Simple average in overlap regions creates feature discontinuities on tile
borders. Fix: Gaussian-weighted reassemble (`sigma = tile_feat_size / 3`).

**SERIOUS-3:** Register token (192 dims) cannot fully capture 21 tiles of information.
Diagnostic: `register_state.std()` per epoch — collapse to <0.001 means stateless mode.

---

## Training strategy

### Phase 0 — Required baseline (before any modification)

Train unmodified ECDet-S on VisDrone with external SAHI:
- Tile VisDrone train offline: `tile_size=448`, `overlap=0.5`
- Train with `Resize→448`, standard pipeline
- Evaluate val with SAHI inference, same tile params
- Record: mAP@0.5, AP_small (area < 32²px in 1080p space)

This is the number TiledECDet must beat.

**Reference result (same approach, GELAN-C backbone):**
Full-frame training: mAP@0.5 = 0.485 after 140 epochs.
Sliced-tile fine-tune: mAP@0.5 = 0.859 after only 40 additional epochs.

### Phase 1 — Decoder recalibration (epochs 1-20)

Backbone frozen. Only `HybridEncoder` and `ECTransformer` train.
Goal: re-calibrate deformable attention offsets for 1080p-scale feature maps.

### Phase 2 — Full fine-tune (epochs 21-50)

All parameters train. Backbone LR = decoder LR / 50.
Monitor `register_state.std()` — must stay above 0.001.

---

## Decisions log summary

Full rationale in `DECISIONS.md`.

| ID | Decision |
|---|---|
| DEC-001 | Sequential tile loop (not batched) — avoids OOM at B≥2 |
| DEC-002 | Truncated BPTT with detach() — prevents gradient explosion over 21 steps |
| DEC-003 | Register token as inter-tile state — zero new backbone params |
| DEC-004 | tile_pos_proj in ViTAdapter — 384 params, compensates RoPE blindness |
| DEC-005 | Simple average reassemble first, Gaussian second — easier to debug |
| DEC-006 | Two-phase fine-tune — prevents competing gradients at scale mismatch |
| DEC-007 | PadToMultiple transform — isolates fix to tiled config, no collate changes |
