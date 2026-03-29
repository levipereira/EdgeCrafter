# TiledECDet — Architecture Reference

Complete visual documentation of the TiledECDet architecture, covering every
modified component and the data flow from raw image to detection output.

---

## 1. High-Level Pipeline Overview

The full inference/training pipeline from input image to bounding box predictions.
Components marked with **[MOD]** were modified; **[NEW]** are entirely new.

```mermaid
flowchart TB
    subgraph INPUT["Input Pipeline [MOD]"]
        RAW["Raw Image<br/>(variable size, e.g. 756×1344)"]
        RSZ["Resize<br/>[896, 1344] fixed"]
        CVT["ConvertPILImage<br/>float32, /255"]
        PAD["PadToMultiple [NEW]<br/>tile_size=448, stride=224<br/>→ right+bottom pad only"]
        NORM["Normalize<br/>ImageNet μ/σ"]
    end

    subgraph BACKBONE["ViTAdapter Backbone [MOD]"]
        DISP{"Dispatcher [NEW]<br/>needs_tiling?<br/>is_compatible?"}
        SINGLE["_forward_single<br/>(original path)"]
        TILED["_forward_tiled [NEW]<br/>(sequential tile loop)"]
    end

    subgraph NECK["HybridEncoder (unchanged)"]
        FPN["Feature Pyramid<br/>CSP + cross-scale fusion"]
    end

    subgraph HEAD["ECTransformer Decoder (unchanged)"]
        DEC["Deformable Attention<br/>300 queries → boxes + labels"]
    end

    OUTPUT["Predictions<br/>pred_logits [B, 300, 10]<br/>pred_boxes [B, 300, 4]"]

    RAW --> RSZ --> CVT --> PAD --> NORM
    NORM --> DISP
    DISP -->|"H ≤ tile_size<br/>OR not compatible"| SINGLE
    DISP -->|"H > tile_size<br/>AND compatible"| TILED
    SINGLE --> FPN
    TILED --> FPN
    FPN --> DEC --> OUTPUT

    style INPUT fill:#1a1a2e,stroke:#e94560,color:#fff
    style BACKBONE fill:#16213e,stroke:#0f3460,color:#fff
    style NECK fill:#0f3460,stroke:#533483,color:#fff
    style HEAD fill:#533483,stroke:#e94560,color:#fff
    style PAD fill:#e94560,stroke:#fff,color:#fff
    style TILED fill:#e94560,stroke:#fff,color:#fff
    style DISP fill:#e94560,stroke:#fff,color:#fff
```

---

## 2. ECDet Model Composition

How the top-level `ECDet` wires backbone, encoder, and decoder together.
`tile_size` and `tile_stride` are injected from config into the backbone at init.

```mermaid
classDiagram
    class ECDet {
        +backbone: ViTAdapter
        +encoder: HybridEncoder
        +decoder: ECTransformer
        +tile_size: int = 448
        +tile_stride: int = 224
        +forward(x, targets) → dict
        +forward_features(x) → list
    }

    class ViTAdapter {
        +backbone: VisionTransformer
        +projector: ModuleList[ConvNormLayer]
        +tile_pos_proj: Linear(2, 192) ★NEW
        +tile_size: int ★INJECTED
        +tile_stride: int ★INJECTED
        +forward(x) → list
        +_forward_single(x) → list
        +_forward_tiled(x) → list ★NEW
    }

    class VisionTransformer {
        +patch_embed: ConvPyramidPatchEmbed
        +register_token: Parameter [1,1,192]
        +blocks: ModuleList[Block × 12]
        +rope_embed: RopePositionEmbedding
        +forward(x, register_state?) → (outs, register) ★MOD
    }

    class HybridEncoder {
        +feat_strides: [8, 16, 32]
        +forward(features) → features
    }

    class ECTransformer {
        +num_queries: 300
        +num_classes: 10
        +forward(features, targets) → dict
    }

    ECDet *-- ViTAdapter : backbone
    ECDet *-- HybridEncoder : encoder
    ECDet *-- ECTransformer : decoder
    ViTAdapter *-- VisionTransformer : backbone
```

---

## 3. The Dispatcher — Routing Logic

The dispatcher in `ViTAdapter.forward` decides which path to take based on
the input dimensions and tile configuration.

```mermaid
flowchart TD
    IN["Input x<br/>[B, 3, H, W]"]

    CHECK1{"tile_size and<br/>tile_stride set?"}
    CHECK2{"H > tile_size<br/>OR W > tile_size?"}
    CHECK3{"H ≥ tile_size<br/>AND W ≥ tile_size<br/>AND (H−t) % s == 0<br/>AND (W−t) % s == 0"}

    SINGLE["_forward_single(x)<br/>Standard ViT pass<br/>(e.g. 640×640 profiler input)"]
    TILED["_forward_tiled(x)<br/>Sequential tile loop<br/>(e.g. 896×1344 → 15 tiles)"]

    IN --> CHECK1
    CHECK1 -->|No| SINGLE
    CHECK1 -->|Yes| CHECK2
    CHECK2 -->|No| SINGLE
    CHECK2 -->|Yes| CHECK3
    CHECK3 -->|No — not tile-compatible| SINGLE
    CHECK3 -->|Yes| TILED

    style SINGLE fill:#2d6a4f,stroke:#fff,color:#fff
    style TILED fill:#e94560,stroke:#fff,color:#fff
    style CHECK3 fill:#16213e,stroke:#e94560,color:#fff
```

This prevents the 640×640 profiler input from crashing — 640 is larger than
`tile_size=448` but is not tile-compatible `(640−448) % 224 ≠ 0`.

---

## 4. _forward_single — Original Path (Unchanged Logic)

The standard inference path for small inputs. Numerically identical to the
original `ViTAdapter.forward` from the pretrained model.

```mermaid
flowchart LR
    X["x [B,3,H,W]"] --> VIT["VisionTransformer<br/>12 blocks"]
    VIT -->|"return_layers<br/>[10, 11]"| FUSE["torch.mean<br/>(stack layers)"]
    FUSE --> RESHAPE["reshape to<br/>[B, D, H/16, W/16]"]

    RESHAPE --> S8["F.interpolate ×2<br/>stride 8"]
    RESHAPE --> S16["identity<br/>stride 16"]
    RESHAPE --> S32["F.interpolate ÷2<br/>stride 32"]

    S8 --> PROJ["projector<br/>Conv1×1"]
    S16 --> PROJ
    S32 --> PROJ

    PROJ --> OUT["3 feature maps<br/>strides [8, 16, 32]"]

    style VIT fill:#16213e,stroke:#0f3460,color:#fff
    style FUSE fill:#533483,stroke:#e94560,color:#fff
```

---

## 5. _forward_tiled — The Core Modification

This is the main new code path. It processes a high-resolution frame through
overlapping tiles sequentially, accumulating global context via the register token.

### 5a. Tile Extraction

```mermaid
flowchart TD
    FRAME["Input Frame<br/>[B, 3, 896, 1344]"]

    subgraph EXTRACT["TileExtractor (F.unfold)"]
        UNFOLD["F.unfold<br/>kernel=448, stride=224"]
        RESHAPE["reshape → [B×N, 3, 448, 448]"]
        META["metadata: B, N=15, nh=3, nw=5"]
    end

    GRID["Tile Grid Layout<br/>3 rows × 5 columns<br/>overlap = 224px"]

    FRAME --> UNFOLD --> RESHAPE
    UNFOLD --> META
    RESHAPE --> GRID

    style EXTRACT fill:#1a1a2e,stroke:#e94560,color:#fff
    style GRID fill:#16213e,stroke:#0f3460,color:#fff
```

```
Input 896×1344 → 15 overlapping tiles (3×5)

 ┌────────┬───┬────────┬───┬────────┬───┬────────┬───┬────────┐
 │ tile 0 │   │ tile 1 │   │ tile 2 │   │ tile 3 │   │ tile 4 │  row 0
 │ 448×448│   │        │   │        │   │        │   │        │
 ├────────┤   ├────────┤   ├────────┤   ├────────┤   ├────────┤
 │overlap │224│        │224│        │224│        │224│        │
 ├────────┤   ├────────┤   ├────────┤   ├────────┤   ├────────┤
 │ tile 5 │   │ tile 6 │   │ tile 7 │   │ tile 8 │   │ tile 9 │  row 1
 │        │   │        │   │        │   │        │   │        │
 ├────────┤   ├────────┤   ├────────┤   ├────────┤   ├────────┤
 │ tile 10│   │ tile 11│   │ tile 12│   │ tile 13│   │ tile 14│  row 2
 │        │   │        │   │        │   │        │   │        │
 └────────┘   └────────┘   └────────┘   └────────┘   └────────┘
```

### 5b. Sequential Tile Processing with Register Accumulation

This is the heart of TiledECDet — the tile loop with register state passing.

```mermaid
flowchart TD
    INIT["Initialize:<br/>register_state = register_token.clone()<br/>global_feat = zeros [B, 192, 56, 84]<br/>global_count = zeros [B, 1, 56, 84]"]

    subgraph LOOP["for tile_idx in range(N=15)"]
        direction TB

        SLICE["tile_batch = tiles[tile_idx::N]<br/>[B, 3, 448, 448]"]

        subgraph POSENC["CRITICAL-3 Fix: Global Position Injection"]
            GRIDPOS["tile_pos = [col/nw, row/nh]<br/>normalized grid coordinates"]
            PROJ["tile_pos_proj(tile_pos)<br/>Linear(2→192)"]
            ADD["tile_register =<br/>register_state + pos_signal"]
        end

        subgraph BNFIX["CRITICAL-2 Fix: BN Freeze"]
            BNEVAL["patch_embed BatchNorm → eval()<br/>preserves pretrained running stats"]
        end

        VIT["VisionTransformer.forward<br/>(tile_batch, register_state=tile_register)"]
        RETURNS["returns: (return_layers, new_register)"]

        DETACH["register_state = new_register.detach()<br/>★ Truncated BPTT"]

        FUSE2["fuse return layers → [B, 192, 28, 28]"]

        ACCUM["global_feat[:, :, y0:y1, x0:x1] += fused_tile<br/>global_count[:, :, y0:y1, x0:x1] += 1.0"]
    end

    AVG["global_feat = global_feat / global_count"]

    MULTI["Multi-scale projection<br/>stride 8: interpolate ×2<br/>stride 16: identity<br/>stride 32: interpolate ÷2"]

    PROJECTOR["projector Conv1×1"]

    OUT["3 feature maps<br/>to HybridEncoder"]

    INIT --> LOOP
    SLICE --> GRIDPOS --> PROJ --> ADD
    ADD --> VIT
    BNFIX -.->|"applied before loop"| VIT
    VIT --> RETURNS
    RETURNS --> DETACH
    RETURNS --> FUSE2 --> ACCUM
    LOOP --> AVG --> MULTI --> PROJECTOR --> OUT

    style POSENC fill:#e94560,stroke:#fff,color:#fff
    style BNFIX fill:#d4a373,stroke:#fff,color:#000
    style DETACH fill:#ff6b6b,stroke:#fff,color:#fff
    style LOOP fill:#16213e,stroke:#e94560,color:#fff
```

### 5c. Register State Flow Across Tiles

The register token accumulates global context as it passes through tiles in
raster-scan order (left-to-right, top-to-bottom).

```mermaid
flowchart LR
    R0["register_token<br/>(learned param)"]

    subgraph T0["Tile 0 (0,0)"]
        POS0["+ pos_proj(0.0, 0.0)"]
        VIT0["ViT 12 blocks"]
        OUT0["register_0"]
    end

    subgraph T1["Tile 1 (0,1)"]
        POS1["+ pos_proj(0.2, 0.0)"]
        VIT1["ViT 12 blocks"]
        OUT1["register_1"]
    end

    DOTS1["..."]

    subgraph T14["Tile 14 (2,4)"]
        POS14["+ pos_proj(0.8, 0.67)"]
        VIT14["ViT 12 blocks"]
        OUT14["register_14"]
    end

    R0 -->|clone| POS0 --> VIT0 --> OUT0
    OUT0 -->|".detach()"| POS1 --> VIT1 --> OUT1
    OUT1 -->|".detach()"| DOTS1
    DOTS1 -->|".detach()"| POS14 --> VIT14 --> OUT14

    style R0 fill:#533483,stroke:#fff,color:#fff
    style OUT0 fill:#2d6a4f,stroke:#fff,color:#fff
    style OUT1 fill:#2d6a4f,stroke:#fff,color:#fff
    style OUT14 fill:#2d6a4f,stroke:#fff,color:#fff
```

Key design decisions:
- **`.detach()`** between tiles: Truncated BPTT prevents gradient explosion
  through 15 sequential ViT passes (each 12 blocks deep).
- **`tile_pos_proj`**: Without this, RoPE normalizes every tile's coordinates
  to `[-1, +1]` independently — tile (0,0) and tile (2,4) look identical.
  The position projection adds 576 new parameters to break this ambiguity.

---

## 6. Feature Reassembly — Overlap Averaging

After the tile loop, overlapping tile features are averaged on the global
feature map using a count tensor.

```mermaid
flowchart TD
    subgraph TILES["Individual Tile Features (28×28 each)"]
        T0F["Tile 0 features"]
        T1F["Tile 1 features"]
        T2F["Tile 2 features"]
        TDots["..."]
    end

    subgraph GLOBAL["Global Feature Map [B, 192, 56, 84]"]
        GFEAT["global_feat<br/>(accumulated sums)"]
        GCOUNT["global_count<br/>(overlap counter)"]
    end

    DIV["global_feat / clamp(global_count, min=1)"]

    RESULT["Averaged Feature Map<br/>[B, 192, 56, 84]"]

    T0F --> GFEAT
    T1F --> GFEAT
    T2F --> GFEAT
    TDots --> GFEAT

    T0F --> GCOUNT
    T1F --> GCOUNT
    T2F --> GCOUNT
    TDots --> GCOUNT

    GFEAT --> DIV
    GCOUNT --> DIV
    DIV --> RESULT

    style TILES fill:#16213e,stroke:#0f3460,color:#fff
    style GLOBAL fill:#1a1a2e,stroke:#e94560,color:#fff
    style RESULT fill:#2d6a4f,stroke:#fff,color:#fff
```

```
Overlap count map for 896×1344 input (feature space 56×84):

 count=1 ┊ count=2 ┊ count=2 ┊ count=2 ┊ count=1
 ┈┈┈┈┈┈┈┈┊─────────┊─────────┊─────────┊┈┈┈┈┈┈┈┈
 count=2 ┊ count=4 ┊ count=4 ┊ count=4 ┊ count=2
 ┈┈┈┈┈┈┈┈┊─────────┊─────────┊─────────┊┈┈┈┈┈┈┈┈
 count=1 ┊ count=2 ┊ count=2 ┊ count=2 ┊ count=1

 Corners: 1×  |  Edges: 2×  |  Interior: 4×
```

---

## 7. VisionTransformer.forward — Modified Signature

The only change to VisionTransformer itself: an optional `register_state` input
and a second return value.

```mermaid
flowchart TD
    X["x [B, 3, 448, 448]"]
    RS["register_state<br/>[B, 1, 192] or None"]

    PATCH["ConvPyramidPatchEmbed<br/>3 conv stages + proj<br/>→ [B, 784, 192]"]

    CHECK{"register_state<br/>provided?"}
    USE_PARAM["Use learned<br/>self.register_token"]
    USE_STATE["Use provided<br/>register_state"]

    CAT["Concat: [register, patches]<br/>→ [B, 785, 192]"]

    ROPE["RoPE position encoding<br/>(tile-local coordinates)"]

    subgraph BLOCKS["12 Transformer Blocks"]
        B0["Block 0"]
        B1["Block 1"]
        BDOTS["..."]
        B10["Block 10 → return_layers[0]"]
        B11["Block 11 → return_layers[1]"]
    end

    REG_OUT["x[:, :1, :] → final_register<br/>[B, 1, 192]"]
    FEAT_OUT["x[:, 1:, :] → feature tokens<br/>[B, 784, 192]"]

    X --> PATCH
    RS --> CHECK
    CHECK -->|None| USE_PARAM --> CAT
    CHECK -->|Provided| USE_STATE --> CAT
    PATCH --> CAT
    CAT --> ROPE --> B0 --> B1 --> BDOTS --> B10 --> B11
    B11 --> REG_OUT
    B10 --> FEAT_OUT
    B11 --> FEAT_OUT

    style BLOCKS fill:#16213e,stroke:#0f3460,color:#fff
    style REG_OUT fill:#e94560,stroke:#fff,color:#fff
    style CHECK fill:#533483,stroke:#fff,color:#fff
```

---

## 8. PadToMultiple Transform

Ensures every input image is tile-compatible before entering the model.
Applied after `ConvertPILImage` (operates on tensors, not PIL images).

```mermaid
flowchart LR
    IN["Image tensor<br/>[3, H, W]"]

    CALC["Compute padded size:<br/>H_pad = tile_size + ⌈(H−t)/s⌉ × s<br/>W_pad = tile_size + ⌈(W−t)/s⌉ × s"]

    PAD["F.pad(img,<br/>[0, 0, pad_right, pad_bottom],<br/>fill=0)"]

    OUT["Padded tensor<br/>[3, H_pad, W_pad]"]

    IN --> CALC --> PAD --> OUT

    style CALC fill:#16213e,stroke:#0f3460,color:#fff
    style PAD fill:#e94560,stroke:#fff,color:#fff
```

```
Example: 756×1344 with tile_size=448, stride=224

  Original (756×1344)           Padded (896×1344)
  ┌──────────────────┐          ┌──────────────────┐
  │                  │          │                  │
  │   image content  │  ────►   │   image content  │
  │                  │          │                  │
  └──────────────────┘          │ ─ ─ ─ ─ ─ ─ ─ ─ │
                                │  padding (zeros)  │  +140px bottom
                                └──────────────────┘

  Check: (896 − 448) % 224 = 448 % 224 = 0  ✓
         (1344 − 448) % 224 = 896 % 224 = 0  ✓
         Tiles: 3 rows × 5 cols = 15 tiles   ✓
```

---

## 9. Transform Pipeline — Train vs Val

```mermaid
flowchart TD
    subgraph TRAIN["Training Pipeline"]
        direction TB
        T1["RandomPhotometricDistort (p=0.5)"]
        T2["RandomHorizontalFlip"]
        T3["SanitizeBoundingBoxes"]
        T4["Resize [896, 1344]"]
        T5["SanitizeBoundingBoxes"]
        T6["ConvertPILImage (float32, /255)"]
        T7["PadToMultiple (448, 224) ★NEW"]
        T8["Normalize (ImageNet)"]
        T9["ConvertBoxes (cxcywh, normalize)"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8 --> T9
    end

    subgraph VAL["Validation Pipeline"]
        direction TB
        V1["Resize [896, 1344]"]
        V2["ConvertPILImage (float32, /255)"]
        V3["PadToMultiple (448, 224) ★NEW"]
        V4["Normalize (ImageNet)"]
        V1 --> V2 --> V3 --> V4
    end

    subgraph REMOVED["Removed Transforms"]
        direction TB
        R1["❌ Mosaic — incompatible with variable sizes"]
        R2["❌ RandomZoomOut — incompatible"]
        R3["❌ RandomIoUCrop — incompatible"]
        R4["❌ Mixup (collate) — disabled"]
    end

    style TRAIN fill:#1a1a2e,stroke:#2d6a4f,color:#fff
    style VAL fill:#1a1a2e,stroke:#0f3460,color:#fff
    style REMOVED fill:#1a1a2e,stroke:#e94560,color:#fff
    style T7 fill:#e94560,stroke:#fff,color:#fff
    style V3 fill:#e94560,stroke:#fff,color:#fff
```

---

## 10. Dataset Pipeline — VisDrone MOT to COCO

```mermaid
flowchart LR
    subgraph SRC["VisDrone MOT Source"]
        SEQ["sequences/<br/>├── uav0000013_00000_v/<br/>│   ├── 0000001.jpg<br/>│   ├── 0000002.jpg<br/>│   └── ..."]
        ANN["annotations/<br/>├── uav0000013_00000_v.txt<br/>frame,id,x,y,w,h,score,cat,trunc,occ"]
    end

    CONVERT["visdrone_mot_to_coco.py<br/>• Flatten sequences to flat images<br/>• Remap categories 1-10 → 0-9<br/>• Skip cat 0 (ignored) & 11 (others)<br/>• Output COCO JSON format"]

    subgraph DST["COCO Format Output"]
        IMGS["images/<br/>├── uav0000013_00000_v_0000001.jpg → (symlink)<br/>└── ..."]
        JSON["annotations/<br/>├── train.json (24,201 images, 1.1M anns)<br/>└── val.json (2,846 images, 114K anns)"]
    end

    SEQ --> CONVERT
    ANN --> CONVERT
    CONVERT --> IMGS
    CONVERT --> JSON

    style CONVERT fill:#e94560,stroke:#fff,color:#fff
    style SRC fill:#16213e,stroke:#0f3460,color:#fff
    style DST fill:#2d6a4f,stroke:#fff,color:#fff
```

**Category mapping** (0-indexed to match `num_classes=10`):

| ID | Name | ID | Name |
|:--:|------|:--:|------|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

---

## 11. Optimizer Parameter Groups

Five groups with different learning rates — backbone frozen in Phase 1.

```mermaid
flowchart TD
    subgraph MODEL["ECDet Parameters (484 trainable)"]
        subgraph BB["backbone.backbone.* (VisionTransformer)"]
            BBW["Weights (53 params)<br/>lr = 0.0 (frozen)"]
            BBN["Norm/BN/Bias (103 params)<br/>lr = 0.0, wd = 0.0"]
        end

        subgraph TILE["backbone.tile_pos_proj.* [NEW]"]
            TLP["Weight + Bias (576 params)<br/>lr = 0.0002"]
        end

        subgraph ENCDEC["Encoder + Decoder + Projector"]
            EDN["Norm/BN/Bias (204 params)<br/>lr = 0.0002, wd = 0.0"]
            EDW["Weights (122 params, catch-all)<br/>lr = 0.0002, wd = 0.0001"]
        end
    end

    style BB fill:#e94560,stroke:#fff,color:#fff
    style TILE fill:#2d6a4f,stroke:#fff,color:#fff
    style ENCDEC fill:#16213e,stroke:#0f3460,color:#fff
```

---

## 12. Critical Fixes Summary

Three critical issues were identified and fixed during implementation.

```mermaid
flowchart TD
    subgraph C1["CRITICAL-1: Variable Image Sizes"]
        C1P["Problem: torch.cat in collate<br/>crashes on different sizes"]
        C1F["Fix: Resize to fixed size<br/>+ PadToMultiple transform"]
    end

    subgraph C2["CRITICAL-2: BatchNorm Corruption"]
        C2P["Problem: BN in PatchEmbed<br/>updates running stats per tile<br/>(B=2 per tile pass)"]
        C2F["Fix: Force BN → eval()<br/>during tile loop,<br/>restore in finally block"]
    end

    subgraph C3["CRITICAL-3: Tile-Blind RoPE"]
        C3P["Problem: RoPE normalizes coords<br/>per tile → all tiles look identical<br/>at position (0,0)"]
        C3F["Fix: tile_pos_proj Linear(2→192)<br/>injects (col/nw, row/nh)<br/>into register state"]
    end

    subgraph C4["CRITICAL-4: reset_cfg Override"]
        C4P["Problem: reset_cfg() overrides<br/>Resize to square (640,640)<br/>+ cached pos_embed size mismatch"]
        C4F["Fix: Set eval_spatial_size: ~<br/>forces dynamic pos_embed<br/>and anchor generation"]
    end

    C1P --> C1F
    C2P --> C2F
    C3P --> C3F
    C4P --> C4F

    style C1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style C2 fill:#1a1a2e,stroke:#d4a373,color:#fff
    style C3 fill:#1a1a2e,stroke:#533483,color:#fff
    style C4 fill:#1a1a2e,stroke:#2d6a4f,color:#fff
    style C1F fill:#2d6a4f,stroke:#fff,color:#fff
    style C2F fill:#2d6a4f,stroke:#fff,color:#fff
    style C3F fill:#2d6a4f,stroke:#fff,color:#fff
    style C4F fill:#2d6a4f,stroke:#fff,color:#fff
```

---

## 13. Modified Files Map

```mermaid
flowchart LR
    subgraph MODIFIED["Modified Files"]
        ECVIT["engine/edgecrafter/ecvit.py<br/>• TileExtractor class [NEW]<br/>• VisionTransformer.forward signature [MOD]<br/>• ViTAdapter.forward dispatcher [MOD]<br/>• ViTAdapter._forward_single [NEW]<br/>• ViTAdapter._forward_tiled [NEW]<br/>• ViTAdapter.tile_pos_proj [NEW]"]

        MODELING["engine/edgecrafter/modeling.py<br/>• ECDet.__init__ accepts<br/>  tile_size, tile_stride [MOD]"]

        TRANSFORMS["engine/data/transforms/_transforms.py<br/>• PadToMultiple class [NEW]"]
    end

    subgraph CONFIG["New Config Files"]
        TILED_YML["configs/ecdet/<br/>ecdet_s_visdrone_tiled.yml [NEW]"]
        VD_YML["configs/dataset/<br/>visdrone_detection.yml [NEW]"]
    end

    subgraph TOOLS["New Tools"]
        CONVERTER["tools/visdrone_mot_to_coco.py [NEW]"]
        TESTS["tools/test_tiled_forward.py [NEW]<br/>112 validation tests"]
    end

    subgraph UNTOUCHED["Untouched Components"]
        HE["HybridEncoder"]
        DEC["ECTransformer / Decoder"]
        PP["PostProcessor"]
        DL["DataLoader / Collate"]
        ROPE["RopePositionEmbedding"]
        CPE["ConvPyramidPatchEmbed"]
    end

    style MODIFIED fill:#e94560,stroke:#fff,color:#fff
    style CONFIG fill:#2d6a4f,stroke:#fff,color:#fff
    style TOOLS fill:#533483,stroke:#fff,color:#fff
    style UNTOUCHED fill:#16213e,stroke:#0f3460,color:#fff
```

---

## 14. Size Reference Table

| Input | After Resize | After Pad | Tiles | Feature Map (s8) | Feature Map (s32) |
|-------|-------------|-----------|-------|-------------------|-------------------|
| 756×1344 | 896×1344 | 896×1344 | 3×5=15 | 112×168 | 28×42 |
| 1080×1920 | 896×1344 | 896×1344 | 3×5=15 | 112×168 | 28×42 |
| 640×640 | — | — | single pass | 80×80 | 20×20 |
| 448×448 | — | 448×448 | 1×1=1 | 56×56 | 14×14 |

**Model stats** (profiled at 640×640 single pass):
- Parameters: **9.77M** (576 new from `tile_pos_proj`)
- FLOPs: **25.8 GFLOPS** per single-tile pass
- Estimated FLOPs per 896×1344 frame: **~387 GFLOPS** (15 tiles × 25.8)
