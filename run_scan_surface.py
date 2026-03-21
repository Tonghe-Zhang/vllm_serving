#!/usr/bin/env python3
"""Standalone scan_surface pipeline — runs on a local image file.

1. Load image, crop to left third
2. SAM3 surface segmentation via serve_sam3 HTTP (/segment endpoint)
3. SAM2 auto-segmentation on cropped surface region
4. Qwen3-VL identification of each segment
5. Filter by overlap with surface mask
6. Save all intermediate images

Usage:
    python run_scan_surface.py
"""

import base64
import io
import os
import sys
import time

import cv2
import numpy as np
import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/usr0/tonghez/vllm_serving/media/00_input_cropped (1).jpg"
SURFACE_QUERY = "table"  # the surface to scan
SAM3_URL = "http://localhost:6767"
QWEN_URL = "http://localhost:8000/v1"
QWEN_MODEL = "Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "/usr0/tonghez/vllm_serving/media/scan_output"
MIN_AREA = 100
MAX_SEGMENTS = 40
OVERLAP_THRESH = 0.1
SCORE_THRESH = 0.05

# Palette for mask overlays
PALETTE = [
    (108, 92, 231), (0, 184, 148), (253, 121, 168), (9, 132, 227),
    (255, 234, 167), (214, 48, 49), (0, 206, 209), (255, 159, 67),
    (162, 155, 254), (85, 239, 196), (129, 236, 236), (250, 177, 160),
]
SURFACE_COLOR = (0, 200, 255)
BG_LABELS = frozenset({"none", "unknown", "background", ""})


def encode_image_b64_jpeg(img_rgb: np.ndarray) -> str:
    """RGB numpy → base64 JPEG string."""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_rgb(path: str, rgb: np.ndarray):
    """Save RGB numpy array as image file."""
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  → saved: {path}")


def segment_surface_sam3(rgb: np.ndarray, text: str):
    """Call SAM3 server /segment to get surface mask."""
    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    resp = requests.post(
        f"{SAM3_URL}/segment",
        json={"text": text, "image_b64": image_b64},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    mask_bytes = base64.b64decode(data["mask_b64"])
    mask = np.load(io.BytesIO(mask_bytes))
    return mask, data["bbox_xywh"], data["score"]


def segment_everything_sam2(rgb: np.ndarray, min_area=500, max_segments=20):
    """Run SAM2 auto-segmentation using transformers pipeline directly."""
    from transformers import pipeline as hf_pipeline

    print("    Loading SAM2 mask-generation pipeline...")
    pipe = hf_pipeline(
        "mask-generation",
        model="facebook/sam2.1-hiera-large",
        device="cuda",
    )
    print("    SAM2 ready.")

    pil_img = Image.fromarray(rgb)
    raw = pipe(pil_img, points_per_side=64,
               pred_iou_thresh=0.5,
               stability_score_thresh=0.7,
               crop_n_layers=1,
               crops_nms_thresh=0.3)

    masks = raw.get("masks", [])
    scores = raw.get("scores", [])

    segments = []
    for m, s in zip(masks, scores):
        m_np = np.asarray(m, dtype=np.uint8)
        area = int(m_np.sum())
        if area < min_area:
            continue
        ys, xs = np.where(m_np > 0)
        bbox_xywh = (int(xs.min()), int(ys.min()),
                      int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        segments.append({
            "mask": m_np,
            "bbox_xywh": bbox_xywh,
            "area": area,
            "score": float(s),
        })

    segments.sort(key=lambda d: d["score"], reverse=True)

    # Free pipeline
    del pipe
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return segments[:max_segments]


def _query_qwen(b64: str, prompt: str) -> str:
    """Send a single image+prompt to Qwen and return the raw response."""
    import openai
    client = openai.OpenAI(base_url=QWEN_URL, api_key="EMPTY", timeout=30.0)
    resp = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=30,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


def identify_object_vlm(rgb: np.ndarray, bbox_xywh, mask=None):
    """Crop RGB to bbox, optionally mask, query Qwen3-VL, return (label, crop, raw_response)."""
    x, y, w, h = bbox_xywh
    H, W = rgb.shape[:2]
    pad_x, pad_y = int(w * 0.2), int(h * 0.2)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)

    if mask is not None:
        isolated = rgb.copy()
        isolated[mask == 0] = 128
        crop = isolated[y0:y1, x0:x1]
    else:
        crop = rgb[y0:y1, x0:x1]

    b64 = encode_image_b64_jpeg(crop)

    # Pass 1: try to identify the object
    prompt1 = (
        "What single object is shown in the center of this image? "
        "Reply with ONLY the object name (2-4 words, e.g. 'red coffee mug'). "
        "If no distinct object or just background, say 'none'."
    )
    raw1 = _query_qwen(b64, prompt1)
    label = raw1.strip().lower()[:50]

    # Pass 2: if pass 1 said "none", ask if it's still a physical object
    raw_response = raw1
    if label in BG_LABELS:
        prompt2 = (
            "Ignore what this object is. Just answer: is there a distinct "
            "physical object in the center of this image, or is it just "
            "background / surface / wall / table / structural material? "
            "Reply ONLY 'object' or 'none'."
        )
        raw2 = _query_qwen(b64, prompt2)
        raw_response = f"pass1: {raw1} | pass2: {raw2}"
        if "object" in raw2.lower() and "none" not in raw2.lower():
            label = "unknown object"

    return label, crop, raw_response, ""


def draw_mask_overlay(rgb, mask, color, alpha=0.35):
    """Draw a single mask overlay."""
    out = rgb.copy()
    colored = np.zeros_like(out)
    colored[mask > 0] = color
    cv2.addWeighted(colored, alpha, out, 1 - alpha, 0, out)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_raw_sam2_overlay(rgb, segs):
    """Draw all raw SAM2 masks with IDs."""
    out = rgb.copy()
    for i, seg in enumerate(segs):
        mask = seg["mask"]
        color = PALETTE[i % len(PALETTE)]
        colored = np.zeros_like(out)
        colored[mask > 0] = color
        cv2.addWeighted(colored, 0.35, out, 0.65, 0, out)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        x, y, w, h = seg["bbox_xywh"]
        label = f"#{i} (s={seg['score']:.2f}, a={seg['area']})"
        cv2.putText(out, label, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(out, label, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.perf_counter()

    # ── Step 0: Load & crop ──────────────────────────────────────────────
    print("=" * 70)
    print("SCAN SURFACE PIPELINE")
    print("=" * 70)
    print(f"\n[0] Loading image: {IMAGE_PATH}")
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print(f"ERROR: Cannot read {IMAGE_PATH}")
        return
    rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb_full.shape[:2]
    print(f"    Full image: {W}x{H}")

    # Crop to left third if multi-panel, otherwise use full image
    if W > H * 3:  # multi-panel (e.g. 1375x285 = 3 views side by side)
        w_third = W // 3
        rgb = rgb_full[:, :w_third]
        print(f"    Multi-panel detected, cropping to left third")
    else:
        rgb = rgb_full
    H, W = rgb.shape[:2]
    print(f"    Cropped to left third: {W}x{H}")
    save_rgb(os.path.join(OUTPUT_DIR, "00_input_cropped.jpg"), rgb)

    # ── Step 1: SAM3 surface segmentation ────────────────────────────────
    print(f"\n[1] SAM3 surface segmentation: query='{SURFACE_QUERY}'")
    t1 = time.perf_counter()
    cached_mask_path = os.path.join(OUTPUT_DIR, "cached_surface_mask.npy")
    try:
        surface_mask, surface_bbox, surface_score = segment_surface_sam3(
            rgb, SURFACE_QUERY
        )
        # Cache for reuse if SAM3 goes down
        np.save(cached_mask_path, surface_mask)
    except Exception as e:
        if os.path.exists(cached_mask_path):
            print(f"    SAM3 unavailable, using cached mask")
            surface_mask = np.load(cached_mask_path)
            ys, xs = np.where(surface_mask > 0)
            surface_bbox = [int(xs.min()), int(ys.min()),
                            int(xs.max() - xs.min()), int(ys.max() - ys.min())]
            surface_score = 0.8
        else:
            print(f"    ERROR: SAM3 failed and no cached mask: {e}")
            return
    print(f"    Score: {surface_score:.4f}")
    print(f"    Bbox: {surface_bbox}")
    print(f"    Mask area: {int(surface_mask.sum())} px")
    print(f"    Time: {time.perf_counter() - t1:.2f}s")

    if surface_score < SCORE_THRESH:
        print(f"    WARN: Score below threshold {SCORE_THRESH}")

    # Save surface segmentation overlay
    surface_overlay = draw_mask_overlay(rgb, surface_mask, SURFACE_COLOR, alpha=0.4)
    sx, sy, sw, sh = surface_bbox
    cv2.rectangle(surface_overlay, (sx, sy), (sx + sw, sy + sh), SURFACE_COLOR, 2)
    cv2.putText(surface_overlay, f"{SURFACE_QUERY} (s={surface_score:.3f})",
                (sx, max(sy - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SURFACE_COLOR, 2)
    save_rgb(os.path.join(OUTPUT_DIR, "01_surface_segmentation.jpg"), surface_overlay)

    # Save pure mask
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[surface_mask > 0] = (255, 255, 255)
    save_rgb(os.path.join(OUTPUT_DIR, "01b_surface_mask_pure.jpg"), mask_vis)

    # Fill holes + convex hull — SAM3 excludes objects on the table and
    # may cut off at the front edge. Convex hull covers the full table area.
    contours_fill, _ = cv2.findContours(
        surface_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Merge all contour points and take convex hull
    all_pts = np.concatenate(contours_fill)
    hull = cv2.convexHull(all_pts)
    surface_mask_filled = np.zeros_like(surface_mask)
    cv2.drawContours(surface_mask_filled, [hull], -1, 1, thickness=cv2.FILLED)
    print(f"    Filled mask area: {int(surface_mask_filled.sum())} px (was {int(surface_mask.sum())})")
    mask_filled_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_filled_vis[surface_mask_filled > 0] = (255, 255, 255)
    save_rgb(os.path.join(OUTPUT_DIR, "01c_surface_mask_filled.jpg"), mask_filled_vis)

    # ── Step 2: Crop to surface bbox ─────────────────────────────────────
    pad_x, pad_y = int(sw * 0.05), int(sh * 0.05)
    cx0, cy0 = max(0, sx - pad_x), max(0, sy - pad_y)
    cx1, cy1 = min(W, sx + sw + pad_x), min(H, sy + sh + pad_y)
    cropped_rgb = rgb[cy0:cy1, cx0:cx1]
    surface_mask_cropped = surface_mask_filled[cy0:cy1, cx0:cx1]
    print(f"\n[2] Cropped to surface region: [{cx0}:{cx1}, {cy0}:{cy1}] = {cx1-cx0}x{cy1-cy0}px")
    save_rgb(os.path.join(OUTPUT_DIR, "02_surface_crop.jpg"), cropped_rgb)

    # ── Step 3: SAM2 auto-segmentation on cropped region ─────────────────
    print(f"\n[3] SAM2 auto-segmentation (min_area={MIN_AREA}, max_segments={MAX_SEGMENTS})")
    t3 = time.perf_counter()
    segs = segment_everything_sam2(cropped_rgb, min_area=MIN_AREA, max_segments=MAX_SEGMENTS)
    print(f"    {len(segs)} raw segments found  ({time.perf_counter() - t3:.2f}s)")
    for i, s in enumerate(segs):
        print(f"    seg#{i}: area={s['area']}  score={s['score']:.3f}  bbox={s['bbox_xywh']}")

    # Save raw SAM2 overlay — white bg with colored contours
    H_c, W_c = cropped_rgb.shape[:2]
    raw_overlay = np.full((H_c, W_c, 3), 255, dtype=np.uint8)
    for i, seg in enumerate(segs):
        color = PALETTE[i % len(PALETTE)]
        mask = seg["mask"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(raw_overlay, contours, -1, color, 2)
        x, y, w, h = seg["bbox_xywh"]
        cv2.putText(raw_overlay, f"#{i}", (x+2, y+h//2+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    save_rgb(os.path.join(OUTPUT_DIR, "03_sam2_raw_masks.jpg"), raw_overlay)

    if not segs:
        print("\n    No segments found. Done.")
        return

    # ── Step 4: VLM identification ───────────────────────────────────────
    print(f"\n[4] Qwen3-VL identification ({len(segs)} segments)")
    t4 = time.perf_counter()
    vlm_results = []
    for i, seg in enumerate(segs):
        print(f"\n    --- Segment #{i} ---")
        label, crop_img, raw_resp, thinking = identify_object_vlm(
            cropped_rgb, seg["bbox_xywh"], mask=seg["mask"]
        )
        print(f"    Label: {label}")
        if thinking:
            print(f"    Thinking: {thinking[:200]}...")
        print(f"    Raw response: {raw_resp[:200]}")

        # Save VLM input crop
        crop_path = os.path.join(OUTPUT_DIR, f"04_vlm_input_seg{i}.jpg")
        save_rgb(crop_path, crop_img)

        vlm_results.append({
            "seg_idx": i,
            "seg": seg,
            "label": label,
            "crop": crop_img,
            "raw_response": raw_resp,
            "thinking": thinking,
        })

    print(f"\n    VLM total: {time.perf_counter() - t4:.2f}s")

    # ── Step 5: Filter — bg labels, surface itself, and overlap ─────────
    print(f"\n[5] Filtering")
    # Use original (unfilled) surface mask area to detect "IS the surface"
    original_surface_area_cropped = int(surface_mask[cy0:cy1, cx0:cx1].sum())
    kept_objects = []

    for vr in vlm_results:
        seg = vr["seg"]
        label = vr["label"]
        seg_mask = seg["mask"]
        mask_px = int(seg_mask.sum())

        if label in BG_LABELS:
            print(f"    '{label}' (seg#{vr['seg_idx']}): DROP (background)")
            continue

        if mask_px == 0:
            print(f"    '{label}' (seg#{vr['seg_idx']}): DROP (empty mask)")
            continue

        # Skip if the segment IS the surface itself — compare against
        # the original SAM3 mask area (not the convex hull)
        if original_surface_area_cropped > 0 and mask_px > original_surface_area_cropped * 0.5:
            print(f"    '{label}' (seg#{vr['seg_idx']}): DROP (IS the surface, "
                  f"{mask_px}/{original_surface_area_cropped}px)")
            continue

        # Unknown objects from pass2 need a higher area bar — small unknowns
        # are usually texture artifacts or sub-parts of other objects
        if label == "unknown object" and mask_px < 400:
            print(f"    '{label}' (seg#{vr['seg_idx']}): DROP (unknown too small, {mask_px}px)")
            continue

        # Check overlap with convex-hull surface mask — object must have
        # >=25% of its area on the surface to count as "on it"
        overlap = int((seg_mask & surface_mask_cropped).sum())
        overlap_frac = overlap / mask_px if mask_px > 0 else 0
        if overlap_frac < 0.25:
            print(f"    '{label}' (seg#{vr['seg_idx']}): DROP (overlap={overlap_frac:.0%}, not on surface)")
            continue

        print(f"    '{label}' (seg#{vr['seg_idx']}): KEEP (area={mask_px}px, overlap={overlap_frac:.0%})")

        # Remap to full image coords
        ox, oy, ow, oh = seg["bbox_xywh"]
        full_bbox = [ox + cx0, oy + cy0, ow, oh]
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[cy0:cy1, cx0:cx1] = seg_mask
        kept_objects.append({
            "label": label,
            "bbox_xywh": full_bbox,
            "mask": full_mask,
            "score": seg["score"],
            "area": mask_px,
            "raw_response": vr["raw_response"],
        })

    print(f"\n    Kept: {len(kept_objects)}/{len(vlm_results)} objects")

    # ── Step 6: Final visualization ──────────────────────────────────────
    print(f"\n[6] Final visualization")

    # White background with colored contours + labels (same style as raw SAM2)
    out = np.full((H, W, 3), 255, dtype=np.uint8)

    # Surface contour (amber)
    s_contours, _ = cv2.findContours(
        surface_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, s_contours, -1, SURFACE_COLOR, 2)
    cv2.putText(out, SURFACE_QUERY, (sx, max(sy - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SURFACE_COLOR, 1)

    # Object contours + labels
    for i, obj in enumerate(kept_objects):
        mask = obj["mask"]
        color = PALETTE[i % len(PALETTE)]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        x, y, w, h = obj["bbox_xywh"]
        label_text = f"#{i} {obj['label']}"
        cv2.putText(out, label_text, (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    save_rgb(os.path.join(OUTPUT_DIR, "06_final_result.jpg"), out)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Surface: '{SURFACE_QUERY}' (score={surface_score:.3f})")
    print(f"Objects found: {len(kept_objects)}")
    for i, obj in enumerate(kept_objects):
        print(f"  [{i}] {obj['label']}: bbox={obj['bbox_xywh']}, score={obj['score']:.3f}, "
              f"area={obj['area']}px")

    print(f"\nTotal time: {time.perf_counter() - t_total:.2f}s")
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
