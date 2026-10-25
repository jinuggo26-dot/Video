from __future__ import annotations
import asyncio
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

app = FastAPI(title='Killlog auto extractor')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
    allow_credentials=False
)

ROOT = Path(__file__).resolve().parent
GATE = asyncio.Semaphore(2)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 6_000_000
OUT_W, OUT_H = 760, 190


def transparent_extract(raw: bytes) -> bytes:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('JPG·PNG·WEBP 이미지를 읽을 수 없습니다.')
    H, W = image.shape[:2]
    if W < 350 or H < 200 or W * H > MAX_PIXELS:
        raise ValueError('이미지 크기가 너무 작거나 큽니다(최대 600만 픽셀).')

    ax, bx = round(W * 0.285), round(W * 0.715)
    ay, by = round(H * 0.11), round(H * 0.305)
    roi = image[ay:by, ax:bx].copy()
    h, w = roi.shape[:2]

    above_y = max(0, round(H * 0.102))
    below_y = min(H - 1, round(H * 0.318))
    above = cv2.GaussianBlur(image[above_y:above_y + 1, ax:bx].copy(), (25, 1), 0)[0]
    below = cv2.GaussianBlur(image[below_y:below_y + 1, ax:bx].copy(), (25, 1), 0)[0]

    ys = np.arange(ay, by, dtype=np.float32)
    weight = np.clip((ys - above_y) / max(1, below_y - above_y), 0, 1)[:, None, None]
    background = above.astype(np.float32)[None, :, :] * (1 - weight) + below.astype(np.float32)[None, :, :] * weight
    background = cv2.GaussianBlur(background, (7, 3), 0)

    lab_fore = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_back = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab_fore - lab_back) ** 2, axis=2))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    strength = np.clip((diff - 17.0) * 6.0, 0, 255)
    saturated = np.clip((sat - 75) * 2.0, 0, 120) * np.clip((diff - 9) / 25, 0, 1)
    bright = np.clip((val - 145) * 2.4, 0, 180) * np.clip((diff - 12) / 30, 0, 1)
    strength = np.maximum.reduce([strength, saturated, bright])

    hard = np.where(strength > 45, 255, 0).astype(np.uint8)
    joined = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    plausible = []
    for i in range(1, num):
        x, y, ww, hh, area = map(int, stats[i])
        if ww < max(95, round(w * .18)) or hh < 10 or area < 250:
            continue
        cy = y + hh / 2
        score = ww * 4 + area / 4 - abs(cy - h * .42) * 3
        plausible.append((score, i))

    chosen = np.zeros_like(hard)
    if plausible:
        _, idx = max(plausible)
        chosen[lab == idx] = 255
        dilated = cv2.dilate(chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        chosen = np.where((dilated > 0) & (strength > 17), 255, chosen).astype(np.uint8)
        hard_local = np.where((dilated > 0) & (strength > 18), 255, 0).astype(np.uint8)
        chosen = cv2.bitwise_or(chosen, hard_local)
    else:
        chosen = hard

    yy, xx = np.where(chosen > 0)
    if len(xx) >= 30:
        xl, xr = max(0, int(np.percentile(xx, 1)) - 18), min(w, int(np.percentile(xx, 99)) + 19)
        yt, yb = max(0, int(np.percentile(yy, 1)) - 15), min(h, int(np.percentile(yy, 99)) + 16)
        region = np.zeros_like(chosen)
        region[yt:yb, xl:xr] = 255
        alpha = (np.maximum(chosen, np.clip((strength - 12) * 3.0, 0, 255).astype(np.uint8)) & region)
    else:
        alpha = np.where(strength > 32, np.clip(strength, 0, 255), 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(alpha, (3, 3), .6)
    yy, xx = np.where(alpha > 18)
    if len(xx) < 15:
        raise ValueError('킬로그 후보를 찾지 못했습니다. 킬 알림이 보이는 원본 화면을 올려 주세요.')

    x0, x1 = max(0, int(xx.min()) - 5), min(w, int(xx.max()) + 6)
    y0, y1 = max(0, int(yy.min()) - 5), min(h, int(yy.max()) + 6)

    color = roi[y0:y1, x0:x1]
    matte = alpha[y0:y1, x0:x1]

    cw, ch = x1 - x0, y1 - y0
    scale = min((OUT_W - 16) / max(1, cw), (OUT_H - 16) / max(1, ch))
    dw, dh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    color = cv2.resize(color, (dw, dh), interpolation=interp)
    matte = cv2.resize(matte, (dw, dh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
    xpad, ypad = (OUT_W - dw) // 2, (OUT_H - dh) // 2
    canvas[ypad:ypad + dh, xpad:xpad + dw, :3] = color
    canvas[ypad:ypad + dh, xpad:xpad + dw, 3] = matte

    success, png = cv2.imencode('.png', canvas)
    if not success:
        raise ValueError('PNG 인코딩에 실패했습니다.')
    return png.tobytes()


def transparent_extract_detail(raw: bytes) -> bytes:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('JPG·PNG·WEBP 이미지를 읽을 수 없습니다.')
    H, W = image.shape[:2]
    if W < 350 or H < 200 or W * H > MAX_PIXELS:
        raise ValueError('이미지 크기가 너무 작거나 큽니다(최대 600만 픽셀).')

    ax, bx = round(W * 0.265), round(W * 0.765)
    ay, by = round(H * 0.098), round(H * 0.312)
    roi = image[ay:by, ax:bx].copy()
    h, w = roi.shape[:2]

    above_y = max(0, round(H * 0.090))
    below_y = min(H - 1, round(H * 0.327))
    above = cv2.GaussianBlur(image[above_y:above_y + 1, ax:bx].copy(), (25, 1), 0)[0]
    below = cv2.GaussianBlur(image[below_y:below_y + 1, ax:bx].copy(), (25, 1), 0)[0]

    ys = np.arange(ay, by, dtype=np.float32)
    weight = np.clip((ys - above_y) / max(1, below_y - above_y), 0, 1)[:, None, None]
    background = above.astype(np.float32)[None, :, :] * (1 - weight) + below.astype(np.float32)[None, :, :] * weight
    background = cv2.GaussianBlur(background, (7, 3), 0)

    lab_fore = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_back = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab_fore - lab_back) ** 2, axis=2))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    strength = np.clip((diff - 14.5) * 6.5, 0, 255)
    saturated = np.clip((sat - 75) * 2.0, 0, 120) * np.clip((diff - 9) / 25, 0, 1)
    bright = np.clip((val - 145) * 2.4, 0, 180) * np.clip((diff - 12) / 30, 0, 1)
    strength = np.maximum.reduce([strength, saturated, bright])

    hard = np.where(strength > 43, 255, 0).astype(np.uint8)
    joined = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    plausible = []
    for i in range(1, num):
        x, y, ww, hh, area = map(int, stats[i])
        if ww < max(95, round(w * .18)) or hh < 10 or area < 250:
            continue
        cy = y + hh / 2
        score = ww * 4 + area / 4 - abs(cy - h * .42) * 3
        plausible.append((score, i))

    chosen = np.zeros_like(hard)
    if plausible:
        _, idx = max(plausible)
        chosen[lab == idx] = 255
        dilated = cv2.dilate(chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 11)))
        chosen = np.where((dilated > 0) & (strength > 14), 255, chosen).astype(np.uint8)
        hard_local = np.where((dilated > 0) & (strength > 15), 255, 0).astype(np.uint8)
        chosen = cv2.bitwise_or(chosen, hard_local)
    else:
        chosen = hard

    yy, xx = np.where(chosen > 0)
    if len(xx) >= 30:
        xl, xr = max(0, int(np.percentile(xx, 0.25)) - 40), min(w, int(np.percentile(xx, 99.75)) + 70)
        yt, yb = max(0, int(np.percentile(yy, 1)) - 18), min(h, int(np.percentile(yy, 99)) + 19)
        region = np.zeros_like(chosen)
        region[yt:yb, xl:xr] = 255
        alpha = (np.maximum(chosen, np.clip((strength - 10) * 3.2, 0, 255).astype(np.uint8)) & region)
    else:
        alpha = np.where(strength > 32, np.clip(strength, 0, 255), 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(alpha, (3, 3), .6)
    yy, xx = np.where(alpha > 18)
    if len(xx) < 15:
        raise ValueError('킬로그 후보를 찾지 못했습니다. 킬 알림이 보이는 원본 화면을 올려 주세요.')

    x0, x1 = max(0, int(xx.min()) - 5), min(w, int(xx.max()) + 6)
    y0, y1 = max(0, int(yy.min()) - 5), min(h, int(yy.max()) + 6)

    color = roi[y0:y1, x0:x1]
    matte = alpha[y0:y1, x0:x1]

    cw, ch = x1 - x0, y1 - y0
    scale = min((OUT_W - 16) / max(1, cw), (OUT_H - 16) / max(1, ch))
    dw, dh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    color = cv2.resize(color, (dw, dh), interpolation=interp)
    matte = cv2.resize(matte, (dw, dh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
    xpad, ypad = (OUT_W - dw) // 2, (OUT_H - dh) // 2
    canvas[ypad:ypad + dh, xpad:xpad + dw, :3] = color
    canvas[ypad:ypad + dh, xpad:xpad + dw, 3] = matte

    success, png = cv2.imencode('.png', canvas)
    if not success:
        raise ValueError('PNG 인코딩에 실패했습니다.')
    return png.tobytes()


def clean_background_png(png_data: bytes, strength: str = 'normal') -> bytes:
    pic = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pic is None or pic.ndim != 3 or pic.shape[2] != 4:
        return png_data

    alpha = pic[:, :, 3].copy()
    bgr = pic[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    occupancy = (alpha > 155).sum(axis=1)
    high_rows = np.where(occupancy >= max(100, int(0.29 * pic.shape[1])))[0]
    if len(high_rows) >= 8:
        top = max(0, int(np.percentile(high_rows, 8)) - 5)
        bottom = min(pic.shape[0], int(np.percentile(high_rows, 92)) + 7)
    else:
        top = int(pic.shape[0] * 0.36)
        bottom = int(pic.shape[0] * 0.68)

    yy = np.arange(pic.shape[0])[:, None]
    outside = (yy < top) | (yy >= bottom)

    blue_gray = (
        (h >= 90) & (h <= 130) &
        (s <= (135 if strength == 'normal' else 160)) &
        (v <= (130 if strength == 'normal' else 165))
    )
    gray_dark = (s <= (25 if strength == 'normal' else 40)) & (v <= (90 if strength == 'normal' else 115))

    candidates = (outside & (blue_gray | gray_dark)) | (blue_gray & (v <= 88))
    alpha[candidates] = (alpha[candidates].astype(np.float32) * (0.06 if strength == 'normal' else 0.01)).astype(np.uint8)

    strong_mask = (alpha > 22).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong_mask, 8)
    for i in range(1, count):
        x, y, w, hh, area = map(int, stats[i])
        if area < (10 if strength == 'normal' else 24):
            region = labels[y:y + hh, x:x + w] == i
            sub_s = s[y:y + hh, x:x + w]
            sub_v = v[y:y + hh, x:x + w]
            if region.any() and not np.any((sub_s[region] > 120) & (sub_v[region] > 115)):
                dest = alpha[y:y + hh, x:x + w]
                dest[region] = 0

    pic[:, :, 3] = cv2.GaussianBlur(alpha, (3, 3), 0.45)
    ok, encoded = cv2.imencode('.png', pic)
    return encoded.tobytes() if ok else png_data


def merge_base_detail_png(base_png: bytes, detail_png: bytes) -> bytes:
    base = cv2.imdecode(np.frombuffer(base_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    detail = cv2.imdecode(np.frombuffer(detail_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if base is None or detail is None:
        return base_png
    if base.ndim != 3 or detail.ndim != 3 or base.shape[2] != 4 or detail.shape[2] != 4:
        return base_png
    if base.shape != detail.shape:
        return base_png

    H, W = base.shape[:2]
    base_rgb = base[:, :, :3].copy()
    base_a = base[:, :, 3].copy()
    detail_rgb = detail[:, :, :3]
    detail_a = detail[:, :, 3].copy()

    yy, xx = np.where(base_a > 18)
    if len(xx) < 15:
        return base_png

    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    cy = (y0 + y1) / 2.0

    hsv = cv2.cvtColor(detail_rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    region = np.zeros((H, W), dtype=np.uint8)
    rx0 = max(0, x0 - 100)
    rx1 = min(W, x1 + 100)
    ry0 = max(0, y0 - 22)
    ry1 = min(H, y1 + 22)
    region[ry0:ry1, rx0:rx1] = 255

    base_dil = cv2.dilate(
        (base_a > 18).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 19))
    )

    blue_gray = (
        (h >= 88) & (h <= 132) &
        (s <= 125) &
        (v <= 155)
    )
    dark_gray = (s <= 28) & (v <= 108)
    bg_like = blue_gray | dark_gray

    candidate = (
        (detail_a > 16) &
        (region > 0) &
        ((~bg_like) | (detail_a >= 145))
    )

    mask = candidate.astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, count):
        x, y, w, hh, area = map(int, stats[i])
        if area < 10:
            continue

        comp = (labels[y:y + hh, x:x + w] == i)
        if not np.any(comp):
            continue

        comp_s = s[y:y + hh, x:x + w][comp]
        comp_v = v[y:y + hh, x:x + w][comp]
        comp_a = detail_a[y:y + hh, x:x + w][comp]

        touches_base = np.any(base_dil[y:y + hh, x:x + w][comp] > 0)
        near_row = abs((y + hh / 2) - cy) <= max(26, (y1 - y0) * 0.95)

        mean_s = float(np.mean(comp_s)) if comp_s.size else 0.0
        mean_v = float(np.mean(comp_v)) if comp_v.size else 0.0
        mean_a = float(np.mean(comp_a)) if comp_a.size else 0.0

        colorful_or_visible = (mean_s >= 28) or (mean_v >= 95) or (mean_a >= 120)
        not_tiny = (w >= 10) or (hh >= 10)

        if not_tiny and colorful_or_visible and (touches_base or near_row):
            part = keep[y:y + hh, x:x + w]
            part[comp] = 255

    final_a = np.maximum(base_a, keep)
    final_a = cv2.morphologyEx(
        final_a,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    final_a = cv2.GaussianBlur(final_a, (3, 3), 0.45)

    out = np.zeros_like(base)
    out[:, :, :3] = base_rgb

    add_mask = keep > base_a
    out[:, :, :3][add_mask] = detail_rgb[add_mask]
    out[:, :, 3] = final_a

    ok, encoded = cv2.imencode('.png', out)
    return encoded.tobytes() if ok else base_png


def process_combined(raw: bytes) -> bytes:
    base = transparent_extract(raw)
    try:
        detail = transparent_extract_detail(raw)
        merged = merge_base_detail_png(base, detail)
    except (ValueError, cv2.error, IndexError):
        merged = base

    return clean_background_png(merged, strength='normal')


@app.get('/')
def home():
    return FileResponse(ROOT / 'index.html', media_type='text/html')


@app.get('/health')
def health():
    return {
        'ok': True,
        'engine': 'killlog-auto-combined-v4',
        'result_size': '760x190',
        'endpoint': 'POST /extract-killlog',
        'mode': 'combined'
    }


@app.post('/extract-killlog')
async def extract(file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail='이미지는 10MB 이하여야 합니다.')

    async with GATE:
        try:
            png = await asyncio.to_thread(process_combined, raw)
        except ValueError as ex:
            raise HTTPException(400, detail=str(ex))

    return Response(png, media_type='image/png', headers={'Cache-Control': 'no-store'})
