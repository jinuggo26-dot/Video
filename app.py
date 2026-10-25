from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

app = FastAPI(title="Killlog auto extractor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

ROOT = Path(__file__).resolve().parent
GATE = asyncio.Semaphore(2)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 6_000_000
OUT_W, OUT_H = 760, 190


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(n)))


def _build_background(image: np.ndarray, ax: int, bx: int, ay: int, by: int) -> np.ndarray:
    H, W = image.shape[:2]
    w = bx - ax

    top1 = max(0, ay - 14)
    top2 = max(top1 + 1, ay - 2)
    bot1 = min(H - 2, by + 2)
    bot2 = min(H, by + 14)

    top_strip = image[top1:top2, ax:bx]
    bot_strip = image[bot1:bot2, ax:bx]

    if top_strip.size == 0:
        top_strip = image[max(0, ay - 1):ay, ax:bx]
    if bot_strip.size == 0:
        bot_strip = image[by:min(H, by + 1), ax:bx]

    top_line = cv2.resize(
        cv2.GaussianBlur(top_strip, (0, 0), 3.0),
        (w, 1),
        interpolation=cv2.INTER_AREA,
    )[0]
    bot_line = cv2.resize(
        cv2.GaussianBlur(bot_strip, (0, 0), 3.0),
        (w, 1),
        interpolation=cv2.INTER_AREA,
    )[0]

    h = by - ay
    ys = np.arange(h, dtype=np.float32)
    t = (ys / max(1, h - 1))[:, None, None]

    bg = top_line.astype(np.float32)[None, :, :] * (1.0 - t) + bot_line.astype(np.float32)[None, :, :] * t
    bg = cv2.GaussianBlur(bg, (7, 3), 0)
    return bg.astype(np.uint8)


def _score_strength(roi: np.ndarray, bg: np.ndarray) -> np.ndarray:
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab_roi - lab_bg) ** 2, axis=2))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)

    s1 = np.clip((diff - 13.0) * 7.0, 0, 255)
    s2 = np.clip((sat - 55.0) * 1.8, 0, 140) * np.clip((diff - 8.0) / 25.0, 0, 1)
    s3 = np.clip((val - 118.0) * 1.7, 0, 150) * np.clip((diff - 10.0) / 28.0, 0, 1)
    s4 = np.clip((edge - 18.0) * 3.8, 0, 120) * np.clip((diff - 5.0) / 18.0, 0, 1)

    strength = np.maximum.reduce([s1, s2, s3, s4]).astype(np.float32)

    # 어두운 파랑 / 회색 계열 배경 약화
    bg_like = (
        (hue >= 90) & (hue <= 125) &
        (sat <= 120) &
        (val <= 145) &
        (diff < 18)
    )
    strength[bg_like] *= 0.35

    return np.clip(strength, 0, 255).astype(np.float32)


def _pick_bounds(strength: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    h, w = strength.shape[:2]

    hard = np.where(strength > 42, 255, 0).astype(np.uint8)
    joined = cv2.morphologyEx(
        hard,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)),
    )
    joined = cv2.morphologyEx(
        joined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    comps: list[tuple[float, int, tuple[int, int, int, int, int]]] = []

    for i in range(1, num):
        x, y, ww, hh, area = map(int, stats[i])
        if area < 120 or hh < 8 or ww < max(60, round(w * 0.06)):
            continue

        cy = y + hh / 2.0
        cx = x + ww / 2.0
        score = ww * 4.2 + area * 0.2 - abs(cy - h * 0.43) * 4.0 - abs(cx - w * 0.50) * 0.35
        comps.append((score, i, (x, y, ww, hh, area)))

    if not comps:
        yy, xx = np.where(hard > 0)
        if len(xx) < 15:
            raise ValueError("킬로그 후보를 찾지 못했습니다. 킬 알림이 보이는 원본 화면을 올려 주세요.")

        x0 = _clamp(np.percentile(xx, 1) - 20, 0, w - 1)
        x1 = _clamp(np.percentile(xx, 99) + 35, x0 + 1, w)
        y0 = _clamp(np.percentile(yy, 1) - 10, 0, h - 1)
        y1 = _clamp(np.percentile(yy, 99) + 12, y0 + 1, h)

        chosen = np.zeros_like(hard)
        chosen[y0:y1, x0:x1] = 255
        return x0, y0, x1, y1, chosen

    comps.sort(reverse=True, key=lambda x: x[0])
    _, best_i, (bx0, by0, bw0, bh0, _) = comps[0]

    left = bx0
    right = bx0 + bw0
    top = by0
    bottom = by0 + bh0

    band_top = by0 - 18
    band_bottom = by0 + bh0 + 18
    merge_ids = {best_i}

    for _, i, (x, y, ww, hh, area) in comps[1:]:
        overlap_y = not (y + hh < band_top or y > band_bottom)
        near_x = (x <= right + 85) and (x + ww >= left - 85)

        if overlap_y and near_x:
            merge_ids.add(i)
            left = min(left, x)
            right = max(right, x + ww)
            top = min(top, y)
            bottom = max(bottom, y + hh)
            band_top = min(band_top, y - 10)
            band_bottom = max(band_bottom, y + hh + 10)

    chosen = np.zeros_like(hard)
    for i in merge_ids:
        chosen[labels == i] = 255

    # 왼쪽 장식, 오른쪽 꼬리 보존용 여백
    left = _clamp(left - 40, 0, w - 1)
    right = _clamp(right + 52, left + 1, w)
    top = _clamp(top - 10, 0, h - 1)
    bottom = _clamp(bottom + 12, top + 1, h)

    region = np.zeros_like(chosen)
    region[top:bottom, left:right] = 255

    dilated = cv2.dilate(chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    local_soft = np.where((dilated > 0) & (strength > 18), 255, 0).astype(np.uint8)

    chosen = cv2.bitwise_or(chosen, local_soft)
    chosen = cv2.bitwise_and(chosen, region)

    return left, top, right, bottom, chosen


def transparent_extract(raw: bytes) -> bytes:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPG·PNG·WEBP 이미지를 읽을 수 없습니다.")

    H, W = image.shape[:2]
    if W < 350 or H < 200 or W * H > MAX_PIXELS:
        raise ValueError("이미지 크기가 너무 작거나 큽니다(최대 600만 픽셀).")

    # 킬로그 영역을 넉넉하게 탐색
    ax, bx = round(W * 0.22), round(W * 0.76)
    ay, by = round(H * 0.08), round(H * 0.34)
    roi = image[ay:by, ax:bx].copy()

    bg = _build_background(image, ax, bx, ay, by)
    strength = _score_strength(roi, bg)
    x0, y0, x1, y1, chosen = _pick_bounds(strength)

    # 선택 영역 안에서만 부드러운 alpha 허용
    soft = np.clip((strength - 10.0) * 3.2, 0, 255).astype(np.uint8)
    alpha = np.where(chosen > 0, np.maximum(soft, chosen), 0).astype(np.uint8)

    # 작은 노이즈 제거
    mask = np.where(alpha > 16, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(alpha)

    for i in range(1, n):
        cx, cy, cw, ch, area = map(int, stats[i])
        if area < 40:
            continue
        if cw < 20 and ch < 10:
            continue
        clean[labels == i] = alpha[labels == i]

    alpha = cv2.GaussianBlur(clean, (3, 3), 0.55)

    # 중요:
    # alpha 최소/최대 기준으로 다시 타이트하게 자르지 않고
    # 위에서 잡은 bbox(x0,y0,x1,y1)를 그대로 사용
    color = roi[y0:y1, x0:x1]
    matte = alpha[y0:y1, x0:x1]

    if matte.size == 0 or np.count_nonzero(matte > 16) < 25:
        raise ValueError("킬로그 후보를 찾지 못했습니다. 다른 원본 화면으로 다시 시도해 주세요.")

    cut_h, cut_w = color.shape[:2]
    scale = min((OUT_W - 18) / max(1, cut_w), (OUT_H - 18) / max(1, cut_h))
    dst_w = max(1, int(round(cut_w * scale)))
    dst_h = max(1, int(round(cut_h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

    color = cv2.resize(color, (dst_w, dst_h), interpolation=interp)
    matte = cv2.resize(matte, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
    px = (OUT_W - dst_w) // 2
    py = (OUT_H - dst_h) // 2

    canvas[py:py + dst_h, px:px + dst_w, :3] = color
    canvas[py:py + dst_h, px:px + dst_w, 3] = matte

    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("PNG 인코딩에 실패했습니다.")

    return png.tobytes()


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "killlog-auto-no-grabcut-v2",
        "result_size": "760x190",
        "endpoint": "POST /extract-killlog",
    }


@app.post("/extract-killlog")
async def extract(file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)

    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail="이미지는 10MB 이하여야 합니다.")

    async with GATE:
        try:
            png = await asyncio.to_thread(transparent_extract, raw)
        except ValueError as ex:
            raise HTTPException(400, detail=str(ex))

    return Response(
        png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
