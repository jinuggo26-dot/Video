"""PUBG-style kill notification image crop API.
No GrabCut or third-party background-removal service. Runs in Render with Uvicorn.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

app = FastAPI(title="Killlog auto extractor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"], allow_credentials=False)
ROOT = Path(__file__).resolve().parent
GATE = asyncio.Semaphore(2)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 6_000_000
OUT_W, OUT_H = 760, 190


def transparent_extract(raw: bytes) -> bytes:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPG·PNG·WEBP 이미지를 읽을 수 없습니다.")
    H, W = image.shape[:2]
    if W < 350 or H < 200 or W * H > MAX_PIXELS:
        raise ValueError("이미지 크기가 너무 작거나 큽니다(최대 600만 픽셀).")

    # Search in the notification area; restrict background estimation to the same columns.
    # Geometry is approximate rather than a rigid full image crop and allows decoration overhangs.
    ax, bx = round(W * 0.285), round(W * 0.715)
    ay, by = round(H * 0.11), round(H * 0.305)
    roi = image[ay:by, ax:bx].copy()
    h, w = roi.shape[:2]
    # Estimate the locally varying background by interpolating just outside the banner band.
    above_y = max(0, round(H * 0.102))
    below_y = min(H - 1, round(H * 0.318))
    above = cv2.GaussianBlur(image[above_y:above_y+1, ax:bx].copy(), (25, 1), 0)[0]
    below = cv2.GaussianBlur(image[below_y:below_y+1, ax:bx].copy(), (25, 1), 0)[0]
    # When color around the target changes smoothly, linear interpolation is closer than median border color.
    ys = np.arange(ay, by, dtype=np.float32)
    weight = np.clip((ys - above_y) / max(1, below_y - above_y), 0, 1)[:, None, None]
    background = above.astype(np.float32)[None, :, :] * (1-weight) + below.astype(np.float32)[None, :, :] * weight
    background = cv2.GaussianBlur(background, (7, 3), 0)
    lab_fore = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_back = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab_fore - lab_back) ** 2, axis=2))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    # Color/brightness cues are evidence, not a guarantee of true segmentation.
    strength = np.clip((diff - 17.0) * 6.0, 0, 255)
    saturated = np.clip((sat - 75) * 2.0, 0, 120) * np.clip((diff - 9) / 25, 0, 1)
    bright = np.clip((val - 145) * 2.4, 0, 180) * np.clip((diff - 12) / 30, 0, 1)
    strength = np.maximum.reduce([strength, saturated, bright])

    hard = np.where(strength > 45, 255, 0).astype(np.uint8)
    # Prefer long horizontal connected pieces typical of notification bars.
    joined = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    num, lab, stats, centers = cv2.connectedComponentsWithStats(joined, 8)
    plausible = []
    for i in range(1, num):
        x, y, ww, hh, area = map(int, stats[i])
        if ww < max(95, round(w*.18)) or hh < 10 or area < 250:
            continue
        cy = y+hh/2
        score = ww*4 + area/4 - abs(cy - h*.42)*3
        plausible.append((score, i))
    chosen = np.zeros_like(hard)
    if plausible:
        _, idx = max(plausible)
        chosen[lab == idx] = 255
        # Gently include nearby graphic detail and thin highlights beyond the solid bar.
        dilated = cv2.dilate(chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        chosen = np.where((dilated>0) & (strength>17), 255, chosen).astype(np.uint8)
        # connected fragments like isolated particles close to the banner
        hard_local = np.where((dilated>0) & (strength>18), 255, 0).astype(np.uint8)
        chosen = cv2.bitwise_or(chosen, hard_local)
    else:
        # When no long component is detected, return local high-confidence pixels instead of an HTTP error.
        chosen = hard

    # Localize a sensible horizontal region to prevent stray screen interface artifacts.
    # Limit to the region surrounding detected notification lines, not all foreground in the ROI.
    yy, xx = np.where(chosen > 0)
    if len(xx) >= 30:
        xl, xr = max(0, int(np.percentile(xx, 1))-18), min(w, int(np.percentile(xx, 99))+19)
        yt, yb = max(0, int(np.percentile(yy, 1))-15), min(h, int(np.percentile(yy, 99))+16)
        region = np.zeros_like(chosen)
        region[yt:yb, xl:xr] = 255
        alpha = (np.maximum(chosen, np.clip((strength-12)*3.0, 0, 255).astype(np.uint8)) & region)
    else:
        # Safe fallback that is still transparent rather than opaque screenshot rectangle.
        alpha = np.where(strength>32, np.clip(strength,0,255), 0).astype(np.uint8)
    # Discard tiny noise and avoid turning a dark screen rectangle opaque.
    alpha = cv2.GaussianBlur(alpha, (3, 3), .6)
    yy, xx = np.where(alpha > 18)
    if len(xx) < 15:
        raise ValueError("킬로그 후보를 찾지 못했습니다. 킬 알림이 보이는 원본 화면을 올려 주세요.")

    x0, x1 = max(0,int(xx.min())-5), min(w,int(xx.max())+6)
    y0, y1 = max(0,int(yy.min())-5), min(h,int(yy.max())+6)
    color = roi[y0:y1, x0:x1]
    matte = alpha[y0:y1, x0:x1]
    cw, ch = x1-x0, y1-y0
    scale = min((OUT_W-16)/max(1,cw), (OUT_H-16)/max(1,ch))
    dw, dh = max(1,int(round(cw*scale))), max(1,int(round(ch*scale)))
    interp = cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC
    color = cv2.resize(color, (dw,dh), interpolation=interp)
    matte = cv2.resize(matte, (dw,dh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((OUT_H,OUT_W,4), dtype=np.uint8)
    xpad, ypad = (OUT_W-dw)//2, (OUT_H-dh)//2
    canvas[ypad:ypad+dh,xpad:xpad+dw,:3]=color
    canvas[ypad:ypad+dh,xpad:xpad+dw,3]=matte
    success, png = cv2.imencode('.png',canvas)
    if not success: raise ValueError("PNG 인코딩에 실패했습니다.")
    return png.tobytes()


@app.get('/')
def home():
    return FileResponse(ROOT/'index.html', media_type='text/html')


@app.get('/health')
def health():
    return {'ok': True, 'engine': 'killlog-auto-no-grabcut', 'result_size':'760x190', 'endpoint':'POST /extract-killlog'}


@app.post('/extract-killlog')
async def extract(file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail='이미지는 10MB 이하여야 합니다.')
    async with GATE:
        try:
            png = await asyncio.to_thread(transparent_extract, raw)
        except ValueError as ex:
            raise HTTPException(400, detail=str(ex))
    return Response(png, media_type='image/png', headers={'Cache-Control':'no-store'})
