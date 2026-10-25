"""Sna Killlog external processor. Images processed only in RAM."""
from __future__ import annotations
import json
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Sna Killlog GrabCut", docs_url=None, redoc_url=None)
PROCESSING_GATE = asyncio.Semaphore(2)
MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 5_000_000
MAX_CROP_PIXELS = 1_200_000
MAX_MARKS = 2000

def limit(n: Any, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(n)))

def process_image(data: bytes, rect_json: str, marks_json: str) -> bytes:
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError("이미지는 10MB 이하여야 합니다.")
    im = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("JPG·PNG 이미지 파일만 지원합니다.")
    h, w = im.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError("이미지 크기가 너무 큽니다. 500만 픽셀 이하로 자르세요.")
    try:
        box = json.loads(rect_json)
        marks = json.loads(marks_json)
    except (ValueError, TypeError):
        raise ValueError("선택 영역 또는 브러시 데이터가 올바르지 않습니다.")
    if not isinstance(box, dict) or not isinstance(marks, list) or len(marks) > MAX_MARKS:
        raise ValueError("잘못된 선택 영역 또는 브러시 표시입니다.")
    try:
        x1 = limit(box['x'], 0, w-1)
        y1 = limit(box['y'], 0, h-1)
        x2 = limit(int(box['x'])+int(box['w']), x1+1, w)
        y2 = limit(int(box['y'])+int(box['h']), y1+1, h)
    except (KeyError, TypeError, ValueError):
        raise ValueError("킬로그 선택 영역을 다시 지정하세요.")
    if (x2-x1)<10 or (y2-y1)<8:
        raise ValueError("킬로그 영역을 조금 더 크게 선택하세요.")
    valid_marks: list[tuple[str,int,int,int]] = []
    for point in marks:
        if not isinstance(point, dict) or point.get('type') not in ('fg', 'bg'):
            continue
        try:
            px=limit(point['x'],0,w-1);py=limit(point['y'],0,h-1)
            radius=limit(point.get('r',6),1,55)
        except (KeyError, TypeError, ValueError):
            continue
        valid_marks.append((point['type'],px,py,radius))
        if point['type']=='fg':
            # Explicit restore strokes may reach outside a mistaken initial crop.
            x1=min(x1,max(0,px-radius));y1=min(y1,max(0,py-radius))
            x2=max(x2,min(w,px+radius+1));y2=max(y2,min(h,py+radius+1))
    # Only use the actual specified crop. Do not guess foreground by screenshot size.
    pad = max(12, min(36, round((y2-y1)*.3)))
    ax=max(0,x1-pad);ay=max(0,y1-pad)
    bx=min(w,x2+pad);by=min(h,y2+pad)
    if (bx-ax)*(by-ay)>MAX_CROP_PIXELS:
        raise ValueError("분리 영역이 너무 큽니다. 킬로그 주변만 지정하세요.")
    crop=im[ay:by,ax:bx].copy()
    ch,cw=crop.shape[:2]
    mask=np.full((ch,cw), cv2.GC_BGD, np.uint8)
    lx=x1-ax;ly=y1-ay;rx=x2-ax;ry=y2-ay
    mask[ly:ry,lx:rx]=cv2.GC_PR_FGD
    # Setting a small region inside the box as definite foreground prevents
    # GrabCut from having no seeds when the user only supplied a rectangle.
    cX=(lx+rx)//2;cY=(ly+ry)//2
    coreW=max(2,round((rx-lx)*.08));coreH=max(2,round((ry-ly)*.15))
    mask[max(0,cY-coreH):min(ch,cY+coreH),max(0,cX-coreW):min(cw,cX+coreW)]=cv2.GC_FGD
    # User FG/BG strokes are hard seeds rather than color similarities.
    for typ,px,py,r in valid_marks:
        cv2.circle(mask,(px-ax,py-ay),r,cv2.GC_FGD if typ=='fg' else cv2.GC_BGD,-1)
    if not np.any(mask==cv2.GC_FGD):
        raise ValueError("살릴 부분이 전부 지워졌습니다. 킬로그에 살리기를 표시하세요.")
    bg_model=np.zeros((1,65),np.float64)
    fg_model=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(crop,mask,None,bg_model,fg_model,4,cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        raise ValueError("자동 분리에 실패했습니다. 살리기와 지우기를 조금씩 표시해 보세요.")
    alpha=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype('uint8')
    # Explicit brush strokes are ALWAYS restored or erased after segmentation.
    for typ,px,py,r in valid_marks:
        cv2.circle(alpha,(px-ax,py-ay),r,255 if typ=='fg' else 0,-1)
    # mild edge antialiasing, constrained not to bring back distant background
    alpha=cv2.GaussianBlur(alpha,(3,3),0.55)
    for typ,px,py,r in valid_marks:
        cv2.circle(alpha,(px-ax,py-ay),max(1,r-1),255 if typ=='fg' else 0,-1)
    yy,xx=np.where(alpha>=8)
    if len(xx)<15:
        raise ValueError("킬로그가 감지되지 않았습니다. 살리기 표시와 선택 영역을 확인하세요.")
    l=max(0,int(xx.min())-2);t=max(0,int(yy.min())-2)
    r=min(cw,int(xx.max())+3);b=min(ch,int(yy.max())+3)
    cut=crop[t:b,l:r]
    am=alpha[t:b,l:r]
    scale=min(744/max(1,r-l),174/max(1,b-t))
    ow=max(1,round((r-l)*scale));oh=max(1,round((b-t)*scale))
    method=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC
    dest=cv2.resize(cut,(ow,oh),interpolation=method)
    a=cv2.resize(am,(ow,oh),interpolation=cv2.INTER_LINEAR)
    rgba=cv2.cvtColor(dest,cv2.COLOR_BGR2BGRA)
    rgba[:,:,3]=a
    full=np.zeros((190,760,4),dtype=np.uint8)
    dx=(760-ow)//2;dy=(190-oh)//2
    full[dy:dy+oh,dx:dx+ow]=rgba
    ok,encoded=cv2.imencode('.png',full)
    if not ok:
        raise ValueError("PNG 변환 실패")
    return encoded.tobytes()

@app.get('/')
def editor():
    return FileResponse(APP_DIR / 'editor.html', media_type='text/html')

@app.get('/health')
def health():
    return {'ok':True,'engine':'grabcut','result_size':'760x190'}

@app.post('/api/grabcut')
async def grabcut(file:UploadFile=File(...),rect:str=Form(...),marks:str=Form('[]')):
    if file.content_type not in ('image/jpeg','image/png','image/webp','application/octet-stream'):
        raise HTTPException(status_code=415,detail='JPG 또는 PNG를 선택하세요.')
    raw=await file.read(MAX_UPLOAD+1)
    if len(raw)>MAX_UPLOAD:
        raise HTTPException(status_code=413,detail='이미지는 10MB 이하여야 합니다.')
    async with PROCESSING_GATE:
        try:
            result=await asyncio.to_thread(process_image,raw,rect,marks)
        except ValueError as ex:
            raise HTTPException(status_code=400,detail=str(ex))
    return Response(result,media_type='image/png',headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
