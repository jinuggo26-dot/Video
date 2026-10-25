import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image

app = Flask(__name__)

RESULT_W = 760
RESULT_H = 190

def remove_small_components(mask, min_area=200):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def get_largest_reasonable_component(mask, roi_shape):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = roi_shape[:2]
    best_idx = -1
    best_score = -1

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area < 1200:
            continue
        aspect = cw / max(ch, 1)
        cx, cy = centroids[i]

        # 킬로그 특성: 가로로 긴 형태 + ROI 상단 쪽 + 어느정도 넓이
        score = (
            area * 1.0
            + cw * 25
            + aspect * 1200
            - abs(cy - h * 0.38) * 60
        )
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        return None

    component_mask = np.zeros_like(mask)
    component_mask[labels == best_idx] = 255
    return component_mask

def tight_rgba_from_mask(bgr_crop, mask):
    if mask is None or cv2.countNonZero(mask) == 0:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = int((x2 - x1 + 1) * 0.05) + 8
    pad_y = int((y2 - y1 + 1) * 0.12) + 8

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(bgr_crop.shape[1] - 1, x2 + pad_x)
    y2 = min(bgr_crop.shape[0] - 1, y2 + pad_y)

    crop = bgr_crop[y1:y2+1, x1:x2+1]
    alpha = mask[y1:y2+1, x1:x2+1]

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba

def place_on_canvas(rgba, out_w=760, out_h=190):
    h, w = rgba.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 4), dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    x = (out_w - nw) // 2
    y = (out_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def make_foreground_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 배경은 대체로 어둡고 푸른 회색, 킬로그는 밝거나 채도가 높음
    # 테두리 색 평균으로 배경 추정
    border = np.concatenate([
        roi[0:8, :, :].reshape(-1, 3),
        roi[-8:, :, :].reshape(-1, 3),
        roi[:, 0:8, :].reshape(-1, 3),
        roi[:, -8:, :].reshape(-1, 3),
    ], axis=0)

    bg = np.median(border, axis=0).astype(np.uint8)
    diff = np.sqrt(np.sum((roi.astype(np.int16) - bg.astype(np.int16)) ** 2, axis=2)).astype(np.uint8)

    # 여러 특성 결합
    mask1 = (diff > 34).astype(np.uint8) * 255
    mask2 = ((s > 45) & (v > 55)).astype(np.uint8) * 255
    mask3 = (v > 120).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1, iterations=1)
    mask = remove_small_components(mask, min_area=150)

    return mask

def extract_killlog(img):
    h, w = img.shape[:2]

    # 네가 준 원본들 기준으로 킬로그가 주로 있는 위치
    x1 = int(w * 0.25)
    x2 = int(w * 0.76)
    y1 = int(h * 0.07)
    y2 = int(h * 0.28)

    roi = img[y1:y2, x1:x2].copy()

    raw_mask = make_foreground_mask(roi)
    target_mask = get_largest_reasonable_component(raw_mask, roi.shape)

    if target_mask is None:
        # 실패하면 넓은 고정 영역 fallback
        fx1 = int(w * 0.31)
        fx2 = int(w * 0.72)
        fy1 = int(h * 0.09)
        fy2 = int(h * 0.25)
        fallback = img[fy1:fy2, fx1:fx2].copy()

        raw_mask = make_foreground_mask(fallback)
        target_mask = remove_small_components(raw_mask, min_area=250)
        rgba = tight_rgba_from_mask(fallback, target_mask)
        if rgba is None:
            rgba = cv2.cvtColor(fallback, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
        return place_on_canvas(rgba, RESULT_W, RESULT_H)

    rgba = tight_rgba_from_mask(roi, target_mask)
    if rgba is None:
        rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255

    return place_on_canvas(rgba, RESULT_W, RESULT_H)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mode": "killlog-auto",
        "result_size": f"{RESULT_W}x{RESULT_H}"
    })

@app.route("/extract-killlog", methods=["POST"])
def extract_killlog_api():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "file 없음"}), 400

    file = request.files["file"]
    data = file.read()
    img_np = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"ok": False, "error": "이미지 읽기 실패"}), 400

    result = extract_killlog(img)

    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)

    return send_file(
        bio,
        mimetype="image/png",
        as_attachment=False,
        download_name="killlog.png"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
