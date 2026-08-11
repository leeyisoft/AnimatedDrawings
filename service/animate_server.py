# DrawCub 本地动画服务：PNG -> 透明 GIF
# 跳过 torchserve：用固定人形骨架按前景包围盒比例生成 char_cfg 标注。
# 渲染放在子进程里跑，避免 glfw 在同一进程重复 init/terminate 的问题。
# 必须在主线程、单线程运行（macOS 上 glfw 窗口只能在主线程创建）。

import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage
from flask import Flask, request, jsonify, send_file

SERVICE_DIR = Path(__file__).resolve().parent
AD_ROOT = SERVICE_DIR.parent
JOBS_DIR = SERVICE_DIR / 'jobs'
MOTION_DIR = AD_ROOT / 'examples' / 'config' / 'motion'
RETARGET_CFG = AD_ROOT / 'examples' / 'config' / 'retarget' / 'fair1_ppf.yaml'

MOTIONS = {p.stem: p for p in MOTION_DIR.glob('*.yaml')}

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 关节位置以 (宽占比, 高占比) 表示，比例参照 examples/characters/char1/char_cfg.yaml
SKELETON_TEMPLATE = [
    ('root',            None,             (0.50, 0.66)),
    ('hip',             'root',           (0.50, 0.66)),
    ('torso',           'hip',            (0.50, 0.39)),
    ('neck',            'torso',          (0.50, 0.20)),
    ('right_shoulder',  'torso',          (0.30, 0.40)),
    ('right_elbow',     'right_shoulder', (0.18, 0.47)),
    ('right_hand',      'right_elbow',    (0.09, 0.53)),
    ('left_shoulder',   'torso',          (0.70, 0.40)),
    ('left_elbow',      'left_shoulder',  (0.82, 0.44)),
    ('left_hand',       'left_elbow',     (0.91, 0.50)),
    ('right_hip',       'root',           (0.38, 0.66)),
    ('right_knee',      'right_hip',      (0.33, 0.78)),
    ('right_foot',      'right_knee',     (0.28, 0.92)),
    ('left_hip',        'root',           (0.62, 0.66)),
    ('left_knee',       'left_hip',       (0.68, 0.77)),
    ('left_foot',       'left_knee',      (0.75, 0.91)),
]


def segment(bgr: np.ndarray) -> np.ndarray:
    """从纸面照片里抠角色：自适应阈值 + 形态学闭运算封边 + 边缘洪水填充取反。
    逻辑移植自 examples/image_to_annotations.py 的 segment()（同仓库 MIT）。"""
    img = np.min(bgr, axis=2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 115, 8)
    img = cv2.bitwise_not(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=2)

    h, w = img.shape
    flood_mask = np.zeros([h + 2, w + 2], np.uint8)
    flood_mask[1:-1, 1:-1] = img.copy()

    # 从四边种子点洪水填充背景；角色线条在 flood_mask 里是障碍物
    im_floodfill = np.full(img.shape, 255, np.uint8)
    for x in range(0, w - 1, 10):
        cv2.floodFill(im_floodfill, flood_mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, flood_mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(im_floodfill, flood_mask, (0, y), 0)
        cv2.floodFill(im_floodfill, flood_mask, (w - 1, y), 0)

    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    # 洪水填充后仍为 255 的 = 角色线条 + 被围住的内部；只保留最大连通域
    count, labels = cv2.connectedComponents(im_floodfill)
    if count <= 1:
        raise ValueError('前景提取失败：纸面照片里没找到角色轮廓')
    biggest = 1 + np.argmax([np.sum(labels == i) for i in range(1, count)])
    mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    return mask


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """前景掩码：有真实 alpha 用 alpha（drawcub 导出）；否则按纸面照片抠图。"""
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        if alpha.min() < 250:  # 存在真实透明区域
            return (alpha > 10).astype(np.uint8) * 255
    return segment(img[:, :, :3])


def snap_joints_to_mask(skeleton, mask: np.ndarray) -> None:
    """关节落在角色外时，吸附到最近的前景像素，保证网格化不失败。"""
    h, w = mask.shape
    fg = np.column_stack(np.where(mask > 0))  # (y, x)
    for joint in skeleton:
        x, y = joint['loc']
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if mask[y, x] == 0 and len(fg) > 0:
            dists = (fg[:, 0] - y) ** 2 + (fg[:, 1] - x) ** 2
            nearest = fg[np.argmin(dists)]
            y, x = int(nearest[0]), int(nearest[1])
        joint['loc'] = [x, y]


def build_annotations(img: np.ndarray, job_dir: Path, skeleton_override=None) -> None:
    mask = foreground_mask(img)
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        raise ValueError('前景太小，未检测到可动画的角色（需要透明背景或白底彩绘）')
    l, r, t, b = xs.min(), xs.max(), ys.min(), ys.max()

    cropped_img = img[t:b + 1, l:r + 1]
    cropped_mask = mask[t:b + 1, l:r + 1]
    h, w = cropped_mask.shape

    # 缩到最长边 <= 1000，与官方流程一致
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        new_w, new_h = round(w * scale), round(h * scale)
        cropped_img = cv2.resize(cropped_img, (new_w, new_h))
        cropped_mask = cv2.resize(cropped_mask, (new_w, new_h))
        h, w = new_h, new_w

    # texture：BGRA，alpha = mask
    if cropped_img.shape[2] == 4:
        bgra = cropped_img.copy()
        bgra[:, :, 3] = cropped_mask
    else:
        bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = cropped_mask
    cv2.imwrite(str(job_dir / 'texture.png'), bgra)
    cv2.imwrite(str(job_dir / 'mask.png'), cropped_mask)

    if skeleton_override:
        skeleton = skeleton_override
    else:
        skeleton = [
            {'name': name, 'parent': parent,
             'loc': [round(fx * w), round(fy * h)]}
            for name, parent, (fx, fy) in SKELETON_TEMPLATE
        ]
        snap_joints_to_mask(skeleton, cropped_mask)

    char_cfg = {'skeleton': skeleton, 'height': h, 'width': w}
    with open(str(job_dir / 'char_cfg.yaml'), 'w') as f:
        yaml.dump(char_cfg, f)

    overlay = bgra.copy()
    for joint in skeleton:
        x, y = joint['loc']
        cv2.circle(overlay, (x, y), 5, (0, 0, 255, 255), -1)
    cv2.imwrite(str(job_dir / 'joint_overlay.png'), overlay)


def render_gif(job_dir: Path, motion: str) -> Path:
    motion_cfg = MOTION_DIR / f'{motion}.yaml'
    
    # Select appropriate retargeting config based on motion source
    if motion == 'jumping_jacks':
        retarget_fn = AD_ROOT / 'examples' / 'config' / 'retarget' / 'cmu1_pfp.yaml'
    else:
        retarget_fn = RETARGET_CFG

    mvc_cfg = {
        'scene': {'ANIMATED_CHARACTERS': [{
            'character_cfg': str(job_dir / 'char_cfg.yaml'),
            'motion_cfg': str(motion_cfg),
            'retarget_cfg': str(retarget_fn),
        }]},
        'view': {
            'WINDOW_DIMENSIONS': [300, 300],
        },
        'controller': {
            'MODE': 'video_render',
            'OUTPUT_VIDEO_PATH': str(job_dir / 'video.gif'),
        },
    }
    mvc_cfg_fn = job_dir / 'mvc_cfg.yaml'
    with open(str(mvc_cfg_fn), 'w') as f:
        yaml.dump(mvc_cfg, f)

    proc = subprocess.run(
        [sys.executable, '-c',
         'import sys; from animated_drawings import render; render.start(sys.argv[1])',
         str(mvc_cfg_fn)],
        capture_output=True, text=True, timeout=600, cwd=str(AD_ROOT),
    )
    gif = job_dir / 'video.gif'
    if proc.returncode != 0 or not gif.exists():
        logging.error('render failed\nstdout: %s\nstderr: %s', proc.stdout[-3000:], proc.stderr[-3000:])
        raise RuntimeError('渲染失败，详情见服务日志')
    return gif


@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Private-Network'] = 'true'
    return resp


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'motions': sorted(MOTIONS.keys())})


@app.route('/animate', methods=['POST', 'OPTIONS'])
def animate():
    if request.method == 'OPTIONS':
        return ('', 204)

    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'error': '缺少 multipart 字段 image（PNG）'}), 400
    motion = request.form.get('motion', 'jumping')
    if motion not in MOTIONS:
        return jsonify({'error': f'未知 motion={motion}，可选: {sorted(MOTIONS.keys())}'}), 400

    skeleton_override = None
    if 'skeleton' in request.form:
        try:
            skeleton_override = json.loads(request.form['skeleton'])
        except json.JSONDecodeError:
            return jsonify({'error': 'skeleton 字段不是合法 JSON'}), 400

    file_bytes = np.frombuffer(request.files['image'].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None or len(img.shape) != 3:
        return jsonify({'error': 'image 不是可解码的 PNG 图片'}), 400

    logging.info(f"🚀 [Request] 收到活化请求: motion={motion}, 图像通道={img.shape}")

    job_dir = JOBS_DIR / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(job_dir / 'image.png'), img)

    try:
        t0 = time.time()
        logging.info(f"✍️ [Step 1/2] 正在计算前景轮廓并进行自适应骨骼位置校准...")
        build_annotations(img, job_dir, skeleton_override)
        t1 = time.time()
        logging.info(f"   -> 步骤 1 耗时: {t1 - t0:.3f} 秒")

        logging.info(f"🎨 [Step 2/2] 正在拉起 Meta FAIR 3D 渲染子进程（开启 3x 帧率降采样 + 512x512 高速分辨率）...")
        gif = render_gif(job_dir, motion)
        t2 = time.time()
        logging.info(f"   -> 步骤 2 耗时: {t2 - t1:.3f} 秒")

        logging.info(f"✨ [Success] 活化完成！输出透明动画 GIF: {gif.name} | 全程总耗时: {t2 - start_time:.3f} 秒")
    except (ValueError, RuntimeError) as e:
        logging.error(f"❌ [Error] 活化失败: {e} | 耗时: {time.time() - start_time:.3f} 秒")
        return jsonify({'error': str(e), 'job': job_dir.name}), 422

    return send_file(str(gif), mimetype='image/gif')


if __name__ == '__main__':
    JOBS_DIR.mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=8765, threaded=True)
