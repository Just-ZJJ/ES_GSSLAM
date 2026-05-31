"""
语义标签加噪脚本（严格对齐版 + 帧级随机采样）

任务目标：
    1. 模拟边缘分割错误：对 semantic_ids 进行腐蚀/膨胀，semantic_colors 同步重渲染
    2. 模拟语义缺失：随机矩形块区域内 semantic_ids 设为 255，semantic_colors 对应区域设为黑色
    3. 模拟随机分类噪声：将指定比例像素的标签随机替换为其他合法类别，semantic_colors 同步更新
    4. ✅ 帧级采样：仅 NOISE_FRAME_PROB 比例的图像被施加噪声，其余保持原始干净状态

核心原则：
    - 所有噪声先作用于 semantic_ids，再根据 noisy_ids 重新渲染 semantic_colors
    - 保证输出的一对图像在任意像素位置上语义严格一致
    - 不依赖外部调色板字典，直接从原始 color 图中提取当前帧的实际颜色映射
    - 干净帧也写入输出目录并走统一渲染流程，保证下游数据索引不断裂
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


# ======================== 配置区 ========================

DATA_ROOT = "/home/6t/ZJJ/dataset/gsData/semScannet/scene0169_00"
OUTPUT_ROOT = "/home/6t/ZJJ/dataset/gsData/semScannet/scene0169_00/noisy_semantic_output"
NUM_SAMPLES = None  # 设为 None 处理全部

# ---------- 噪声开关 ----------
ENABLE_BOUNDARY = True      # 任务1: 边缘分割错误（腐蚀/膨胀）
ENABLE_BLOCK = True         # 任务2: 语义缺失（矩形块遮挡）
ENABLE_SALT_PEPPER = True   # 任务3: 随机椒盐噪点

# ---------- ✅ 帧级采样参数 ----------
NOISE_FRAME_PROB = 1      # 10% 的图像会被加噪，其余保持原样
GLOBAL_SEED = 42            # 帧级采样全局种子，保证每次运行选中的噪声帧相同（可复现）

# ---------- 噪声参数 ----------
BOUNDARY_KERNEL_SIZE = 20     # 边缘噪声结构元素大小（奇数）
BLOCK_NUM = 4                 # 遮挡矩形块数量
BLOCK_SIZE_RANGE = (50, 100)   # 矩形块边长范围 (min, max)
NOISE_RATIO = 0.01            # 椒盐噪点：被替换像素占总像素的比例
IGNORE_LABEL = 255            # 语义缺失区域的标签值
UNKNOWN_COLOR_BGR = (0, 0, 0) # 语义缺失区域的颜色（黑色）

# ======================================================


def extract_color_map(semantic_id: np.ndarray, semantic_color: np.ndarray) -> dict:
    """从原始数据中动态提取当前帧的 {label_id: bgr_color} 映射"""
    color_map = {}
    unique_labels = np.unique(semantic_id)
    for label in unique_labels:
        if label == IGNORE_LABEL:
            continue
        mask = semantic_id == label
        y, x = np.where(mask)
        if len(y) > 0:
            color_map[int(label)] = tuple(semantic_color[y[0], x[0]].tolist())
    return color_map


def apply_boundary_noise(semantic_id: np.ndarray, kernel_size: int) -> np.ndarray:
    """任务1: 模拟边缘分割错误（腐蚀/膨胀）"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if random.random() > 0.5:
        result = cv2.dilate(semantic_id.astype(np.uint8), kernel)
    else:
        result = cv2.erode(semantic_id.astype(np.uint8), kernel)
    return result.astype(np.int64)


def apply_block_masking(semantic_id: np.ndarray, num_blocks: int,
                        size_range: tuple, ignore_label: int) -> np.ndarray:
    """任务2: 模拟语义缺失（矩形块遮挡）"""
    noisy = semantic_id.copy()
    H, W = semantic_id.shape[:2]
    for _ in range(num_blocks):
        bh = random.randint(*size_range)
        bw = random.randint(*size_range)
        top = random.randint(0, max(0, H - bh))
        left = random.randint(0, max(0, W - bw))
        noisy[top:top + bh, left:left + bw] = ignore_label
    return noisy


def apply_salt_pepper_noise(semantic_id: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    任务3: 随机椒盐噪点
    
    将指定比例的像素随机替换为图中已存在的其他合法类别ID。
    由于只从当前帧实际出现的类别中采样，配合 render_color_from_ids 重渲染，
    从根本上杜绝了 color_map 缺失导致的黑边问题。
    """
    noisy = semantic_id.copy()
    H, W = noisy.shape[:2]

    valid_labels = np.unique(noisy)
    valid_labels = valid_labels[valid_labels != IGNORE_LABEL]

    if len(valid_labels) < 2:
        print("  ⚠️ 有效类别数<2，跳过椒盐噪点")
        return noisy

    num_noise = int(H * W * noise_ratio)
    if num_noise == 0:
        return noisy

    # 随机选择加噪位置
    noise_ys = np.random.randint(0, H, num_noise)
    noise_xs = np.random.randint(0, W, num_noise)

    # 仅对非忽略区域进行替换，避免污染已有mask
    valid_mask = noisy[noise_ys, noise_xs] != IGNORE_LABEL
    target_ys = noise_ys[valid_mask]
    target_xs = noise_xs[valid_mask]

    # 从合法类别中随机采样并赋值
    new_labels = np.random.choice(valid_labels, size=len(target_ys))
    noisy[target_ys, target_xs] = new_labels

    print(f"  🧂 椒盐噪点: {len(target_ys)}/{H*W} 像素被替换 (ratio={noise_ratio})")
    return noisy


def render_color_from_ids(semantic_id: np.ndarray, color_map: dict,
                          ignore_label: int, unknown_color: tuple) -> np.ndarray:
    """根据语义ID图和颜色映射表重新渲染颜色图（严格对齐核心）"""
    H, W = semantic_id.shape[:2]
    color_img = np.zeros((H, W, 3), dtype=np.uint8)

    for label_id, bgr in color_map.items():
        mask = semantic_id == label_id
        color_img[mask] = bgr

    ignore_mask = semantic_id == ignore_label
    color_img[ignore_mask] = unknown_color

    return color_img


def process_single(id_path: Path, color_path: Path, output_root: Path,
                   apply_noise: bool = True):
    """
    处理单帧：读取 → [可选]加噪ID → 重渲染Color → 保存
    apply_noise=False 时跳过所有噪声，直接保存原始数据的重渲染版本
    """
    semantic_id = cv2.imread(str(id_path), cv2.IMREAD_UNCHANGED).astype(np.int64)
    semantic_color = cv2.imread(str(color_path))

    if semantic_id is None or semantic_color is None:
        print(f"  ❌ 读取失败: {id_path.name}")
        return

    color_map = extract_color_map(semantic_id, semantic_color)

    # ✅ 核心改动：未命中采样时直接使用原始ID，但仍走重渲染流程
    if apply_noise:
        base_seed = hash(id_path.stem) % (2 ** 31)
        random.seed(base_seed)
        np.random.seed(base_seed)

        noisy_id = semantic_id.copy()
        if ENABLE_BOUNDARY:
            noisy_id = apply_boundary_noise(noisy_id, BOUNDARY_KERNEL_SIZE)
        if ENABLE_BLOCK:
            noisy_id = apply_block_masking(noisy_id, BLOCK_NUM, BLOCK_SIZE_RANGE, IGNORE_LABEL)
        if ENABLE_SALT_PEPPER:
            noisy_id = apply_salt_pepper_noise(noisy_id, NOISE_RATIO)
        status_tag = "🔊 NOISY"
    else:
        noisy_id = semantic_id.copy()
        status_tag = "🔇 CLEAN"

    noisy_color = render_color_from_ids(noisy_id, color_map, IGNORE_LABEL, UNKNOWN_COLOR_BGR)

    stem = id_path.stem.replace("semantic_id", "")
    id_out_dir = output_root / "semantic_ids"
    color_out_dir = output_root / "semantic_colors"
    id_out_dir.mkdir(parents=True, exist_ok=True)
    color_out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 自适应保存精度，防止 ID>255 时 uint8 截断导致对齐失效
    max_label = int(noisy_id.max())
    if max_label <= 255:
        save_dtype = np.uint8
    elif max_label <= 65535:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    cv2.imwrite(str(id_out_dir / f"semantic_id{stem}.png"), noisy_id.astype(save_dtype))
    cv2.imwrite(str(color_out_dir / f"semantic_color{stem}.png"), noisy_color)

    print(f"  {status_tag} | {id_path.name}")


if __name__ == "__main__":
    id_dir = Path(DATA_ROOT) / "semantic_id"
    all_id_files = sorted(id_dir.glob("*.png"))

    if not all_id_files:
        raise FileNotFoundError(f"未在 {id_dir} 中找到 *.png")

    samples = all_id_files if NUM_SAMPLES is None else all_id_files[:NUM_SAMPLES]

    # ✅ 帧级随机采样：预先决定每帧是否加噪
    rng = random.Random(GLOBAL_SEED)
    noise_flags = {fp: (rng.random() < NOISE_FRAME_PROB) for fp in samples}
    noisy_count = sum(noise_flags.values())
    clean_count = len(samples) - noisy_count

    active = []
    if ENABLE_BOUNDARY: active.append("boundary")
    if ENABLE_BLOCK: active.append("block")
    if ENABLE_SALT_PEPPER: active.append("salt_pepper")
    tag = "+".join(active) if active else "original"

    print(f"📂 共 {len(all_id_files)} 张，本次处理 {len(samples)} 张")
    print(f"🎲 帧级采样: {noisy_count} 张加噪 ({NOISE_FRAME_PROB*100:.0f}%) + {clean_count} 张干净")
    print(f"⚙️ 噪声类型: [{tag}]")
    print("-" * 60)

    skipped = 0
    for id_path in tqdm(samples, desc="Processing"):
        idx = id_path.stem.replace("semantic_id", "")
        color_path = Path(DATA_ROOT) / "semantic_color" / f"{idx}.png"
        if not color_path.exists():
            skipped += 1
            continue
        # ✅ 传入该帧的噪声标志
        process_single(id_path, color_path, Path(OUTPUT_ROOT),
                       apply_noise=noise_flags[id_path])

    if skipped:
        print(f"⚠️ 跳过 {skipped} 张（缺少对应color文件）")
    print(f"\n🎉 完成 → {os.path.abspath(OUTPUT_ROOT)}")