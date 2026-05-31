import torch



def select_keyframe_by_semantics(current_semantic_id: torch.Tensor,
                                           last_kf_semantic_id: torch.Tensor,
                                           overlap_threshold=0.9,
                                           min_new_ratio=0.05):
    """
    基于语义图张量判断是否应选为关键帧

    Args:
        current_semantic_id: [H, W] tensor, 当前帧语义类别ID
        last_kf_semantic_id: [H, W] tensor, 上一个关键帧语义类别ID
        overlap_threshold: 语义重叠率阈值，低于此值认为变化大
        min_new_ratio: 新类别像素占比阈值，超过此值认为有新内容
    Returns:
        is_keyframe: bool
        info: dict, 包含重叠率、新类别等信息（用于调试）
    """
    # 确保在CPU上操作

    curr = current_semantic_id
    kf = last_kf_semantic_id

    # 1. 计算语义重叠率（Spatial Overlap）
    valid_mask = (curr != 0) & (kf != 0)  # 忽略背景（如ID=0），可按需调整
    if not torch.any(valid_mask):
        overlap_ratio = torch.tensor(0.0)
    else:
        match = (curr[valid_mask] == kf[valid_mask])
        overlap_ratio = match.float().mean()

    # 2. 检测新语义类别（Novel Classes）
    unique_kf = torch.unique(kf).tolist()           # 上一关键帧出现的类别
    unique_curr = torch.unique(curr).tolist()       # 当前帧出现的类别
    new_classes = set(unique_curr) - set(unique_kf)  # 新出现的类别

    # 计算新类别像素占比
    new_pixels_mask = torch.zeros_like(curr, dtype=torch.bool)
    for cls in new_classes:
        new_pixels_mask |= (curr == cls)
    new_ratio = new_pixels_mask.float().mean()

    # 判断是否为关键帧
    condition_overlap = overlap_ratio < overlap_threshold  # 重叠太少
    condition_new = new_ratio > min_new_ratio              # 有显著新内容

    is_keyframe = condition_overlap or condition_new

    return bool(is_keyframe), {
        'overlap_ratio': float(overlap_ratio),
        'new_ratio': float(new_ratio),
        'new_classes': list(new_classes),
        'condition_overlap': condition_overlap.item(),
        'condition_new': condition_new.item()
    }