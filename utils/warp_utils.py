import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def build_transform_matrix(R, t):
    T = torch.zeros(4, 4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    T[3, 3] = 1
    return T

def quat_to_rot(q):
    q = F.normalize(q, dim=1)
    R = torch.ones((3, 3)).cuda()
    qr = q[0]
    qi = q[ 1]
    qj = q[ 2]
    qk = q[ 3]
    R[ 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[ 0, 1] = 2 * (qj * qi - qk * qr)
    R[ 0, 2] = 2 * (qi * qk + qr * qj)
    R[ 1, 0] = 2 * (qj * qi + qk * qr)
    R[ 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[ 1, 2] = 2 * (qj * qk - qi * qr)
    R[ 2, 0] = 2 * (qk * qi - qj * qr)
    R[ 2, 1] = 2 * (qj * qk + qi * qr)
    R[ 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R

def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    sk = intrinsics[0, 1]

    x_lift = (
        (
            x
            - cx
            + cy * sk / fy
            - sk * y / fy
        )
        / fx
        * z
    )
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)

def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[ 4:]
        R = quat_to_rot(pose[ :4])
        p = torch.eye(4).cuda().float()
        p[:3, :3] = R
        p[:3, 3] = cam_loc
    else:  # In case of pose matrix representation
        cam_loc = pose[:3, 3]
        p = pose

    num_samples = uv.shape[0]
    depth = torch.ones((num_samples)).cuda()
    x_cam = uv[:, 0]
    y_cam = uv[:, 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(1, 0)

    world_coords = torch.matmul(p, pixel_points_cam).permute(1, 0)[:, :3]
    ray_dirs = world_coords - cam_loc
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    return ray_dirs, cam_loc

def get_warp(intrinsics, last_data, rendered_depth, curr_data):
    image_l = last_data["im"].cuda()  # 关键帧的图像
    _,height, width = image_l.shape  # 图像的尺寸

    warp_rendered_depth = rendered_depth.reshape(-1, 1, 1)

    uv = np.mgrid[0:height, 0:width].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
    uv = uv.reshape(2, -1).transpose(1, 0).cuda()

    def uv2patch(uv, patchsize):
        """
        Given the center point of a patch and patch size, return the uv coordinates of the whole patch.
        """
        if patchsize == 1:
            patch_uv = uv.clone()
            patch_uv = patch_uv.reshape(-1, uv.shape[1], patchsize, patchsize, 2)
            return patch_uv
        half = patchsize // 2
        x = torch.tensor(range(-half, half + 1)).cuda()
        y = torch.tensor(range(-half, half + 1)).cuda()
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        gridxy = torch.stack([grid_x, grid_y], -1).unsqueeze(0).unsqueeze(0)
        uv = uv.unsqueeze(2).unsqueeze(2)
        patch_uv = uv + gridxy  # torch.Size([batch_size, N_pixels, patch_size, patch_size, 2])
        return patch_uv

    uv_patch = uv2patch(uv, 1)
    uv_patch = uv_patch.reshape(-1, 2)  # 重塑为 [N, 2]

    intrinsics = intrinsics.to(image_l.device)

    # 旋转矩阵R（3x3）和平移向量t（3x1）
    R_c = curr_data["R"]
    t_c = curr_data["T"]

    R_k = last_data["R"]
    t_k = last_data["T"]

    pose_c = build_transform_matrix(R_c, t_c)
    pose_k = build_transform_matrix(R_k, t_k)

    pose_c2w=torch.linalg.inv(pose_c)
    ray_dirs_patch, cam_loc_patch = get_camera_params(uv_patch, pose_c2w, intrinsics)
    # ray_dirs_patch 形状为 [num_samples, 3] 调整为 [num_samples, 1, 3]
    ray_dirs_patch = ray_dirs_patch.reshape(-1, 1, 3)
    # unsqueeze(1).unsqueeze(1) 将其形状调整为 [1, 1, 3]
    uv_patch_points = cam_loc_patch.unsqueeze(0).unsqueeze(0) + warp_rendered_depth * ray_dirs_patch
    uv_patch_points = uv_patch_points.reshape(-1, 3).permute(1, 0)
    target_pose = pose_k.clone()
    #target_pose = torch.linalg.inv(target_pose)  # (4, 4)
    cam_cord_points = target_pose[:3, :3] @ uv_patch_points + target_pose[:3, 3:]
    target_intrinsics = intrinsics.clone()
    tmp = (target_intrinsics[:3, :3] @ cam_cord_points).permute(1, 0)
    tmp = tmp.reshape(-1, 3)
    target_uv = tmp[..., :2] / (tmp[..., 2:] + 1e-8)  # (N_pixels, 2)
    target_uv_depth = tmp[..., 2:]
    target_uv[..., 0] = target_uv[..., 0] / width
    target_uv[..., 1] = target_uv[..., 1] / height
    target_uv = target_uv * 2 - 1.0  # change range to [-1, 1]
    target_uv = target_uv.reshape(height, width, 2)
    target_uv_depth = target_uv_depth.reshape(height, width)
    sampled_rgb = F.grid_sample(
        image_l.unsqueeze(0), target_uv.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(0)
    target_sampled_rgb_mask = (
        (target_uv[..., 0] > -1)
        & (target_uv[..., 0] < 1)
        & (target_uv[..., 1] > -1)
        & (target_uv[..., 1] < 1)
        & (target_uv_depth > 0)
    )

    return sampled_rgb, target_sampled_rgb_mask

def visualize_differences(image_c, sampled_rgb, target_sampled_rgb_mask):
    # 检查图像尺寸是否一致
    assert image_c.shape == sampled_rgb.shape, "The shapes of image_c and sampled_rgb must be the same."
    # 扩展掩码的维度以匹配图像张量
    mask_expanded = target_sampled_rgb_mask.unsqueeze(0).expand_as(image_c)

    # 提取目标区域的图像
    image_c_target = image_c*mask_expanded
    image_k_target = sampled_rgb*mask_expanded

    # 计算差异图像
    difference = torch.abs(image_c - sampled_rgb) * mask_expanded

    # 将图像从 [3, H, W] 转换为 [H, W, 3] 以便于显示
    image_c_target = image_c_target.permute(1, 2, 0)
    image_k_target = image_k_target.permute(1, 2, 0)
    difference = difference.permute(1, 2, 0)
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 显示掩码
    axes[0].imshow(target_sampled_rgb_mask.cpu().numpy(), cmap='gray')
    axes[0].set_title('Target Sampled RGB Mask')
    axes[0].axis('off')

    # 显示原目标区域的图像
    axes[1].imshow(image_c_target.detach().cpu().numpy())
    axes[1].set_title('Image C Target Area')
    axes[1].axis('off')

    axes[2].imshow(image_k_target.detach().cpu().numpy())
    axes[2].set_title('Image K Target Area')
    axes[2].axis('off')

    # 显示差异图像
    axes[3].imshow(difference.detach().cpu().numpy())
    axes[3].set_title('Difference in Target Area')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

def get_warp_loss_tracking_rgb( last_data, rendered_depth, curr_data):

    sampled_rgb, target_sampled_rgb_mask=get_warp(curr_data["intrinsics"], last_data, rendered_depth, curr_data)
    image_c = curr_data["im"]  # 当前相机的图像
    #image_l = lastFrame.original_image.cuda()  # 关键帧的图像
    _, h, w = image_c.shape  # 图像的尺寸

    """
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask_c = (image_c.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask_c = rgb_pixel_mask_c * viewpoint.grad_mask

    rgb_pixel_mask_k = (image_k.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask_k = rgb_pixel_mask_k * viewpoint.grad_mask
    """

    mask_shape = (1, h, w)
    rgb_boundary_threshold = 0.01
    rgb_pixel_mask_c = (image_c.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask_k = (sampled_rgb.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    target_sampled_rgb_mask=target_sampled_rgb_mask&rgb_pixel_mask_c.squeeze().bool()&rgb_pixel_mask_k.squeeze().bool()

    l1_warp_loss=(image_c-sampled_rgb)[:,target_sampled_rgb_mask].abs().sum()
    #print(f"l1_w#arp_loss:{l1_warp_loss}")
    #if tracking_itr%50==0 :
    #visualize_differences(image_c, sampled_rgb, target_sampled_rgb_mask)
    #print(f"l1_warp_loss grad:{l1_warp_loss.requires_grad}")
    #print("Gradient function of l1_warp_loss: ", l1_warp_loss.grad_fn)
    return l1_warp_loss