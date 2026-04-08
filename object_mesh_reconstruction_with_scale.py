#!/usr/bin/env python3
"""
物体点云重建脚本（带尺度校正）
使用提供的mask图片和MASt3R进行物体3D点云重建，并使用传感器深度图计算尺度因子

使用方法:
    # 目录模式（推荐）：提供RGB图像目录、mask目录和深度图目录
    python object_mesh_reconstruction_with_scale.py \
        --image_dir ./rgb_images \
        --mask_dir ./mask_images \
        --depth_dir ./depth_images \
        --output object_pointcloud_scaled.ply
    
    # 如果深度图文件名有后缀，例如 img1.jpg -> img1_depth.png
    python object_mesh_reconstruction_with_scale.py \
        --image_dir ./rgb_images \
        --mask_dir ./mask_images \
        --depth_dir ./depth_images \
        --depth_suffix "_depth" \
        --output object_pointcloud_scaled.ply
    
    # 指定深度图单位（默认：米，如果深度图是毫米单位，使用 --depth_scale 0.001）
    python object_mesh_reconstruction_with_scale.py \
        --image_dir ./rgb_images \
        --mask_dir ./mask_images \
        --depth_dir ./depth_images \
        --depth_scale 0.001 \
        --output object_pointcloud_scaled.ply
    
    # 尺度因子计算逻辑：
    # 1. 生成点云后，将点云投影到各视图得到深度图
    # 2. 在mask区域内比较重建深度和传感器深度
    # 3. 每张图片计算一个尺度因子（中位数比率）
    # 4. 对所有图片的尺度因子取平均得到最终尺度

注意:
    - mask图片应该是二值图像（0表示背景，非0表示前景）
    - 深度图应该是单通道图像（uint16毫米单位或float32米单位）
    - 深度图文件名与图像文件名一一对应（支持不同扩展名）
    - 如果提供了depth_suffix，会查找 image{suffix}.ext 格式的深度图
    - 支持多种扩展名：.png, .jpg, .jpeg, .bmp (大小写不敏感)
    - 尺度因子会保存到输出目录的 scale_factor.txt 文件中
    
尺度因子计算逻辑:
    1. 生成点云后，根据算法生成的相机位姿将点云投影到2D平面
    2. 在物体mask区域内提取重建深度信息
    3. 与实际深度图（同样经过mask处理）进行比对
    4. 每张图片计算一个尺度因子（使用中位数比率）
    5. 对所有图片的尺度因子取平均得到最终尺度
"""
import argparse
import os
import cv2
import numpy as np
import trimesh
from typing import List, Optional, Tuple
from pathlib import Path

# MASt3R imports
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, prepare_intrinsics_init
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


intrinsics = np.array([[923.75616455,   0.,         652.2166748 ],
 [  0.,         923.80419922, 355.84133911],    
 [  0.,           0.,           1.        ]])


def scan_image_files(image_dir: str) -> List[str]:
    """
    扫描目录中的图像文件
    
    Args:
        image_dir: 图像目录路径
    
    Returns:
        图像文件路径列表（按文件名排序）
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise ValueError(f"图像目录不存在: {image_dir}")
    
    # 支持的图像扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir_path.glob(f"*{ext}"))
    
    # 按文件名排序
    image_files = sorted([str(f) for f in image_files])
    
    if not image_files:
        raise ValueError(f"在目录 {image_dir} 中未找到任何图像文件")
    
    return image_files


def find_mask_file(image_path: str, mask_dir: str, mask_suffix: str = "") -> Optional[str]:
    """
    根据图像路径在mask目录中查找对应的mask文件
    
    Args:
        image_path: 图像路径
        mask_dir: mask目录路径
        mask_suffix: mask文件名后缀（在扩展名之前，如果为空则使用相同文件名）
    
    Returns:
        mask路径，如果不存在则返回None
    """
    image_path_obj = Path(image_path)
    base_name = image_path_obj.stem  # 不含扩展名的文件名
    mask_dir_path = Path(mask_dir)
    
    if not mask_dir_path.exists():
        return None
    
    # 尝试多种命名规则和扩展名
    possible_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP']
    
    # 规则1: 如果提供了suffix，尝试 image.jpg -> image{suffix}.png
    if mask_suffix:
        for ext in possible_extensions:
            mask_path = mask_dir_path / f"{base_name}{mask_suffix}{ext}"
            if mask_path.exists():
                return str(mask_path)
    
    # 规则2: 使用相同文件名，不同扩展名
    for ext in possible_extensions:
        mask_path = mask_dir_path / f"{base_name}{ext}"
        if mask_path.exists():
            return str(mask_path)
    
    return None


def load_mask_images_from_dirs(image_paths: List[str], mask_dir: str, 
                                mask_suffix: str = "") -> List[np.ndarray]:
    """
    从mask目录加载mask图片，与图像文件一一对应
    
    Args:
        image_paths: 图像路径列表
        mask_dir: mask目录路径
        mask_suffix: mask文件名后缀（在扩展名之前，如果为空则使用相同文件名）
    
    Returns:
        mask列表，每个mask是numpy数组 (H, W) uint8，值为0或255
    """
    masks = []
    mask_paths = []
    
    print(f"在mask目录 {mask_dir} 中查找对应的mask文件...")
    for img_path in image_paths:
        mask_path = find_mask_file(img_path, mask_dir, mask_suffix)
        if mask_path:
            mask_paths.append(mask_path)
            img_name = Path(img_path).name
            mask_name = Path(mask_path).name
            print(f"  匹配: {img_name} -> {mask_name}")
        else:
            mask_paths.append(None)
            img_name = Path(img_path).name
            print(f"  警告: 未找到 {img_name} 对应的mask")
    
    # 加载所有找到的mask
    for i, mask_path in enumerate(mask_paths):
        if mask_path is None:
            masks.append(None)
            continue
        
        if not os.path.exists(mask_path):
            raise ValueError(f"Mask文件不存在: {mask_path}")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载mask图片: {mask_path}")
        
        # 转换为二值mask（0或255）
        # 如果已经是二值的，保持原样；否则将非0值设为255
        mask_binary = (mask > 0).astype(np.uint8) * 255
        masks.append(mask_binary)
    
    return masks

def find_depth_file(image_path: str, depth_dir: str, depth_suffix: str = "") -> Optional[str]:
    """
    根据图像路径在深度图目录中查找对应的深度图文件
    
    Args:
        image_path: 图像路径
        depth_dir: 深度图目录路径
        depth_suffix: 深度图文件名后缀（在扩展名之前，如果为空则使用相同文件名）
    
    Returns:
        深度图路径，如果不存在则返回None
    """
    image_path_obj = Path(image_path)
    base_name = image_path_obj.stem  # 不含扩展名的文件名
    depth_dir_path = Path(depth_dir)
    
    if not depth_dir_path.exists():
        return None
    
    # 尝试多种命名规则和扩展名
    possible_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP', '.exr', '.EXR', '.npy', '.NPY']
    
    # 规则1: 如果提供了suffix，尝试 image.jpg -> image{suffix}.png
    if depth_suffix:
        for ext in possible_extensions:
            depth_path = depth_dir_path / f"{base_name}{depth_suffix}{ext}"
            if depth_path.exists():
                return str(depth_path)
    
    # 规则2: 使用相同文件名，不同扩展名
    for ext in possible_extensions:
        depth_path = depth_dir_path / f"{base_name}{ext}"
        if depth_path.exists():
            return str(depth_path)
    
    return None


def load_depth_images_from_dirs(image_paths: List[str], depth_dir: str, 
                                 depth_suffix: str = "", depth_scale: float = 1.0) -> List[Optional[np.ndarray]]:
    """
    从深度图目录加载深度图，与图像文件一一对应
    
    Args:
        image_paths: 图像路径列表
        depth_dir: 深度图目录路径
        depth_suffix: 深度图文件名后缀（在扩展名之前，如果为空则使用相同文件名）
        depth_scale: 深度图缩放因子（例如，如果深度图是毫米单位，使用0.001转换为米）
    
    Returns:
        深度图列表，每个深度图是numpy数组 (H, W) float32，单位为米
    """
    depths = []
    depth_paths = []
    
    print(f"在深度图目录 {depth_dir} 中查找对应的深度图文件...")
    for img_path in image_paths:
        depth_path = find_depth_file(img_path, depth_dir, depth_suffix)
        if depth_path:
            depth_paths.append(depth_path)
            img_name = Path(img_path).name
            depth_name = Path(depth_path).name
            print(f"  匹配: {img_name} -> {depth_name}")
        else:
            depth_paths.append(None)
            img_name = Path(img_path).name
            print(f"  警告: 未找到 {img_name} 对应的深度图")
    
    # 加载所有找到的深度图
    for i, depth_path in enumerate(depth_paths):
        if depth_path is None:
            depths.append(None)
            continue
        
        if not os.path.exists(depth_path):
            raise ValueError(f"深度图文件不存在: {depth_path}")
        
        # 加载深度图（支持图像格式和 .npy 格式）
        if depth_path.lower().endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
        if depth is None:
            raise ValueError(f"无法加载深度图: {depth_path}")
        
        # 如果是多通道，只取第一个通道
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        # 转换为float32并应用缩放因子
        depth_float = depth.astype(np.float32) * depth_scale
        
        # 过滤无效深度值（0或负数）
        depth_float[depth_float <= 0] = np.nan
        
        depths.append(depth_float)
    
    return depths


def project_points_to_uv(pointcloud: np.ndarray, K: np.ndarray, 
                         c2w: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将点云投影到图像平面，得到每个点对应的(u, v)坐标和深度
    
    Args:
        pointcloud: 点云 (N, 3)，世界坐标系
        K: 相机内参矩阵 (3, 3)
        c2w: 相机到世界坐标系的变换矩阵 (4, 4)
        img_shape: 图像尺寸 (H, W)
    
    Returns:
        (uv_coords, depths_cam)
        - uv_coords: (N, 2) 每个点对应的(u, v)像素坐标
        - depths_cam: (N,) 每个点在相机坐标系下的深度（Z值）
    """
    H, W = img_shape
    
    # 将世界坐标转换为相机坐标
    w2c = np.linalg.inv(c2w)
    pts3d_homo = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=1)
    pts3d_cam = (w2c @ pts3d_homo.T).T[:, :3]  # (N, 3)
    
    # 提取深度（相机坐标系下的Z值）
    depths_cam = pts3d_cam[:, 2]
    
    # 投影到图像平面
    pts2d_homo = (K @ pts3d_cam.T).T  # (N, 3)
    uv_coords = pts2d_homo[:, :2] / (pts2d_homo[:, 2:3] + 1e-8)  # (N, 2)
    
    return uv_coords, depths_cam


def filter_points_by_mask_via_projection(pointcloud: np.ndarray, mask: np.ndarray,
                                         K: np.ndarray, c2w: np.ndarray, 
                                         img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过投影将点云投影到图像平面，根据mask过滤点云
    
    Args:
        pointcloud: 点云 (N, 3)，世界坐标系
        mask: mask图像 (H, W)，bool类型
        K: 相机内参矩阵 (3, 3)
        c2w: 相机到世界坐标系的变换矩阵 (4, 4)
        img_shape: 图像尺寸 (H, W)
    
    Returns:
        (filtered_points, uv_coords)
        - filtered_points: 过滤后的点云 (M, 3)，M <= N
        - uv_coords: 过滤后点云对应的(u, v)坐标 (M, 2)
    """
    H, W = img_shape
    
    # 投影得到(u, v)坐标（只投影一次）
    uv_coords, depths_cam = project_points_to_uv(pointcloud, K, c2w, img_shape)
    
    # 调整mask尺寸
    if mask.shape != (H, W):
        mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), 
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask_resized = mask.astype(bool) if mask.dtype != bool else mask
    
    # 判断哪些点在mask内且深度有效 (向量化优化)
    u = np.round(uv_coords[:, 0]).astype(int)
    v = np.round(uv_coords[:, 1]).astype(int)
    
    # 创建掩码：在图像范围内、深度为正且有限、且在 mask 区域内
    in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_mask = np.zeros(len(pointcloud), dtype=bool)
    
    if np.any(in_view):
        # 只对在视野内的点进行 mask 检查
        valid_mask[in_view] = mask_resized[v[in_view], u[in_view]] & (depths_cam[in_view] > 0) & np.isfinite(depths_cam[in_view])
    
    # 返回过滤后的点云和对应的uv坐标
    return pointcloud[valid_mask], uv_coords[valid_mask]


def project_pointcloud_to_depth(pointcloud: np.ndarray, K: np.ndarray, 
                                 c2w: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    将点云投影到图像平面，生成深度图
    
    Args:
        pointcloud: 点云 (N, 3)，世界坐标系
        K: 相机内参矩阵 (3, 3)
        c2w: 相机到世界坐标系的变换矩阵 (4, 4)
        img_shape: 图像尺寸 (H, W)
    
    Returns:
        深度图 (H, W)，每个像素值是该像素对应的3D点的深度（相机坐标系下的Z值）
    """
    H, W = img_shape
    
    # 使用投影函数获取(u, v)坐标和深度
    uv_coords, depths_cam = project_points_to_uv(pointcloud, K, c2w, img_shape)
    
    # 创建深度图
    depth_map = np.full((H, W), np.nan, dtype=np.float32)
    
    # 向量化处理：将点投影到图像平面
    u = np.round(uv_coords[:, 0]).astype(int)
    v = np.round(uv_coords[:, 1]).astype(int)
    
    # 过滤无效点
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths_cam > 0) & np.isfinite(depths_cam)
    if not np.any(valid_mask):
        return depth_map
        
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    d_valid = depths_cam[valid_mask]
    
    # 处理遮挡：按照深度从大到小排序，这样后面的近点会覆盖远点
    sort_idx = np.argsort(-d_valid)
    depth_map[v_valid[sort_idx], u_valid[sort_idx]] = d_valid[sort_idx]
    
    return depth_map


def compute_scale_factor_per_image(recon_depth: np.ndarray, 
                                   sensor_depth: np.ndarray,
                                   mask: np.ndarray,
                                   min_depth: float = 0.1) -> Optional[float]:
    """
    计算单张图片的尺度因子
    
    Args:
        recon_depth: 重建的深度图 (H, W) float32
        sensor_depth: 传感器深度图 (H, W) float32
        mask: mask，用于只考虑物体区域 (H, W) bool
        min_depth: 最小深度阈值（米）
    
    Returns:
        尺度因子，如果无法计算则返回None
    """
    # 创建有效深度mask（只考虑mask区域内的点）
    valid_mask = (
        np.isfinite(recon_depth) & 
        np.isfinite(sensor_depth) & 
        (recon_depth > 0) & 
        (sensor_depth > min_depth) &
        mask
    )
    
    if np.sum(valid_mask) == 0:
        return None
    
    # 提取有效深度值
    recon_valid = recon_depth[valid_mask]
    sensor_valid = sensor_depth[valid_mask]
    
    # 计算比率：sensor_depth / reconstructed_depth
    ratios = sensor_valid / (recon_valid + 1e-8)
    
    # 过滤异常值（使用IQR方法）
    q1, q3 = np.percentile(ratios, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        inlier_mask = (ratios >= lower_bound) & (ratios <= upper_bound)
        
        if np.sum(inlier_mask) > 0:
            ratios_clean = ratios[inlier_mask]
            # 使用中位数作为该图片的尺度因子
            scale_factor = float(np.median(ratios_clean))
            return scale_factor
    
    # 如果没有足够的inlier，使用所有有效点的中位数
    scale_factor = float(np.median(ratios))
    return scale_factor


def compute_scale_factor_from_pointcloud(pointcloud: np.ndarray,
                                         sensor_depths: List[Optional[np.ndarray]],
                                         masks: List[np.ndarray],
                                         cams2world: np.ndarray,
                                         Ks: List[np.ndarray],
                                         img_shapes: List[Tuple[int, int]],
                                         output_path: str = "object_pointcloud_scaled.ply") -> float:
    """
    从点云计算尺度因子：将点云投影到每个视图，计算每张图片的尺度因子，然后取平均
    
    Args:
        pointcloud: 点云 (N, 3)，世界坐标系
        sensor_depths: 传感器深度图列表，每个是 (H, W) float32
        masks: mask列表，用于只考虑物体区域
        cams2world: 相机到世界坐标系的变换矩阵列表，每个是 (4, 4)
        Ks: 每个视图对应的相机内参矩阵列表 (3, 3)
        img_shapes: 图像尺寸列表，每个是 (H, W)
        output_path: 输出路径，用于保存调试图
    
    Returns:
        平均尺度因子
    """
    scale_factors = []
    
    print("将点云投影到各视图并计算尺度因子...")
    for i, (sensor_depth, mask, c2w, K, img_shape) in enumerate(zip(sensor_depths, masks, cams2world, Ks, img_shapes)):
        if sensor_depth is None:
            print(f"  视图 {i+1}: 跳过（无传感器深度图）")
            continue
        
        # 将点云投影到该视图
        recon_depth = project_pointcloud_to_depth(pointcloud, K, c2w, img_shape)
        
        # 调整mask和传感器深度图尺寸以匹配投影深度图
        H, W = recon_depth.shape
        if sensor_depth.shape != (H, W):
            sensor_depth_resized = cv2.resize(sensor_depth, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            sensor_depth_resized = sensor_depth
        
        if mask.shape != (H, W):
            mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_resized = mask.astype(bool) if mask.dtype != bool else mask
        
        # 计算该视图的尺度因子
        scale_factor_i = compute_scale_factor_per_image(recon_depth, sensor_depth_resized, mask_resized)
        
        # --- 调试可视化逻辑 ---
        debug_dir = os.path.join(os.path.dirname(output_path), "debug_projection")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 归一化深度图用于显示
        def normalize_depth(d):
            d_valid = d[np.isfinite(d) & (d > 0)]
            if d_valid.size == 0: return np.zeros_like(d, dtype=np.uint8)
            d_min, d_max = np.percentile(d_valid, [1, 99])
            d_norm = np.clip((d - d_min) / (d_max - d_min + 1e-8), 0, 1)
            d_norm[~np.isfinite(d)] = 0
            return (d_norm * 255).astype(np.uint8)

        recon_viz = cv2.applyColorMap(normalize_depth(recon_depth), cv2.COLORMAP_JET)
        sensor_viz = cv2.applyColorMap(normalize_depth(sensor_depth_resized), cv2.COLORMAP_JET)
        mask_viz = (mask_resized.astype(np.uint8) * 255)
        mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
        
        # 在重建深度图上叠加 mask 轮廓
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(recon_viz, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(sensor_viz, contours, -1, (0, 255, 0), 2)
        
        # 拼接显示：重建深度 | 传感器深度 | Mask
        debug_tile = np.hstack([recon_viz, sensor_viz, mask_viz])
        debug_file = os.path.join(debug_dir, f"view_{i+1:03d}_debug.jpg")
        cv2.imwrite(debug_file, debug_tile)
        # ---------------------
        
        if scale_factor_i is not None:
            scale_factors.append(scale_factor_i)
            print(f"  视图 {i+1}: 尺度因子 = {scale_factor_i:.6f}")
        else:
            print(f"  视图 {i+1}: 无法计算尺度因子（无有效对应点）")
    
    if len(scale_factors) == 0:
        raise ValueError("无法计算尺度因子：没有有效的深度对应关系")
    
    # 计算平均尺度因子
    avg_scale_factor = np.mean(scale_factors)
    
    print(f"\n尺度因子统计:")
    print(f"  有效视图数量: {len(scale_factors)}")
    print(f"  各视图尺度因子: {[f'{s:.6f}' for s in scale_factors]}")
    print(f"  平均尺度因子: {avg_scale_factor:.6f}")
    print(f"  尺度因子范围: [{np.min(scale_factors):.6f}, {np.max(scale_factors):.6f}]")
    print(f"  尺度因子标准差: {np.std(scale_factors):.6f}")
    
    return float(avg_scale_factor)


def reconstruct_object_pointcloud_with_scale(
    image_paths: List[str],
    masks: List[np.ndarray],
    sensor_depths: List[Optional[np.ndarray]],
    model: AsymmetricMASt3R,
    device: str = "cuda",
    image_size: int = 512,
    min_conf_thr: float = 1.5,
    TSDF_thresh: float = 0.0,
    clean_depth: bool = True,
    output_path: str = "object_pointcloud_scaled.ply",
    scenegraph_type: str = "complete",
    optim_level: str = "refine+depth",
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,
    niter2: int = 300,
) -> Tuple[str, float]:
    """
    使用MASt3R重建物体的点云，并使用传感器深度图计算尺度因子
    
    Args:
        image_paths: 图像路径列表
        masks: 物体mask列表，每个mask是numpy数组 (H, W) uint8
        sensor_depths: 传感器深度图列表，每个深度图是numpy数组 (H, W) float32（米）
        model: MASt3R模型
        device: 设备
        image_size: 图像尺寸
        min_conf_thr: 最小置信度阈值
        TSDF_thresh: TSDF阈值
        clean_depth: 是否清理深度图
        output_path: 输出点云路径
        scenegraph_type: 场景图类型
        optim_level: 优化级别
        lr1, niter1: 粗优化学习率和迭代次数
        lr2, niter2: 精细优化学习率和迭代次数
    
    Returns:
        (输出点云文件路径, 尺度因子)
    
    注意:
        尺度因子通过将点云投影到各视图，计算每张图片的尺度因子，然后取平均得到
    """
    print(f"Loading {len(image_paths)} images...")
    imgs = load_images(image_paths, size=image_size, verbose=True)
    
    if len(imgs) == 1:
        # 单张图像需要复制一份
        imgs = [imgs[0], imgs[0].copy()]
        imgs[1]['idx'] = 1
        if masks and len(masks) > 0:
            masks = masks + [masks[0].copy()]
        else:
            masks = [None, None]
        if sensor_depths and len(sensor_depths) > 0:
            sensor_depths = sensor_depths + [sensor_depths[0].copy() if sensor_depths[0] is not None else None]
        else:
            sensor_depths = [None, None]
    
    # 调整mask尺寸以匹配加载的图像尺寸
    adjusted_masks = []
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        if mask is not None:
            H, W = img['true_shape'][0]
            if mask.shape != (H, W):
                mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask.astype(np.uint8)
            adjusted_masks.append(mask_resized)
        else:
            # 如果没有mask，创建全1的mask
            H, W = img['true_shape'][0]
            adjusted_masks.append(np.ones((H, W), dtype=np.uint8))
    
    # 创建图像对
    print("Creating image pairs...")
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append("2")  # 默认窗口大小
    scene_graph = '-'.join(scene_graph_params)
    
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    
    # 运行稀疏全局对齐
    print("Running sparse global alignment...")
    cache_dir = os.path.join(os.path.dirname(output_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    if optim_level == 'coarse':
        niter2 = 0

    # 单张图像，使用resize前的内参矩阵
    K_orig = intrinsics
    
    # 回退到与 object_mesh_reconstruction.py 一致的硬编码尺寸，以对齐优化效果
    init = prepare_intrinsics_init(K_orig, image_paths[:len(imgs)], 
                                    original_sizes=[(1281, 721)] * len(imgs),
                                    resized_sizes=[(512, 288)] * len(imgs))
    
    scene = sparse_global_alignment(
        image_paths[:len(imgs)],
        pairs,
        cache_dir,
        model,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        device=device,
        opt_depth='depth' in optim_level,
        shared_intrinsics=True,
        matching_conf_thr=0.0,
        init=init
    )
    
    # 获取密集点云
    print("Getting dense point cloud...")
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, depthmaps, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    
    # 应用mask和置信度阈值
    print("Applying masks and confidence threshold...")
    rgbimgs = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    print(f' >> Focals: {focals}')
    
    # 创建组合mask：物体mask AND 置信度mask
    combined_masks = []
    for i, (mask, conf) in enumerate(zip(adjusted_masks, confs)):
        conf_mask = conf > min_conf_thr
        # 调整mask尺寸以匹配conf的形状
        if mask.shape != conf_mask.shape:
            mask_resized = cv2.resize(mask.astype(np.uint8), (conf_mask.shape[1], conf_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_resized = mask_resized > 0
        else:
            mask_resized = mask > 0
        combined_mask = mask_resized & conf_mask
        combined_masks.append(combined_mask)
    
    # 生成点云（采用与 object_mesh_reconstruction.py 一致的 Reshape 过滤逻辑，确保点云完整性）
    print("\n生成点云...")
    all_points_raw = []
    all_colors = []

    for i in range(len(imgs)):
        H_img, W_img = rgbimgs[i].shape[:2]
        H_conf, W_conf = confs[i].shape
        pts3d_len = pts3d[i].shape[0]
        
        # 确定点云形状并 reshape
        if pts3d_len == H_conf * W_conf:
            pts3d_i = pts3d[i].reshape(H_conf, W_conf, 3)
            target_H, target_W = H_conf, W_conf
        elif pts3d_len == H_img * W_img:
            pts3d_i = pts3d[i].reshape(H_img, W_img, 3)
            target_H, target_W = H_img, W_img
        else:
            side = int(np.sqrt(pts3d_len))
            if side * side == pts3d_len:
                pts3d_i = pts3d[i].reshape(side, side, 3)
                target_H, target_W = side, side
            else:
                pts3d_i = pts3d[i][:H_conf * W_conf].reshape(H_conf, W_conf, 3)
                target_H, target_W = H_conf, W_conf
        
        # 调整 combined_masks[i] 到点云尺寸
        if combined_masks[i].shape != (target_H, target_W):
            msk_i = cv2.resize(combined_masks[i].astype(np.uint8), (target_W, target_H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            msk_i = combined_masks[i]
        
        # 应用有限性检查
        msk_i = msk_i & np.isfinite(pts3d_i.sum(axis=-1))
        
        # 调整图像尺寸以匹配点云
        if rgbimgs[i].shape[:2] != (target_H, target_W):
            rgb_img_resized = cv2.resize(rgbimgs[i], (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_img_resized = rgbimgs[i]

        # 收集点云
        if np.any(msk_i):
            all_points_raw.append(pts3d_i[msk_i].reshape(-1, 3))
            all_colors.append(rgb_img_resized[msk_i].reshape(-1, 3))

    if len(all_points_raw) == 0:
        raise ValueError("No valid point cloud generated")
    
    # 合并所有点云
    pts_cat_raw = np.concatenate(all_points_raw, axis=0)
    
    # 计算尺度因子（如果有传感器深度图）
    scale_factor = 1.0
    if sensor_depths and any(d is not None for d in sensor_depths):
        print("\n计算尺度因子...")
        # 准备数据：获取每个视图的mask和图像尺寸
        valid_masks_for_scale = []
        img_shapes_for_scale = []
        cams2world_list = []
        Ks_optimized = []  # 保存每个视图优化后的内参
        
        # 从场景中获取优化后的内参 (针对推理尺寸，如 512x288)
        # SparseGA 对象中内参存储在 scene.intrinsics 中
        K_all_opt = scene.intrinsics.cpu().numpy()
        
        for i in range(len(imgs)):
            # 注意：投影必须在推理尺寸下进行，因为位姿是相对于该尺寸优化的
            H_img, W_img = rgbimgs[i].shape[:2]
            
            img_shapes_for_scale.append((H_img, W_img))
            cams2world_list.append(cams2world[i].numpy())
            Ks_optimized.append(K_all_opt[i])
            
            # 调整 mask 到推理尺寸以匹配投影深度图
            if combined_masks[i].shape != (H_img, W_img):
                mask_for_scale = cv2.resize(combined_masks[i].astype(np.uint8), (W_img, H_img), 
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask_for_scale = combined_masks[i]
            valid_masks_for_scale.append(mask_for_scale)
        
        # 计算尺度因子
        try:
            scale_factor = compute_scale_factor_from_pointcloud(
                pts_cat_raw,
                sensor_depths,
                valid_masks_for_scale,
                cams2world_list,
                Ks_optimized,  # 传入优化后的内参列表
                img_shapes_for_scale,
                output_path=output_path
            )
        except Exception as e:
            print(f"警告: 无法计算尺度因子: {e}")
            print("使用默认尺度因子 1.0")
            scale_factor = 1.0
    else:
        print("未提供传感器深度图，跳过尺度因子计算")
    
    # 应用尺度因子到点云
    print(f"\n应用尺度因子 {scale_factor:.6f} 到点云...")
    pts_cat = pts_cat_raw * scale_factor
    
    # 保存点云
    if len(pts_cat) > 0:
        print("Saving point cloud...")
        cols_cat = np.concatenate(all_colors, axis=0)

        # 将颜色转换为uint8 [0, 255]
        if cols_cat.dtype != np.uint8:
            max_val = cols_cat.max() if cols_cat.size > 0 else 1.0
            if max_val <= 1.0 + 1e-3:
                cols_uint8 = np.clip(cols_cat * 255.0, 0, 255).astype(np.uint8)
            else:
                cols_uint8 = np.clip(cols_cat, 0, 255).astype(np.uint8)
        else:
            cols_uint8 = cols_cat.astype(np.uint8)

        # 确保输出路径是.ply格式
        if not output_path.endswith('.ply'):
            output_path = os.path.splitext(output_path)[0] + '.ply'
        
        print(f"Saving point cloud to {output_path}...")
        pct = trimesh.PointCloud(pts_cat, colors=cols_uint8)
        pct.export(output_path)
        print(f"Point cloud saved successfully! ({len(pts_cat)} points)")
        
        # 保存尺度因子
        scale_file = os.path.splitext(output_path)[0] + "_scale_factor.txt"
        with open(scale_file, 'w') as f:
            f.write(f"Scale Factor: {scale_factor:.10f}\n")
            f.write(f"Method: average of per-image scales\n")
        print(f"Scale factor saved to {scale_file}")
        
        return output_path, scale_factor
    else:
        raise ValueError("No valid point cloud generated")


def main():
    parser = argparse.ArgumentParser(description="物体点云重建（带尺度校正）：使用mask图片、传感器深度图和MASt3R")
    
    # 输入参数 - 目录模式
    parser.add_argument("--image_dir", type=str, required=True,
                       help="RGB图像目录路径")
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="mask图片目录路径")
    parser.add_argument("--depth_dir", type=str, required=True,
                       help="传感器深度图目录路径")
    parser.add_argument("--mask_suffix", type=str, default="",
                       help="mask文件名后缀（在扩展名之前，如果为空则使用与图像相同的文件名）")
    parser.add_argument("--depth_suffix", type=str, default="",
                       help="深度图文件名后缀（在扩展名之前，如果为空则使用与图像相同的文件名）")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="深度图缩放因子（例如，如果深度图是毫米单位，使用0.001转换为米）")
    
    # 输入参数 - 文件列表模式（向后兼容）
    parser.add_argument("--images", type=str, nargs="+", default=None,
                       help="输入图像路径列表（如果提供，将使用文件列表模式而不是目录模式）")
    parser.add_argument("--masks", type=str, nargs="+", default=None,
                       help="mask图片路径列表（仅在文件列表模式下使用）")
    parser.add_argument("--depths", type=str, nargs="+", default=None,
                       help="深度图路径列表（仅在文件列表模式下使用）")
    
    # MASt3R参数
    parser.add_argument("--model_name", type=str,
                       default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                       help="MASt3R模型名称")
    parser.add_argument("--weights", type=str, default="/home/turingzero/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                       help="模型权重路径（如果提供，将覆盖model_name）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备：cuda或cpu")
    parser.add_argument("--image_size", type=int, default=512,
                       help="图像尺寸")
    
    # 重建参数
    parser.add_argument("--min_conf_thr", type=float, default=1.5,
                       help="最小置信度阈值")
    parser.add_argument("--TSDF_thresh", type=float, default=0.01,
                       help="TSDF阈值（0表示不使用TSDF）")
    parser.add_argument("--clean_depth", action="store_true", default=True,
                       help="清理深度图")
    parser.add_argument("--scenegraph_type", type=str, default="complete",
                       choices=["complete", "swin", "logwin", "oneref"],
                       help="场景图类型")
    parser.add_argument("--optim_level", type=str, default="refine+depth",
                       choices=["coarse", "refine", "refine+depth"],
                       help="优化级别")
    parser.add_argument("--lr1", type=float, default=0.07,
                       help="粗优化学习率")
    parser.add_argument("--niter1", type=int, default=300,
                       help="粗优化迭代次数")
    parser.add_argument("--lr2", type=float, default=0.01,
                       help="精细优化学习率")
    parser.add_argument("--niter2", type=int, default=300,
                       help="精细优化迭代次数")
    
    # 尺度计算参数（已废弃，现在总是使用平均）
    # parser.add_argument("--scale_method", type=str, default="median",
    #                    choices=["median", "mean"],
    #                    help="尺度计算方法：median（中位数）或mean（均值）")
    
    # 输出参数
    parser.add_argument("--output", type=str, default="object_pointcloud_scaled.ply",
                       help="输出点云文件路径（PLY格式）")
    parser.add_argument("--visualize", action="store_true",
                       help="生成点云后自动可视化")
    parser.add_argument("--visualize_backend", type=str, default="trimesh",
                       choices=["trimesh", "open3d"],
                       help="可视化后端：trimesh 或 open3d")
    
    args = parser.parse_args()
    
    # 确定使用哪种模式：目录模式或文件列表模式
    if args.images is not None:
        # 文件列表模式（向后兼容）
        print("使用文件列表模式...")
        image_paths = args.images
        if not image_paths:
            raise ValueError("至少需要提供一张图像")
        
        # 加载mask图片
        print("Loading mask images...")
        if args.masks:
            masks = []
            for mask_path in args.masks:
                if mask_path and os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks.append((mask > 0).astype(np.uint8) * 255)
                    else:
                        masks.append(None)
                else:
                    masks.append(None)
            while len(masks) < len(image_paths):
                masks.append(masks[-1] if masks else None)
        elif args.mask_dir:
            masks = load_mask_images_from_dirs(image_paths, args.mask_dir, args.mask_suffix)
        else:
            masks = [None] * len(image_paths)
        
        # 加载深度图
        print("Loading depth images...")
        if args.depths:
            sensor_depths = []
            for depth_path in args.depths:
                if depth_path and os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if depth is not None:
                        if len(depth.shape) == 3:
                            depth = depth[:, :, 0]
                        if depth.dtype == np.uint16:
                            depth_float = depth.astype(np.float32) * args.depth_scale
                        else:
                            depth_float = depth.astype(np.float32) * args.depth_scale
                        depth_float[depth_float <= 0] = np.nan
                        sensor_depths.append(depth_float)
                    else:
                        sensor_depths.append(None)
                else:
                    sensor_depths.append(None)
            while len(sensor_depths) < len(image_paths):
                sensor_depths.append(sensor_depths[-1] if sensor_depths else None)
        elif args.depth_dir:
            sensor_depths = load_depth_images_from_dirs(image_paths, args.depth_dir, args.depth_suffix, args.depth_scale)
        else:
            sensor_depths = [None] * len(image_paths)
    else:
        # 目录模式
        if not args.image_dir:
            raise ValueError("必须提供 --image_dir 参数")
        if not args.mask_dir:
            raise ValueError("必须提供 --mask_dir 参数")
        if not args.depth_dir:
            raise ValueError("必须提供 --depth_dir 参数")
        
        print(f"扫描图像目录: {args.image_dir}")
        image_paths = scan_image_files(args.image_dir)
        print(f"找到 {len(image_paths)} 张图像")
        
        # 在mask目录中查找对应的mask文件
        masks = load_mask_images_from_dirs(image_paths, args.mask_dir, args.mask_suffix)
        
        # 在深度图目录中查找对应的深度图文件
        sensor_depths = load_depth_images_from_dirs(image_paths, args.depth_dir, args.depth_suffix, args.depth_scale)
    
    num_loaded_masks = len([m for m in masks if m is not None])
    num_loaded_depths = len([d for d in sensor_depths if d is not None])
    print(f"成功加载 {num_loaded_masks} 个mask，{num_loaded_depths} 个深度图，共 {len(image_paths)} 张图像")
    
    if num_loaded_masks == 0:
        raise ValueError("未找到任何mask图片，请检查文件命名或目录路径")
    
    if num_loaded_depths == 0:
        print("警告: 未找到任何深度图，将使用默认尺度因子 1.0")
    
    # 加载MASt3R模型
    print("Loading MASt3R model...")
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    model.eval()
    
    # 重建物体点云（带尺度校正）
    print("Reconstructing object point cloud with scale correction...")
    output_path, scale_factor = reconstruct_object_pointcloud_with_scale(
        image_paths=image_paths,
        masks=masks,
        sensor_depths=sensor_depths,
        model=model,
        device=args.device,
        image_size=args.image_size,
        min_conf_thr=args.min_conf_thr,
        TSDF_thresh=args.TSDF_thresh,
        clean_depth=args.clean_depth,
        output_path=args.output,
        scenegraph_type=args.scenegraph_type,
        optim_level=args.optim_level,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
    )
    
    print(f"\n完成！物体点云已保存到: {output_path}")
    print(f"尺度因子: {scale_factor:.10f}")
    
    # 可视化
    # if args.visualize:
    #     print("\n开始可视化...")
    #     visualize_pointcloud(output_path, backend=args.visualize_backend)


if __name__ == "__main__":
    main()

