import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results

import os
import random
import argparse
import math

logging.basicConfig(level=logging.INFO)

def get_random_cornell_pair(base_dir="cornell_dataset"):
    """随机选择一对 RGB 和 Depth 文件路径（从 01~09 随机选一个目录，再随机选一个文件对）"""
    # 1. 随机选择一个子目录（01~09）
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit() and 1 <= int(d) <= 9]
    if not subdirs:
        raise FileNotFoundError(f"No valid subdirectories (01-09) found in {base_dir}")

    subdir = random.choice(subdirs)  # 如 "03"
    subdir_path = os.path.join(base_dir, subdir)

    # 2. 获取所有 RGB 文件（pcd0x**r.png）
    rgb_files = [f for f in os.listdir(subdir_path) if f.endswith("r.png") and f.startswith(f"pcd{subdir}")]
    if not rgb_files:
        raise FileNotFoundError(f"No RGB files found in {subdir_path}")

    # 3. 随机选一个 RGB 文件，并生成对应的 Depth 文件名
    rgb_file = random.choice(rgb_files)  # 如 "pcd0307r.png"
    prefix = rgb_file.split("r.png")[0]  # 提取前缀，如 "pcd0307"
    depth_file = f"{prefix}d.tiff"  # 如 "pcd0307d.tiff"
    cpos_file = f"{prefix}cpos.txt"  # 如 "pcd0307d.txt"

    rgb_path = os.path.join(subdir_path, rgb_file)
    depth_path = os.path.join(subdir_path, depth_file)
    cpos_path = os.path.join(subdir_path, cpos_file) 

    return rgb_path, depth_path, cpos_path

def parse_args():

    # 随机选择默认路径
    try:
        default_rgb, default_depth, default_cpos = get_random_cornell_pair()
        print(default_rgb)
        print(default_depth)
        print(default_cpos)

    except FileNotFoundError as e:
        print(f"Warning: {e}. Using fallback paths.")
        default_rgb = "cornell_dataset/01/pcd0100r.png"
        default_depth = "cornell_dataset/01/pcd0100d.tiff"
        default_cpos = "cornell_dataset/01/pcd0100cpos.txt"

    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default=default_rgb,
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default=default_depth,
                        help='Depth Image path')
    parser.add_argument('--cpos_path', type=str, default=default_cpos,
                        help='cpos data path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def map_224_to_640x480(x_224, y_224):
    """将 224x224 图像的坐标映射到 640x480 图像（中心对齐）"""
    x_640 = 320 + x_224 - 112
    y_480 = 240 + y_224 - 112
    return int(round(x_640)), int(round(y_480))

def calculate_average_center(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 处理数据：去除空格和换行符，分割x,y坐标
    points = []
    for line in lines:
        line = line.strip()  # 去除首尾空白字符
        if line:  # 跳过空行
            x, y = map(float, line.split())
            points.append((x, y))
    
    # 将点分组为矩形（每组4个点）
    rectangles = [points[i:i+4] for i in range(0, len(points), 4)]
    
    # 计算每个矩形的中心点
    centers = []
    for rect in rectangles:

        # 计算中心点（保持不变）
        center_x = sum(p[0] for p in rect) / 4
        center_y = sum(p[1] for p in rect) / 4
        centers.append((center_x, center_y))
    
    # 计算平均中心
    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    
    center = (avg_x, avg_y)

    return center

def evaluate_network():
    args = parse_args()

    # 定义文件路径
    file_path = args.cpos_path
    average_center = calculate_average_center(file_path)

    print(f"\n平均中心坐标: ({average_center[0]:.2f}, {average_center[1]:.2f})")

    # Load image
    logging.info('Loading image...')
    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)
    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)

    # Load Network
    logging.info('Loading model...')
    # net = torch.load(args.network)  # NOTE: CHANGE TO THIS IF YOU USE OLDER VESION TORCH
    net = torch.load(args.network, weights_only=False)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        # 找到最高分的抓取点（仅1个）
        top_idx = np.argsort(q_img.flatten())[-1]  # 取排序后的最后一个（最高分）
        y, x = np.unravel_index(top_idx, q_img.shape)
        x_640, y_480 = map_224_to_640x480(x, y)
        pose  = (x_640, y_480, ang_img[y, x])
        print(pose)
        # 打印最高分抓取点
        print("\n最佳抓取位姿:")
        print(f"坐标(y,x)=({y_480},{x_640}) | 角度={np.rad2deg(ang_img[y,x]):.1f}° | 宽度={width_img[y,x]:.2f} | 质量分数={q_img[y,x]:.3f}")
            
        if args.save:
            save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img)
            fig.savefig('img_result.pdf')

    return pose, average_center

