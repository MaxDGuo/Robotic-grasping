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

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='cornell_dataset/08/pcd0845r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='cornell_dataset/08/pcd0845d.tiff',
                        help='Depth Image path')
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


if __name__ == '__main__':
    args = parse_args()

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

       # 打印网络原始输出
        print("\n=== 网络原始输出 ===")
        print("位置(pos)张量形状:", pred['pos'].shape)       # 例如: torch.Size([1, 1, 224, 224])
        print("角度cos分量范围:", pred['cos'].min().item(), "~", pred['cos'].max().item())
        print("角度sin分量范围:", pred['sin'].min().item(), "~", pred['sin'].max().item())
        print("宽度(width)范围:", pred['width'].min().item(), "~", pred['width'].max().item())

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        # 打印后处理结果
        print("\n=== 后处理结果 ===")
        print("抓取质量图(q_img)形状:", q_img.shape)          # 例如: (224, 224)
        print("质量分数范围:", np.min(q_img), "~", np.max(q_img))
        print("角度图(ang_img)范围(弧度):", np.min(ang_img), "~", np.max(ang_img))
        print("抓取宽度图(width_img)范围:", np.min(width_img), "~", np.max(width_img))
        
        # 打印前5个最高分抓取点
        top5_indices = np.argsort(q_img.flatten())[-5:][::-1]
        print("\nTop 5 抓取位姿:")
        for i, idx in enumerate(top5_indices):
            y, x = np.unravel_index(idx, q_img.shape)
            print(f"{i+1}. 坐标(y,x)=({y},{x}) | 角度={np.rad2deg(ang_img[y,x]):.1f}° | 宽度={width_img[y,x]:.2f} | 质量分数={q_img[y,x]:.3f}")
            
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
