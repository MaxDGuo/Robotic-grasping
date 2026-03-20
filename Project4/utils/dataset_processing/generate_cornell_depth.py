import argparse
import glob
import os
import re
import numpy as np
from imageio import imsave

from utils.dataset_processing.image import DepthImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    parser.add_argument('path', type=str, help='Path to Cornell Grasping Dataset')
    args = parser.parse_args()

    all_pcds = glob.glob(os.path.join(args.path, '*', 'pcd*.txt'))
    pcds = []

    for f in all_pcds:
        basename = os.path.basename(f)
        match = re.match(r'pcd(\d{4})\.txt', basename)  # 只匹配4位数字
        if match:
            num = int(match.group(1))  # 比如 "0100" -> 100
            if 100 <= num <= 1034:
                pcds.append(f)

    pcds.sort()
    
    for pcd in pcds:
        di = DepthImage.from_pcd(pcd, (480, 640))
        di.inpaint()

        of_name = pcd.replace('.txt', 'd.tiff')
        print(of_name)
        imsave(of_name, di.img.astype(np.float32))
