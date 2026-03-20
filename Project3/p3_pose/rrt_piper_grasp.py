# 导入 cv2 库
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
from scipy.spatial.transform import Rotation

# 从 XML 文件加载 MuJoCo 物理引擎模型
model = mujoco.Physics.from_xml_path('assets/cracker_box.xml')

# 从 URDF 文件中加载机械臂的运动学链，用于后续的运动学分析
my_chain = ikpy.chain.Chain.from_urdf_file("assets/piper_right.urdf")

class RRT:
    class Node:
        def __init__(self, q):
            self.q = q
            self.path_q = []
            self.parent = None

    def __init__(self, start, goal, joint_limits, expand_dis=0.1, path_resolution=0.01, goal_sample_rate=5, max_iter=1000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, model):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, model):
                self.node_list.append(new_node)
            
            if self.calc_dist_to_goal(self.node_list[-1].q) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, model):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def get_nearest_node_index(self, node_list, rnd_node):
        """
        查找与随机节点距离最近的节点的索引。
        参数说明：
            node_list: RRT 树中的节点列表。
            rnd_node: 随机生成的节点。
        返回值：
            返回节点列表中距离随机节点最近的节点的索引。
        """
        # 计算每个节点与随机节点之间的欧氏距离
        dlist = [np.linalg.norm(np.array(node.q) - np.array(rnd_node.q[:6])) for node in node_list]

        # 找到距离最小的节点索引
        min_index = dlist.index(min(dlist))
        return min_index
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(np.array(from_node.q))
        distance = np.linalg.norm(np.array(to_node.q[:6]) - np.array(from_node.q))
        if extend_length > distance:
            extend_length = distance
        num_steps = int(extend_length / self.path_resolution)
        delta_q = (np.array(to_node.q[:6]) - np.array(from_node.q)) / distance

        for i in range(num_steps):
            new_q = new_node.q + delta_q * self.path_resolution
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            new_node.path_q.append(new_node.q)

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
        else:
            rand_q = self.end.q
        return self.Node(rand_q)

    def check_collision(self, node, model):
        return check_collision_with_dm_control(model, node.q)

    def generate_final_course(self, goal_ind):
        path = [self.end.q]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q)
            node = node.parent
        path.append(self.start.q)
        return path[::-1]
    
    def calc_dist_to_goal(self, q):
        return np.linalg.norm(np.array(self.end.q[:6]) - np.array(q))

def get_depth(sim):
    # depth 是一个浮动数组，单位为米，表示每个像素点的深度信息
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # 将深度图中的最小值移至原点
    depth -= depth.min()
    # 根据接近光线的平均距离进行缩放，缩放因子为 2 倍的近距离光线的均值
    depth /= 2*depth[depth <= 1].mean()
   # 将深度值缩放至 [0, 255] 范围，用于图像显示
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def check_collision_with_dm_control(model, joint_config):
    """
    用于检查给定的关节配置是否会导致碰撞，使用 dm_control 的碰撞检测功能。
    参数说明：
        model: 传入的 dm_control Mujoco 模型，包含了物理仿真和碰撞检测的数据。
        joint_config: 要检查碰撞的关节角度列表，表示机器人的关节配置。
    返回值：
        如果没有碰撞，返回 True；如果存在碰撞，返回 False。
    """
    model.data.qpos[0:6] = joint_config  # 设置关节的位置（即角度）
    model.forward()  # 更新仿真状态，计算新的物理状态

    # 检查是否有碰撞
    contacts = model.data.ncon  #获取当前的接触数（即碰撞次数）
    # 如果没有接触（即碰撞自由），返回 True；否则，检查是否是夹爪发生的碰撞
    return contacts == 0 or check_gripper_collision(model) # 如果没有接触（无碰撞），返回 True

def check_gripper_collision(model):
    all_contact_pairs = []
    for i_contact in range(model.data.ncon):
        id_geom_1 = model.data.contact[i_contact].geom1
        id_geom_2 = model.data.contact[i_contact].geom2
        name_geom_1 = model.model.id2name(id_geom_1, 'geom')
        name_geom_2 = model.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    touch_banana_right = ("piper_gripper_finger_touch_right", "cracker_box") in all_contact_pairs
    touch_banana_left = ("piper_gripper_finger_touch_left", "cracker_box") in all_contact_pairs
    return touch_banana_left or touch_banana_right

def get_end_effector_pose(physics, body_name="link6"):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")
    
    # 位置 (x,y,z)
    pos = physics.named.data.xpos[body_name]  # 使用named接口
    
    # 旋转矩阵 (3x3)
    rot = physics.named.data.xmat[body_name].reshape(3, 3)
    
    return pos.copy(), rot.copy()  # 返回拷贝避免后续修改
    
def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_1.mp4"):
    """
    将通过 RRT 规划生成的路径（关节配置列表）应用到 dm_control 仿真中，并录制视频。
    参数说明：
        model: 传入的 dm_control Mujoco 模型，包含物理仿真数据和控制接口。
        path: 由 RRT 规划器生成的关节配置列表，表示机器人从起点到终点的路径。
        video_name: 输出视频文件的名称，默认为 "rrt_robot_motion_1.mp4"。
    """
    # 设置视频录制参数
    width, height = 640, 480  # 设置每个相机的分辨率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码格式为 mp4
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 480))  # 录制两个 640x480 的图像并排

    # 设置初始的关节角度
    model.data.qpos[0:6] = start  # 初始化关节配置
    model.forward()  # 更新仿真状态

    # 将路径应用到仿真中，并录制每一步的图像
    for q in path:
        # model.data.qpos[:] = q  # 设置关节角度
        model.data.ctrl[0:6] = q[0:6]  # 设置控制器的关节角度
        
        # 从两个相机渲染图像并将它们并排拼接
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)  # 拼接图像
        
        # 将图像从 RGB 格式转换为 BGR 格式（OpenCV 使用 BGR）
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 将图像写入视频文件
        out.write(frame_bgr)

        # 使仿真步进到下一状态
        model.step()

    # 这里为了稳定悬空
    for i in range(100):
        # 从两个相机渲染图像，并将它们并排拼接
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)  # 拼接两张图像
        
        # 将渲染的图像从 RGB 格式转换为 BGR 格式，OpenCV 使用 BGR 格式进行处理
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 将处理后的图像写入视频文件
        out.write(frame_bgr)

        # 使仿真步进到下一状态
        model.step()

    # 获取当前的起始关节配置（关节位置）
    start_joints_down = model.data.qpos[0:6]

    # 设置目标位置，准备进行后续操作
    target_position_down = target_position

    # 设置最终抓取位置
    target_position_down[2] = 0.15
    target_orientation_euler_down = target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    joint_angles_down = joint_angles_down[1:7]
    print("joint_angles_down:",joint_angles_down * 57.2958)
    # 生成插值因子，num 表示插值的数量，比如 10 表示插值 10 次
    num_interpolations = 50
    t_values = np.linspace(0, 1, num=num_interpolations)
    # 生成角度路径
    interpolated_lists_down = np.array([(1-t)*start_joints_down + t*joint_angles_down for t in t_values])
    # 如果插值路径不为空，说明路径已找到
    if interpolated_lists_down.size > 0:
        print("down path found")  # 打印路径找到的消息

        # 将路径应用到仿真中，并录制视频
        for q in interpolated_lists_down:
            model.data.ctrl[0:6] = q[0:6]  # 设置当前的关节角度
            
            # 从两个相机渲染图像，并将它们并排拼接
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)  # 拼接图像
            
            # 将图像从 RGB 格式转换为 BGR 格式，以便 OpenCV 处理
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            
            # 将当前帧写入视频文件
            out.write(frame_bgr)

            # 更新仿真状态，进行下一步计算
            model.step()
    
    # 现在已经找到了路径并渲染了 目前需要关闭夹爪并渲染
  
    for i in range(30):
        # 从两个相机渲染图像，并将它们并排拼接
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # 将图像从 RGB 格式转换为 BGR 格式，适应 OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 当渲染超过 20 帧时，执行关闭夹爪的操作
        if i >= 20:
            close_gripper()  # 关闭夹爪
        
        # 将当前帧写入视频文件
        out.write(frame_bgr)
        
        # 更新仿真状态，进行下一步计算
        model.step()

    start_joints_up = model.data.qpos[0:6]
    target_position_up = target_position
    target_position_up[2] = 0.5

    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")

    joint_angles_up = joint_angles_up[1:7]
    print("joint_angles_up:",joint_angles_up * 57.2958)

    interpolated_lists_up = np.array([(1-t)*start_joints_up + t*joint_angles_up for t in t_values])
    if interpolated_lists_up.size > 0:
        print("up path found")  # 打印路径找到的消息
        # 将路径应用到仿真中，并录制视频
        for q in interpolated_lists_up:
            model.data.ctrl[0:6] = q  # 设置关节角度（通过控制器进行设置）
            
            # 从两个相机渲染图像，并将它们并排拼接
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)  # 将两个图像并排拼接
            
            # 将图像从 RGB 格式转换为 BGR 格式，以便 OpenCV 处理
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            
            # 将当前帧写入视频文件
            out.write(frame_bgr)

            # 更新仿真状态，进行下一步计算
            model.step()

    for i in range(50):
        # 从两个相机渲染图像，并将它们并排拼接
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)  # 拼接图像
        
        # 将图像从 RGB 格式转换为 BGR 格式，适应 OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 将当前帧写入视频文件
        out.write(frame_bgr)
        
        # 更新仿真状态，进行下一步计算
        model.step()

    # 释放视频写入器资源
    out.release()

    # 输出视频保存路径
    print(f"Video saved as {video_name}")

def close_gripper():
    # 夹爪闭合控制
    model.data.ctrl[6]=0.0
    model.data.ctrl[7]=0.0

def open_gripper():
    # 夹爪打开控制
    model.data.ctrl[6]=0.035
    model.data.ctrl[7]=-0.035

# 初始化物体的真实位姿
def pose_gt_init(physics, body_name="cracker_box", joint_name = 'cracker_box_joint'):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")

    T = cracker_gt_pose[:3,3]
    R = cracker_gt_pose[:3,:3]


    print("真实目标位置1:", T)

    T[0] = -T[0] - 0.1
    T[1] = T[1] + 0.3
    T[2] = 0.1

    print("真实目标位置:", T)

    model.named.data.qpos[joint_name][0:3] = T[0:3]

    # TODO 1 获取的位姿是旋转矩阵，我们这里只保留Z轴旋转角度来验证算法，在这里，你需要从旋转矩阵 `rot` 中提取欧拉角
    rot = R  # cracker_gt_pose的旋转矩阵部分
    # TODO 2 提取欧拉角（XYZ 顺序），确保正确提取旋转的角度，并理解 XYZ 顺序在三维空间中的意义。
    euler_angles = Rotation.from_matrix(rot).as_euler('xyz')

    # 仅保留 Z 轴旋转
    rx = 0.0
    ry = 0.0
    rz = euler_angles[2]
    print("预测Z轴旋转角度", rz * 57.2958) 

    # TODO 3 最终的逆解需要四元数，因此重新构造仅Z轴旋转的四元数，在这里，你需要将提取出的仅绕 Z 轴旋转的欧拉角转换为四元数
    #提示：使用 `Rotation.from_euler()` 方法，可以将欧拉角转换为四元数。四元数用于表示旋转，避免了欧拉角的万向节锁问题
    rot_z_only = Rotation.from_euler('xyz', [rx, ry, rz])
    

    quat = rot_z_only.as_quat()
    quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]  # MuJoCo 格式: [w, x, y, z]
    model.named.data.qpos[joint_name][3:7] = quat_mujoco  

    return T, R

# 初始化物体的预测位姿
def pose_predict_init(physics, body_name="cracker_box", joint_name = 'cracker_box_joint'):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")

    T = cracker_pose_predict[:3,3]
    R = cracker_pose_predict[:3,:3]

    print("预测目标位置1:", T)

    temp = T[0]
    T[0] = T[1] + 0.3
    T[1] = temp + 0.1
    T[2] = 0.4

    print("预测目标位置:", T)
    #与前三个任务结构相同，符号不同代表旋转方向的不同
    # TODO 4 从旋转矩阵中提取欧拉角
    rot = R  # cracker_pose_predict的旋转矩阵部分
    # TODO 5 提取欧拉角（XYZ 顺序）
    euler_angles = Rotation.from_matrix(rot).as_euler('xyz')

    # 仅保留 Z 轴旋转
    rx = 0.0
    ry = 0.0
    rz = -euler_angles[2]   # 保留负符号的 Z 轴旋转角度
    print("预测Z轴旋转角度", -rz * 57.2958)   # 输出负的 Z 轴旋转角度
 
    # TODO 6 构造仅绕 Z 轴旋转的四元数：
    rot_z_only = Rotation.from_euler('xyz', [rx, ry, rz])

    R = rot_z_only.as_matrix()
    print("R",R)
    return T, R

start = [0.0/57.2958, 110/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 45/57.2958]   
reflect_flag = np.array([[1., 0, 0.],[0., 1., 0.],[0., 0, -1.]])
target_orientation_init = np.array([[0., -1, 0.],[-1., -0., 0.],[0., 0, 1.]])

##### 以下是利用训练出来的PoseCNN模型，对随机抽取一张测试集图像进行识别推理#####
import os
import torch
from torch.utils.data import DataLoader
import torchvision.models as models

import utils
from pose_cnn import PoseCNN, eval
from PROPSPoseDataset import PROPSPoseDataset
import matplotlib
matplotlib.use('Agg')  # 必须在其他 matplotlib 导入前设置
import matplotlib.pyplot as plt
import random
import numpy as np

BATCH_SIZE = 4
DRIVE_PATH = '.'

DATA_PATH = "./data"
train_dataset = PROPSPoseDataset(
    DATA_PATH, "train",
    download=False  # =True (for the first time)
) 
val_dataset = PROPSPoseDataset(DATA_PATH, "val")
dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

utils.reset_seed(0)
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# 定义设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

posecnn_model = PoseCNN(pretrained_backbone = vgg16, 
                models_pcd = torch.tensor(val_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                cam_intrinsic = val_dataset.cam_intrinsic).to(DEVICE)
posecnn_model.load_state_dict(torch.load(os.path.join(DRIVE_PATH, "posecnn_model.pth"),weights_only=True))

# 创建 output 文件夹（如果不存在）
output_dir = os.path.join(os.getcwd(), "output")  # 获取当前路径并拼接 output
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹已存在则不会报错

def pose_eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    #可视化结果
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_predict, label = model(inputdict)

    rgb =  (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    vis_result = dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_predict[0],
        alpha = alpha
        )
    
    return vis_result, pose_predict,sample_idx

def evaluate_and_save_samples(model = posecnn_model, 
                              dataloader = dataloader, 
                              dataset = val_dataset, 
                              device = DEVICE, 
                              output_dir = output_dir, 
                              num_samples=1):
    results = []
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估指定数量的样本
    sample_indices = []  # 用于记录选取的样本索引
    for i in range(num_samples):
        # 获取预测结果
        vis_result, pose_predict, sample_idx = pose_eval(model, dataloader, device)
        sample_indices.append(sample_idx)
        
        # 保存可视化结果
        plt.axis('off')
        plt.imshow(vis_result)
        plt.savefig(os.path.join(output_dir, f"output_{i}.png"))
        plt.close()
        
        # 获取真实位姿
        sample = dataset[sample_idx]
        gt_pose_dict = sample['RTs']
        
        # 存储结果
        result = {
            'sample_idx': sample_idx,
            'predicted_poses': pose_predict,
            'ground_truth_poses': gt_pose_dict
        }
        results.append(result)
    
    return sample_idx,pose_predict,gt_pose_dict


# TODO 7 调用解算函数获取预测位姿和真实位姿，在此处填写调用 evaluate_and_save_samples 函数的代码，确保正确传入所有参数
sample_idx,pose_predict,gt_pose = evaluate_and_save_samples(
    model=posecnn_model,
    dataloader=dataloader,
    dataset=val_dataset,
    device=DEVICE,
    output_dir=output_dir,
    num_samples=1
)

##### 以上是利用训练出来的PoseCNN模型，对随机抽取一张测试集图像进行识别推理#####

print('采样图片id:', sample_idx)
# print("PoseCNN预测位姿:", pose_predict)
# print("Ground Truth位姿:", gt_pose)

cracker_pose_predict = pose_predict[0][2]
cracker_gt_pose = gt_pose[1]

# 初始化物体真实位姿与预测物资
gt_postion, gt_orientation = pose_gt_init(model)
predict_position, predict_orientation = pose_predict_init(model)

target_position = predict_position
target_orientation = target_orientation_init @ predict_orientation  @ reflect_flag

print("target_orientation" , target_orientation)
rot = Rotation.from_matrix(target_orientation)
target_orientation_euler = rot.as_euler('xyz')
print("euler_angles = " , target_orientation_euler)

# IK
joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")


goal = joint_angles[1:7]
print("goal",goal*57.2958)
#joint limits
joint_limits = [[-2.618,2.618],[0,3.14158],[-2.697,0],[-1.832,1.832],[-1.22,1.22],[-3.14158,3.14158]] 

# 设置初始关节角度
model.data.qpos[:6] = start
model.forward()

# 初始化RRT
rrt = RRT(start, goal, joint_limits)
rrt_path = rrt.planning(model) 

if rrt_path:
    print("Path found!")

    # 打开夹爪
    open_gripper()
    # Apply RRT Path 
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_posecnn_grasp.mp4")
else:
    print("No path found!")