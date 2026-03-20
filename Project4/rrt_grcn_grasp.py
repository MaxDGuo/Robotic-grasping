# import cv2
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
from scipy.spatial.transform import Rotation
#TODO 1.从grcn_generate.py中找到合适的函数导入，以使用训练好的GRCN网络完成抓取流程
from grcn_generate import evaluate_network



# 加载 dm_control 的物理模型
model = mujoco.Physics.from_xml_path('assets/cracker_box.xml')

# 从 URDF 文件中加载机械臂的关节链模型
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
        查找与随机节点最近的已有节点在 node_list 中的索引。
        参数：
            node_list: 当前 RRT 树中的节点列表
            rnd_node: 随机生成的新节点
        返回：
            node_list 中最近节点的索引值
        """
        dlist = [np.linalg.norm(np.array(node.q) - np.array(rnd_node.q[:6])) for node in node_list]
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
    # 获取深度图（单位为米）
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # 将最近的深度值平移至0
    depth -= depth.min()
    # 归一化到 2 倍的平均距离
    depth /= 2*depth[depth <= 1].mean()
    # 缩放至 [0, 255] 范围
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def check_collision_with_dm_control(model, joint_config):
    """
    检查给定的关节配置是否在 dm_control 模拟中产生碰撞。
    参数：
        model: dm_control 模型对象
        joint_config: 需要检测的关节角度
    返回：
        True 表示无碰撞，False 表示发生碰撞
    """
    model.data.qpos[0:6] = joint_config  # 设置当前关节角
    model.forward()  # 更新模拟状态

    contacts = model.data.ncon  # 当前接触点数量
    return contacts == 0 or check_gripper_collision(model) # 如果没有接触点，说明无碰撞

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
    
    # 获取位置 (x,y,z)
    pos = physics.named.data.xpos[body_name]  # 使用named接口
    
    # 旋转矩阵 (3x3)
    rot = physics.named.data.xmat[body_name].reshape(3, 3)
    
    return pos.copy(), rot.copy()  # 返回拷贝避免后续修改
    
def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_1.mp4"):
    """
    将RRT生成的路径（关节角度序列）应用到 dm_control 模拟环境中，
    同时将执行过程录制成视频。
    参数：
        model: dm_control 中的 Mujoco 模型对象
        path: RRT 规划生成的关节角度路径（关节序列列表）
        video_name: 输出视频的文件名
    """
    # 设置视频录制参数
    width, height = 640, 480  # 每个摄像头图像的分辨率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码格式
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 480))  # 两张 640x480 图像拼接，帧率为20fps

    # 设置初始关节角度
    model.data.qpos[0:6] = start
    model.forward()

    # 遍历路径，每一步设置关节控制值并渲染图像
    for q in path:
        model.data.ctrl[0:6] = q[0:6]  # 设置前6个关节的角度（控制值）

        # 从两个摄像头分别渲染图像并横向拼接
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)

        # 将图像从RGB转为BGR（适用于OpenCV）
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)

        # 写入视频文件
        out.write(frame_bgr)

        # 推进仿真一帧
        model.step()

    # 为了使抓取姿态稳定停留几帧
    for i in range(50):
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        model.step()

    # 获取当前关节角度并计算下降到目标高度的路径
    start_joints_down=model.data.qpos[0:6]
    target_position_down = target_position

    target_position_down[2] = 0.13
    target_orientation_euler_down = target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    joint_angles_down = joint_angles_down[1:7]
    print("joint_angles_down:",joint_angles_down * 57.2958)
    # 生成插值因子，num 表示插值的数量，比如 10 表示插值 10 次
    num_interpolations = 50
    t_values = np.linspace(0, 1, num=num_interpolations)

    interpolated_lists_down = np.array([(1-t)*start_joints_down + t*joint_angles_down for t in t_values])
    if interpolated_lists_down.size > 0:
        print("down path found")

        for q in interpolated_lists_down:
            model.data.ctrl[0:6] = q[0:6]
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            model.step()
    
    # #现在已经找到了路径并渲染了 目前需要关闭夹爪 并渲染
  
    for i in range(30):
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        close_gripper()
        out.write(frame_bgr)
        model.step()      


    start_joints_up = model.data.qpos[0:6]
    target_position_up = target_position
    target_position_up[2] = 0.4

    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")

    joint_angles_up = joint_angles_up[1:7]
    print("joint_angles_up:",joint_angles_up * 57.2958)

    interpolated_lists_up = np.array([(1-t)*start_joints_up + t*joint_angles_up for t in t_values])
    if interpolated_lists_up.size > 0:
        print("up path found")
        for q in interpolated_lists_up:
            model.data.ctrl[0:6] = q  # Set joint angles
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            model.step()

    for i in range(50):
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        model.step()
    out.release()
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
    
    T = np.zeros(3) 

    T[0] = -predict_pose[0] * 0.001 + 0.05
    T[1] = predict_pose[1] * 0.001 - 0.05
    T[2] = 0.05

    print("真实目标位置:", T)

    model.named.data.qpos[joint_name][0:3] = T[0:3]

    # 提取欧拉角（XYZ 顺序）
    euler_angles = predict_pose[2]

    # 仅保留 z 轴旋转，
    rx = 0.0
    ry = 0.0
    rz = euler_angles
    print("真实Z轴旋转角度", rz * 57.2958)
    # 重新构造仅 X 轴旋转的四元数
    rot_z_only = Rotation.from_euler('xyz', [rx, ry, rz])
    
    quat = rot_z_only.as_quat()
    quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]  # MuJoCo 格式: [w, x, y, z]
    model.named.data.qpos[joint_name][3:7] = quat_mujoco  

    R = rot_z_only.as_matrix()

    return T, R

# 初始化物体的预测位姿
def pose_predict_init(physics, body_name="cracker_box", joint_name = 'cracker_box_joint'):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")

    T = np.zeros(3) 

    T[0] = predict_pose[1] * 0.001 - 0.05
    T[1] = predict_pose[0] * 0.001 - 0.05
    T[2] = 0.4

    print("预测目标位置:", T)

    # 提取欧拉角（XYZ 顺序）
    euler_angles = predict_pose[2]
    print("预测欧拉角:", euler_angles)
    # 仅保留 Z 轴旋转
    rx = 0.0
    ry = 0.0
    rz = -euler_angles
    print("预测Z轴旋转角度", -rz * 57.2958) 
    # 重新构造仅 z 轴旋转的四元数(这里是x轴旋转)
    rot_z_only = Rotation.from_euler('xyz', [rx,ry,rz])

    R = rot_z_only.as_matrix()
    return T, R

start = [30.0/57.2958, 90/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 45/57.2958]   # Start joint angles
reflect_flag = np.array([[1., 0, 0.],[0., 1., 0.],[0., 0, -1.]])
target_orientation_init = np.array([[0., -1, 0.],[-1., -0., 0.],[0., 0, 1.]])

# TODO 2.使用grcn获取预测位姿和真实位姿
predict_pose, gt_center = evaluate_network()

# 初始化物体真实位姿和预测位姿
gt_postion, gt_orientation = pose_gt_init(model)
predict_position, predict_orientation = pose_predict_init(model)

print("预测旋转矩阵:\n", predict_orientation)

target_position = predict_position
target_orientation = target_orientation_init @ predict_orientation  @ reflect_flag

rot = Rotation.from_matrix(target_orientation)
target_orientation_euler = rot.as_euler('xyz')

print("目标位置:", target_position)
print("目标旋转矩阵:\n", target_orientation)
joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")

goal = joint_angles[1:7]
print("goal",goal * 57.2958)

joint_limits = [[-2.618,2.618],[0,3.14158],[-2.697,0],[-1.832,1.832],[-1.22,1.22],[-3.14158,3.14158]] 

# ----------------- 新增代码 -----------------
# 设置初始关节角度
model.data.qpos[:6] = start
model.forward()

# 初始化RRT算法
rrt = RRT(start, goal, joint_limits)
rrt_path = rrt.planning(model) 

if rrt_path:
    print("Path found!")
    open_gripper()
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_grcn_grasp.mp4")
else:
    print("No path found!")