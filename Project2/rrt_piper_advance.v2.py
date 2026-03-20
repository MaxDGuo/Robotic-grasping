# 导入必要的库
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
import sys


# 加载 dm_control 的物理模型（MuJoCo XML 文件）
model = mujoco.Physics.from_xml_path('assets/chef_can.xml')

# 从 URDF 文件加载机械臂的运动学链（用于逆运动学求解）
my_chain = ikpy.chain.Chain.from_urdf_file("assets/piper_right.urdf")
    
class RRTConnect:
    """RRT-Connect（双向RRT）算法实现"""
    class Node:
        def __init__(self, q):
            self.q = q
            self.path_q = []
            self.parent = None

    def __init__(self, start, goal, joint_limits, expand_dis=0.1, path_resolution=0.01, goal_sample_rate=5, max_iter=5000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        # 双向RRT的两棵树
        self.tree_a = [self.start]  # 起点树
        self.tree_b = [self.end]    # 终点树

    def planning(self, model):
        for i in range(self.max_iter):
            # 1. 从树A随机采样并扩展
            rnd_node = self.get_random_node()
            nearest_ind_a = self.get_nearest_node_index(self.tree_a, rnd_node)
            nearest_node_a = self.tree_a[nearest_ind_a]
            new_node_a = self.steer(nearest_node_a, rnd_node, self.expand_dis)

            if self.check_collision(new_node_a, model):
                self.tree_a.append(new_node_a)
                # 2. 从树B向新节点扩展，尝试连接
                nearest_ind_b = self.get_nearest_node_index(self.tree_b, new_node_a)
                nearest_node_b = self.tree_b[nearest_ind_b]
                new_node_b = self.steer(nearest_node_b, new_node_a, self.expand_dis)

                if self.check_collision(new_node_b, model):
                    self.tree_b.append(new_node_b)
                    # 3. 持续扩展直到连接或碰撞
                    while True:
                        new_node_b_attempt = self.steer(new_node_b, new_node_a, self.expand_dis)
                        if self.check_collision(new_node_b_attempt, model):
                            self.tree_b.append(new_node_b_attempt)
                            new_node_b = new_node_b_attempt
                            # 检查是否连接成功
                            if self.calc_dist_to_node(new_node_b.q, new_node_a.q) <= self.path_resolution:
                                return self.generate_final_course(new_node_a, new_node_b)
                        else:
                            break
            # 4. 交换两棵树，平衡扩展
            self.tree_a, self.tree_b = self.tree_b, self.tree_a
            
        return None

    def get_nearest_node_index(self, node_list, rnd_node):
        """找到离随机节点最近的节点索引"""
        dlist = [np.linalg.norm(np.array(node.q) - np.array(rnd_node.q[:6])) for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """从from_node向to_node扩展，生成新节点"""
        new_node = self.Node(np.array(from_node.q))
        distance = np.linalg.norm(np.array(to_node.q[:6]) - np.array(from_node.q))
        if extend_length > distance:
            extend_length = distance
        num_steps = int(extend_length / self.path_resolution)
        delta_q = (np.array(to_node.q[:6]) - np.array(from_node.q)) / distance if distance > 0 else np.zeros(6)

        for i in range(num_steps):
            new_q = new_node.q + delta_q * self.path_resolution
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            new_node.path_q.append(new_node.q)

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        """随机采样节点（优先采样目标点）"""
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
        else:
            rand_q = self.end.q if self.tree_a[0] == self.start else self.start.q
        return self.Node(rand_q)

    def check_collision(self, node, model):
        """碰撞检测"""
        return check_collision_with_dm_control(model, node.q)

    def generate_final_course(self, node_a, node_b):
        """生成最终路径（拼接两棵树的路径）"""
        # 树A路径（起点到连接点）
        path_a = []
        node = node_a
        while node.parent is not None:
            path_a.append(node.q)
            node = node.parent
        path_a.append(self.start.q)
        path_a.reverse()

        # 树B路径（连接点到终点）
        path_b = []
        node = node_b
        while node.parent is not None:
            path_b.append(node.q)
            node = node.parent
        path_b.append(self.end.q)

        # 拼接路径
        final_path = path_a + path_b
        return final_path
    
    def calc_dist_to_node(self, q1, q2):
        """计算两个关节配置的距离"""
        return np.linalg.norm(np.array(q1) - np.array(q2[:6]))

def get_depth(sim):
    # 获取深度图（单位为米），返回的是一个浮点数组
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # 将最近的深度值平移到原点（最小值变为0）
    depth -= depth.min()
    # 以近距离（小于等于1米）的平均值作为归一化因子进行缩放
    depth /= 2*depth[depth <= 1].mean()
    # 将深度值限制在 [0, 1] 范围并映射到 [0, 255]，用于图像显示
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def check_collision_with_dm_control(model, joint_config):
    """
    碰撞检测：无碰撞 或 仅夹爪与目标物体接触 则返回True
    """
    model.data.qpos[0:6] = joint_config 
    model.forward()  

    contacts = model.data.ncon  
    return contacts == 0 or check_gripper_collision(model)

def check_gripper_collision(model):
    """检查是否仅夹爪与目标物体（chef_can）接触"""
    all_contact_pairs = []
    for i_contact in range(model.data.ncon):
        id_geom_1 = model.data.contact[i_contact].geom1
        id_geom_2 = model.data.contact[i_contact].geom2
        name_geom_1 = model.model.id2name(id_geom_1, 'geom')
        name_geom_2 = model.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    touch_chef_can_right = ("piper_gripper_finger_touch_right", "chef_can_collision") in all_contact_pairs
    touch_chef_can_left = ("piper_gripper_finger_touch_left", "chef_can_collision") in all_contact_pairs
    return touch_chef_can_left or touch_chef_can_right

def get_end_effector_pose(physics, body_name="link6"):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()
    
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")
    
    pos = physics.named.data.xpos[body_name]  
    rot = physics.named.data.xmat[body_name].reshape(3, 3)
    
    return pos.copy(), rot.copy()  

def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_1.mp4"):
    """执行RRT路径并录制视频，包含抓取、下降、上抬动作"""
    width, height = 640, 480  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 480))  

    # 设置起始关节角
    model.data.qpos[0:6] = start
    model.forward()

    # 执行RRT路径
    for q in path:
        model.data.ctrl[0:6] = q[0:6]  
        
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        model.step()

    # 稳定悬空
    for i in range(100):
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        model.step()

    # 抓取前下降：记录当前关节角，设置下降后的目标位置
    start_joints_down = model.data.qpos[0:6]
    target_position_down = target_position.copy()

    # TODO 1: 设置Z轴下降高度（0.15米，合理抓取高度）
    target_position_down[2] = target_position[2] - 0.15

    # 计算下降后的关节角
    target_orientation_euler_down = target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    joint_angles_down = joint_angles_down[1:7]
    print("joint_angles_down (deg):", joint_angles_down * 57.29)
    
    # 插值步数
    num_interpolations = 30
    t_values = np.linspace(0, 1, num=num_interpolations)

    # TODO 2: 生成下降的关节角插值轨迹（逐关节线性插值）
    interpolated_lists_down = np.array([
        np.interp(t_values, [0, 1], [start_joints_down[i], joint_angles_down[i]]) 
        for i in range(6)
    ]).T  # 转置为 (30,6) 形状

    # 执行下降动作
    if interpolated_lists_down.size > 0:
        print("down path found")
        for q in interpolated_lists_down:
            # TODO 3: 设置控制指令为当前插值关节角
            model.data.ctrl[0:6] = q
            
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            model.step()
    
    # 闭合夹爪
    for i in range(50):
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        if i >= 40 :
            close_gripper()
        out.write(frame_bgr)
        model.step()      

    # TODO 4: 抓取后上抬 - 读取当前关节角，设置上抬高度
    start_joints_up = model.data.qpos[0:6]  # 抓取后的关节角
    target_position_up = target_position_down.copy()  # 以上降后的位置为基础
    target_position_up[2] = target_position_down[2] + 0.1  # 上抬0.1米

    # 计算上抬后的关节角
    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")
    joint_angles_up = joint_angles_up[1:7]
    print("joint_angles_up (deg):", joint_angles_up * 57.29)

    # TODO 5: 生成上抬的关节角插值轨迹
    interpolated_lists_up = np.array([
        np.interp(t_values, [0, 1], [start_joints_up[i], joint_angles_up[i]]) 
        for i in range(6)
    ]).T

    # 执行上抬动作
    if interpolated_lists_up.size > 0:
        print("up path found")  
        for q in interpolated_lists_up:
            model.data.ctrl[0:6] = q  
            
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            model.step()      

    # 稳定上抬后的位置
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
    """闭合夹爪"""
    model.data.ctrl[6] = 0.0
    model.data.ctrl[7] = 0.0

def open_gripper():
    """打开夹爪"""
    model.data.ctrl[6] = 0.035
    model.data.ctrl[7] = -0.035


# 初始关节角（弧度）
start = [145.0/57.2958, 90/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 45/57.2958]   

# TODO 6: 目标位置（世界坐标系）x=0.15, y=0.25, z=0.3（初始高度，后续下降）
target_position = np.array([0.15, 0.25, 0.3])

# TODO 7: 目标欧拉角（弧度）：x=180°(np.pi), y=0°, z=-90°(-np.pi/2)，并转换为旋转矩阵
target_orientation_euler = np.array([np.pi, 0, -np.pi/2])
target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]

# IK逆运动学求解目标关节角
joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")
goal = joint_angles[1:7]  # 提取前6个关节角（去除基座固定关节）
print("goal joint angles (deg):", goal * 57.2958)

# 关节运动范围（弧度）
joint_limits = [[-2.618,2.618],[0,3.14158],[-2.697,0],[-1.832,1.832],[-1.22,1.22],[-3.14158,3.14158]] 

# 设置初始关节角并更新仿真
model.data.qpos[:6] = start
model.forward()

# 获取初始末端位姿
init_pos, init_rot = get_end_effector_pose(model)
print("\n初始末端位置:", np.round(init_pos, 4))
print("初始旋转矩阵:\n", np.round(init_rot, 4))

# 转换为欧拉角（角度制）
from scipy.spatial.transform import Rotation as R
euler_angles = R.from_matrix(init_rot).as_euler('xyz', degrees=True)
print("初始欧拉角(度):", np.round(euler_angles, 2))

# TODO 8: 初始化RRT-Connect规划器并生成路径
rrt = RRTConnect(start, goal, joint_limits)
rrt_path = rrt.planning(model)

# 执行路径并录制视频
if rrt_path:
    print("Path found!")

    # TODO 9: 打开夹爪准备抓取
    open_gripper()

    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_2.mp4")
else:
    print("No path found!")
