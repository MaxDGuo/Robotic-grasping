import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation as Rotation
from stable_baselines3.common.evaluation import evaluate_policy
import time
import glfw

import argparse

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self, render=True):
        super(PiperEnv, self).__init__()
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 构造 scene.xml 的完整路径
        xml_path = os.path.join(script_dir, '..', 'mujoco_asserts', 'agilex_piper_grasp', 'scene.xml')
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self._glfw_window = glfw.create_window(640, 480, "Hidden", None, None)
        glfw.make_context_current(self._glfw_window)

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')

        # 初始化渲染所需结构体（顺序不能错）
        self.camera = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        self.render_mode = render
        if self.render_mode:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 0
            self.handle.cam.elevation = -30

            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        else:
            self.handle = None

        self.rl_model = PPO.load("./piper_grasp_ppo_model")

        # 各关节运动限位
        self.ctrl_limits = np.array([
            (-1.5728, 1.5728),  
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0, 0.035),
        ])

        self.angle_init = np.array([0.0/57.2958, 90/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 0/57.2958], dtype=np.float32)

        # 动作空间，7 个控制量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        self.init_reward = 0
        self.last_gripper_pos = 0

        # 环境中 robot 关节 
        self.robot_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        # 环境中需要交互的物体
        self.object_names = ["apple"]
        # 计算关节数量
        num_joints = len(self.robot_joint_names)
        # 构建 observation_space

        # 观测空间，包含末端位姿和目标位姿
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(16,),  # 与_get_observation()输出维度一致
            dtype=np.float32
        )

        # 随机采样设置
        self.np_random = None   
        self.step_number = 0
        self._reset_noise_scale = 1e-2
        # 一个环境最大采样次数
        self.episode_len = 75

        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")


    def get_sensor_data(self, sensor_name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]  # ← 这里改了
        return sensor_values

    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos)        # shape (3,)

        # 方向：MuJoCo 已存成9元素向量，无需reshape
        xmat = np.array(self.data.site(site_id).xmat)            # shape (9,)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)                    # [w, x, y, z]

        return position, quaternion

    
    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到每个关节的具体角度范围。

        Args:
            action (np.ndarray): 形状为 (6,) 的数组，值范围在 [-1, 1]

        Returns:
            np.ndarray: 形状为 (6,) 的数组，映射到实际关节角度范围，类型为 numpy.ndarray
        """

        normalized = (action + 1) / 2
        lower_bounds = self.ctrl_limits[:, 0]
        upper_bounds = self.ctrl_limits[:, 1]
        # 插值计算
        mapped_action = lower_bounds + normalized * (upper_bounds - lower_bounds)

        return mapped_action
    
    def _set_state(self, joint_names):
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"未找到名为 '{name}' 的关节")

            qpos_start = self.model.jnt_qposadr[joint_id]
            qvel_start = self.model.jnt_dofadr[joint_id]

            joint_type = self.model.jnt_type[joint_id]
            if joint_type == 0:
                dof = 7
            elif joint_type == 1:
                dof = 4
            else:
                dof = 1

            self.data.qpos[qpos_start : qpos_start + dof] = np.zeros(dof)
            self.data.qvel[qvel_start : qvel_start + dof] = np.zeros(dof)


    def _reset_objects_positions(self, object_names, xy_low=(-0.0, -0.15), xy_high=(0.15, 0.15), fixed_z=0.766):
        if isinstance(object_names, str):
            object_names = [object_names]

        for name in object_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"未找到名为 '{name}' 的关节")

            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]

            joint_type = self.model.jnt_type[joint_id]
            if joint_type == 0:  # 自由关节（free joint）
                dof_qpos = 7  # 位置(3) + 四元数(4)
                dof_qvel = 6  # 线速度(3) + 角速度(3)
            elif joint_type == 1:  # 球关节 (ball joint)
                dof_qpos = 4  # 四元数(4)
                dof_qvel = 3  # 角速度(3)
            else:  # 其他关节（铰链、滑动）
                dof_qpos = 1  # 标量位置
                dof_qvel = 1  # 标量速度

            xy = self.np_random.uniform(low=xy_low, high=xy_high)
            z = fixed_z

            def random_unit_quaternion(rng):
                """生成随机单位四元数, rng为numpy随机生成器"""
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                return q

            if joint_type == 0:  # free joint，位置 + 四元数
                self.data.qpos[qpos_idx : qpos_idx + dof_qpos] = np.concatenate([xy, [z], random_unit_quaternion(self.np_random)])
            elif joint_type == 1:  # 球关节，只设置四元数
                self.data.qpos[qpos_idx : qpos_idx + dof_qpos] = random_unit_quaternion(self.np_random)
            else:  # 其他关节，只设置标量位置（xy和z无意义）
                self.data.qpos[qpos_idx] = 0.0  # 可以改成其他合理的初始值
                self.data.qpos[0:6] = self.angle_init
            self.data.qpos[6] = 0.035  # 设置末端夹爪的打开

            # 速度全部归零
            self.data.qvel[qvel_idx : qvel_idx + dof_qvel] = np.zeros(dof_qvel)
        

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        if self.robot_joint_names is not None and self.object_names is not None:
            self._set_state(self.robot_joint_names)
            self._reset_objects_positions(self.object_names)
        else:
            raise ValueError(f" You need provide robot_joint_names and object_names to reset env. ")
        # mujoco 环境往前走一步
        mujoco.mj_step(self.model, self.data)
        obs = self._get_observation()
        self.step_number = 0
        print(f"reset env successed. ")

        return obs, {}
    
    def _get_observation(self):
        # 1. 获取夹爪位姿
        gripper_pos, gripper_quat = self._get_site_pos_ori("end_ee")  # [3,], [4,]
        
        # 2. 获取目标物块位姿
        target_pos, target_quat = self._get_body_pose("apple")  # [3,], [4,]
        
        # 3. 计算相对位姿（可选但推荐）
        rel_pos = target_pos - gripper_pos  # [3,]
        rel_rot = (Rotation.from_quat(gripper_quat).inv() * 
                Rotation.from_quat(target_quat)).as_rotvec()  # [3,]
        
        # 4. 组合观测
        return np.concatenate([
            gripper_pos,       # 绝对位置 [3]
            gripper_quat,      # 绝对姿态 [4]
            target_pos,        # 目标位置 [3]
            rel_pos,           # 相对位置 [3] 
            rel_rot            # 相对旋转 [3]
        ]).astype(np.float32)  # 总维度: 3+4+3+3+3 = 16


    
    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过body名称获取其位姿信息, 返回一个7维向量
        :param body_name: body名称字符串
        :return: 7维numpy数组, 格式为 [x, y, z, w, x, y, z]
        :raises ValueError: 如果找不到指定名称的body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        
        # 提取位置和四元数并合并为一个7维向量
        position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]
        
        return position, quaternion
    
    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos)        # shape (3,)

        # 方向：MuJoCo 已存成9元素向量，无需reshape
        xmat = np.array(self.data.site(site_id).xmat)            # shape (9,)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)                    # [w, x, y, z]

        return position, quaternion
    
    def _check_table_collision(self):
        """检测夹爪(link7/link8)与桌子的碰撞"""
        # 获取夹爪link7和link8的所有几何体ID
        link7_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        link8_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        
        gripper_geom_ids = [
            i for i, g in enumerate(self.model.geom_bodyid) 
            if g in [link7_body, link8_body] and 
            self.model.geom_group[i] == 1  # 只检测参与碰撞的几何体(group=0)
        ]
        
        # 检查所有接触对
        for contact in self.data.contact:
            if (contact.geom1 in gripper_geom_ids and contact.geom2 == self.table_geom_id) or \
            (contact.geom2 in gripper_geom_ids and contact.geom1 == self.table_geom_id):
                return True
        return False
    
    def _check_apple_collision(self):
        """检测夹爪(link7/link8)与桌子的碰撞"""
        # 获取夹爪link7和link8的所有几何体ID
        link7_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        link8_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        
        gripper_geom_ids = [
            i for i, g in enumerate(self.model.geom_bodyid) 
            if g in [link7_body, link8_body] and 
            self.model.geom_group[i] == 0  # 只检测参与碰撞的几何体(group=0)
        ]
        
        # 检查所有接触对
        for contact in self.data.contact:
            if (contact.geom1 in gripper_geom_ids and contact.geom2 == self.table_geom_id) or \
            (contact.geom2 in gripper_geom_ids and contact.geom1 == self.table_geom_id):
                return True
        return False

    def _get_reward(self, observation):


        target_quat = np.array([0.82570097,-0.00302722,0.56409566,-0.00219734])
        # 从观测中提取各分量
        gripper_quat = observation[3:7]  # 夹爪四元数
        rel_pos = observation[10:13]           # 相对位置
        
        # 1. 位置奖励 (基于相对位置)
        distance = np.linalg.norm(rel_pos)
        pos_reward = 5 * (1 - np.tanh(10 * distance))  # 距离越小奖励越大
        
        # 2. 姿态奖励 (基于旋转向量模长)
        rel_rot = (Rotation.from_quat(gripper_quat).inv() * 
        Rotation.from_quat(target_quat)).as_rotvec()  # [3,]
        rot_error = np.linalg.norm(rel_rot)
        ori_reward = 1 * (1 - np.tanh(3 * rot_error))

        # 3. 成功奖励
        success_reward = 0.0
        if distance <= 0.02:  
            success_reward = 10.0
        
        # 4.碰撞惩罚
        collision_table_penalty = -5 if self._check_table_collision() else 0
        collision_apple_penalty = -5 if self._check_apple_collision() else 0
        collision_penalty = collision_table_penalty + collision_apple_penalty
        # 组合奖励 (可调整权重)
        total_reward = pos_reward + ori_reward + success_reward + collision_penalty
        
        # 调试输出
        # print(f"位置奖励:{pos_reward:.2f} 姿态奖励:{ori_reward:.2f} 成功奖励:{success_reward:.2f} 碰撞惩罚：{collision_penalty:.2f} 距离:{distance:.2f}")
        
        return total_reward
    
    def step(self, action):

        # 将 action 映射回真实机械臂关节空间
        mapped_action = self.map_action_to_joint_limits(np.append(action, 1.0))
        self.data.ctrl[:7] = mapped_action
        # mujoco 仿真向前推进一步 (这里只更新 qpos , 并不会做动力学积分)
        mujoco.mj_step(self.model, self.data)

        self.step_number += 1
        observation = self._get_observation()
        # Check if observation contains only finite values
        is_finite = False
        reward = self._get_reward(observation)
        
        done = False
        # 1. 检查是否成功抓取（距离阈值 + 姿态对齐）
        gripper_pos, _ = self._get_site_pos_ori("end_ee")
        target_pos, _ = self._get_body_pose("apple")
        dist = np.linalg.norm(gripper_pos - target_pos)
        
        # 终止条件判断（也使用观测数据）
        distance = np.linalg.norm(observation[10:13])  # 相对位置模长
        rot_error = np.linalg.norm(observation[13:16]) # 旋转误差
        if distance <= 0.02:
            done = True

        info = {'is_success': done}
        truncated = self.step_number > self.episode_len
        if self.handle is not None:
            self.handle.sync()

        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    env = PiperEnv()
    observation, _ = env.reset()
    
    # 添加统计变量
    total_episodes = 0
    successful_episodes = 0
    episode_rewards = []
    current_episode_reward = 0
    
    # 后处理状态变量
    post_processing = False
    post_step = 0
    saved_joint_pos = np.zeros(6)

    try:
        for step in range(5000):
            if not post_processing:
                # 正常RL控制阶段
                action, _states = env.rl_model.predict(observation, deterministic=True)
                mapped_action = env.map_action_to_joint_limits(np.append(action, 1.0))
                observation, reward, done, truncated, info = env.step(action)
                current_episode_reward += reward
                
                if done :
                    print("开始后处理：抓取并举起物体")
                    post_processing = True
                    post_step = 0
                    saved_joint_pos = env.data.qpos[:6].copy()

                if truncated:
                    total_episodes += 1
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                    
                    # 每10个episode打印统计信息
                    if total_episodes % 10 == 0:
                        success_rate = successful_episodes / total_episodes * 100
                        avg_reward = np.mean(episode_rewards[-10:])
                        print(f"\nEpisode {total_episodes}:")
                        print(f"Success rate: {success_rate:.2f}%")
                        print(f"Average reward (last 10): {avg_reward:.2f}\n")
                    
                    observation, _ = env.reset()
                    
            else:
                # 后处理阶段
                post_step += 1
                
                if post_step <= 50:
                    env.data.qpos[:6] = saved_joint_pos
                    env.data.ctrl[6] = 0.01  # 闭合夹爪
                    if post_step == 50 :
                        env.data.ctrl[:6] = saved_joint_pos

                # 第二阶段：举起物体 (持续50步)
                if post_step <= 100 and post_step > 50:
                    env.data.ctrl[1] = env.data.ctrl[1] - 0.01
                    print("data.ctrl = ", env.data.ctrl)
                    env.data.ctrl[6] = 0.01  # 保持夹爪闭合

                mujoco.mj_step(env.model, env.data)

                if post_step >= 100:
                    post_processing = False
                    print("reward = ",reward)
                    total_episodes += 1
                    if done:
                        successful_episodes += 1
                    
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                    
                    # 每10个episode打印统计信息
                    if total_episodes % 10 == 0:
                        success_rate = successful_episodes / total_episodes * 100
                        avg_reward = np.mean(episode_rewards[-10:])
                        print(f"\nEpisode {total_episodes}:")
                        print(f"Success rate: {success_rate:.2f}%")
                        print(f"Average reward (last 10): {avg_reward:.2f}\n")
                    
                    observation, _ = env.reset()

            if env.handle is not None:
                env.handle.sync()
            time.sleep(0.05)
            
    finally:
        env.close()



