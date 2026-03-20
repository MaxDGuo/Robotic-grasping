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
import glfw
import argparse
from contextlib import redirect_stdout

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self, render=True):
        super(PiperEnv, self).__init__()
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 构造 scene.xml 的完整路径（请确认这个路径和你的项目匹配！）
        xml_path = os.path.join(script_dir, '..', 'mujoco_asserts', 'agilex_piper_grasp', 'scene.xml')
        # 确保路径存在
        xml_path = os.path.abspath(xml_path)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"scene.xml路径不存在: {xml_path}")

        # ========== 核心修复：仅在渲染时初始化OpenGL相关资源 ==========
        self.render_mode = render
        self._glfw_initialized = False
        self._glfw_window = None
        self.handle = None
        self.camera = None
        self.scene = None
        self.context = None

        # 先加载模型（无渲染也需要）
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')

        # 仅在渲染开启时初始化OpenGL/Viewer
        if self.render_mode:
            # 强制设置MuJoCo的GL后端（解决OpenGL加载问题）
            os.environ['MUJOCO_GL'] = 'glfw'
            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW")
            self._glfw_initialized = True
            
            # 创建GLFW窗口（必须有上下文才能创建MjrContext）
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)  # 显示窗口（方便调试）
            self._glfw_window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
            glfw.make_context_current(self._glfw_window)

            # 初始化渲染结构体（仅渲染模式需要）
            self.camera = mujoco.MjvCamera()
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            
            # 启动Viewer
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 0
            self.handle.cam.elevation = -30
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

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

        self.angle_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 动作空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        self.init_reward = 0
        self.last_gripper_pos = 0

        # 环境参数
        self.robot_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.object_names = ["apple"]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(16,),
            dtype=np.float32
        )

        # 随机采样设置
        self.np_random = None   
        self.step_number = 0
        self._reset_noise_scale = 1e-2
        self.episode_len = 300

        # 预存几何体ID
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.apple_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "apple")
        self.apple_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple")

    def get_sensor_data(self, sensor_name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]
        return sensor_values

    def close(self):
        # 释放所有资源
        if self.handle:
            self.handle.close()
            self.handle = None
        if self._glfw_window:
            glfw.destroy_window(self._glfw_window)
            self._glfw_window = None
        if self._glfw_initialized:
            glfw.terminate()
            self._glfw_initialized = False
        if self.context:
            del self.context
        if self.scene:
            del self.scene
        if self.model:
            del self.model
        if self.data:
            del self.data

    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的site")
        position = np.array(self.data.site(site_id).xpos)
        xmat = np.array(self.data.site(site_id).xmat)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)
        return position, quaternion
    
    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        normalized = (action + 1) / 2
        lower_bounds = self.ctrl_limits[:, 0]
        upper_bounds = self.ctrl_limits[:, 1]
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
            dof = 7 if joint_type == 0 else 4 if joint_type == 1 else 1
            self.data.qpos[qpos_start : qpos_start + dof] = np.zeros(dof)
            self.data.qvel[qvel_start : qvel_start + dof] = np.zeros(dof)

    def _reset_objects_positions(self, object_names, xy_low=(-0.0, -0.2), xy_high=(0.2, 0.2), fixed_z=0.766):
        if isinstance(object_names, str):
            object_names = [object_names]
        for name in object_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"未找到名为 '{name}' 的关节")
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]
            joint_type = self.model.jnt_type[joint_id]
            dof_qpos = 7 if joint_type == 0 else 4 if joint_type == 1 else 1
            dof_qvel = 6 if joint_type == 0 else 3 if joint_type == 1 else 1

            xy = self.np_random.uniform(low=xy_low, high=xy_high)
            z = fixed_z

            def random_unit_quaternion(rng):
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                return q

            if joint_type == 0:
                self.data.qpos[qpos_idx : qpos_idx + dof_qpos] = np.concatenate([xy, [z], random_unit_quaternion(self.np_random)])
            elif joint_type == 1:
                self.data.qpos[qpos_idx : qpos_idx + dof_qpos] = random_unit_quaternion(self.np_random)
            else:
                self.data.qpos[qpos_idx] = 0.0
            self.data.qpos[0:6] = self.angle_init
            self.data.qpos[6] = 0.035
            self.data.qvel[qvel_idx : qvel_idx + dof_qvel] = np.zeros(dof_qvel)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.robot_joint_names is not None and self.object_names is not None:
            self._set_state(self.robot_joint_names)
            self._reset_objects_positions(self.object_names)
        else:
            raise ValueError("需要提供robot_joint_names和object_names来重置环境")
        mujoco.mj_step(self.model, self.data)
        obs = self._get_observation()
        self.step_number = 0
        return obs, {}
    
    def _get_observation(self):
        gripper_pos, gripper_quat = self._get_site_pos_ori("end_ee")
        target_pos, target_quat = self._get_body_pose("apple")
        rel_pos = target_pos - gripper_pos
        rel_rot = (Rotation.from_quat(gripper_quat).inv() * 
                Rotation.from_quat(target_quat)).as_rotvec()
        return np.concatenate([
            gripper_pos, gripper_quat, target_pos, rel_pos, rel_rot
        ]).astype(np.float32)

    def _get_body_pose(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        position = np.array(self.data.body(body_id).xpos)
        quaternion = np.array(self.data.body(body_id).xquat)
        return position, quaternion
    
    def _check_table_collision(self):
        link7_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        link8_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        gripper_geom_ids = [
            i for i, g in enumerate(self.model.geom_bodyid) 
            if g in [link7_body, link8_body] and self.model.geom_group[i] == 1
        ]
        for contact in self.data.contact:
            if (contact.geom1 in gripper_geom_ids and contact.geom2 == self.table_geom_id) or \
            (contact.geom2 in gripper_geom_ids and contact.geom1 == self.table_geom_id):
                return True
        return False
    
    def _check_apple_collision(self):
        link7_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        link8_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        gripper_geom_ids = [
            i for i, g in enumerate(self.model.geom_bodyid) 
            if g in [link7_body, link8_body] and self.model.geom_group[i] == 0
        ]
        for contact in self.data.contact:
            if (contact.geom1 in gripper_geom_ids and contact.geom2 == self.apple_geom_id) or \
            (contact.geom2 in gripper_geom_ids and contact.geom1 == self.apple_geom_id):
                return True
        return False

    def _get_reward(self, observation):
        target_quat = np.array([0.82570097,-0.00302722,0.56409566,-0.00219734])
        gripper_quat = observation[3:7]
        rel_pos = observation[10:13]
        distance = np.linalg.norm(rel_pos)
        pos_reward = 5 * (1 - np.tanh(10 * distance))
        rel_rot = (Rotation.from_quat(gripper_quat).inv() * 
        Rotation.from_quat(target_quat)).as_rotvec()
        rot_error = np.linalg.norm(rel_rot)
        ori_reward = 1 * (1 - np.tanh(3 * rot_error))
        success_reward = 10.0 if distance <= 0.02 else 0.0
        collision_table_penalty = -5 if self._check_table_collision() else 0
        collision_apple_penalty = -5 if self._check_apple_collision() else 0
        collision_penalty = collision_table_penalty + collision_apple_penalty
        total_reward = pos_reward + ori_reward + success_reward + collision_penalty
        if distance <= 0.02:
            print(f"位置奖励:{pos_reward:.2f} 姿态奖励:{ori_reward:.2f} 成功奖励:{success_reward:.2f} 碰撞惩罚：{collision_penalty:.2f} 距离:{distance:.2f}")
        return total_reward

    def step(self, action):
        mapped_action = self.map_action_to_joint_limits(np.append(action, 1.0))
        self.data.ctrl[:7] = mapped_action
        mujoco.mj_step(self.model, self.data)
        self.step_number += 1
        observation = self._get_observation()
        reward = self._get_reward(observation)
        done = False
        gripper_pos, _ = self._get_site_pos_ori("end_ee")
        target_pos, _ = self._get_body_pose("apple")
        dist = np.linalg.norm(gripper_pos - target_pos)
        if reward > 10:
            done = True
        info = {'is_success': done}
        truncated = self.step_number > self.episode_len
        # 仅渲染模式同步Viewer
        if self.handle is not None and self.step_number % 10 == 0:
            self.handle.sync()
        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PiperEnv RL simulation.")
    parser.add_argument('--render', action='store_true', help='Enable rendering with GUI viewer')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel envs (训练用4，验证用1)')
    args = parser.parse_args()

    # ========== 核心：根据是否渲染，自动选择环境创建方式 ==========
    if not args.render:
        # 【训练场景】无渲染，用多进程并行环境
        os.environ['MUJOCO_GL'] = 'osmesa'  # 无GUI环境用osmesa后端，避免OpenGL错误
        from stable_baselines3.common.vec_env import SubprocVecEnv  # 导入多进程环境
        
        # 创建多进程并行环境（效率最高）
        env = make_vec_env(
            lambda: PiperEnv(render=False), 
            n_envs=args.n_envs,
            vec_env_cls=SubprocVecEnv  # 必须用SubprocVecEnv，不要用DummyVecEnv
        )
        print(f"训练模式：启动 {args.n_envs} 个并行环境，无渲染")
    else:
        # 【验证/可视化场景】开渲染，强制单环境
        assert args.n_envs == 1, "开启渲染仅支持n_envs=1，多进程会导致MuJoCo卡死"
        os.environ['MUJOCO_GL'] = 'glfw'  # 有GUI环境用glfw后端
        env = PiperEnv(render=args.render)
        print("验证模式：启动 1 个环境，开启渲染")

    # ========== 后面的PPO模型代码保持不变 ==========
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    model = PPO(
        "MlpPolicy",   
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=300,  # 和TODO1的episode_len对齐
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        learning_rate=3e-4,
        device="cpu",
        tensorboard_log="./ppo_piper_grasp/"
    )
    
    total_rollouts = 8000
    _total_timesteps = 300 * total_rollouts * args.n_envs  # 总步数随n_envs自动翻倍
    
    log_file = "./training_log.txt"
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            model.learn(
                total_timesteps=_total_timesteps, 
                progress_bar=True,
            )
    
    model.save("piper_grasp_ppo_model")
    env.close()
    print("模型保存成功！")