# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains import TerrainImporter

from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab.sensors import RayCaster


class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._dof_idx, _ = self.robot.find_joints(self.cfg.dof_name) #获取机器人左右轮关节的索引

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self.terrain = TerrainImporter(self.cfg.terrain)
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # self._height_scanner = RayCaster(self.cfg.height_scanner)
        # self.scene.sensors["height_scanner"] = self._height_scanner #添加地形高度扫描传感器

        self.visualization_markers = define_markers() #标记可视化对象
        
        #初始化环境变量
        self.up_dir = torch.tensor([0.0, 0.0, 1.1]).cuda() #定义向上方向向量（单位向量）
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda() #初始化命令偏航角为0
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda() #初始命令向量为随机值，[num_envs, (x,y,z)]
        self.commands[:,-1] = 0.0 #将命令向量的z分量设为0，确保在水平面内运动
        self.commands = self.commands / torch.linalg.norm(self.commands, dim=-1, keepdim=True) #归一化命令向量

        #计算初始偏航角yaw
        #横向为x，纵向为y，偏航角为0时与x轴正方向一致，[-pi, pi]，逆时针为正方向
        ratio = self.commands[:,1]/(self.commands[:,0]+1e-8) #初始朝向y/x比值，计算反正切以获得初始偏航角
        gzero = torch.where(self.commands[:, :2] > 0, True, False)
        lzero = torch.where(self.commands[:, :2] < 0, True, False)
        # ratio = self._commands[:,1]/(self._commands[:,0]+1e-8) #初始朝向y/x比值，计算反正切以获得初始偏航角
        # gzero = torch.where(self._commands > 0, True, False)
        # lzero = torch.where(self._commands < 0, True, False)
        plus = lzero[:,0] * gzero[:,1] #第二象限 lzero=[TRue, False], gzero=[False, True], plus=True，其余象限为False
        minus = lzero[:,0] * lzero[:,1] #第三象限 lzero=[True, True], minus=True，其余象限为False
        offsets = torch.pi*plus - torch.pi*minus #第二象限加π，第三象限减π，其余象限不变
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1) #计算初始命令的偏航角，并加上象限修正，区分一与三、二与四象限，确保偏航角在[-π, π]范围内

        #初始化标记位置和朝向张量
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda() #标记位置偏移量
        self.marker_offset[:,-1] = 0.5 #标记位置偏移量，z轴方向上0.5米，确保标记悬浮在机器人上方
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda() #前进方向标记四元数初始化，四元数将箭头图元绕z轴旋转由命令定义的偏航角yaw方向，四元数格式为(w, x, y, z)
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda() #命令方向标记四元数初始化

    #标记可视化函数
    def _visualize_markers(self):
        self.marker_locations = self.robot.data.root_pos_w #机器人当前世界坐标系位置，作为标记的初始位置
        self.forward_marker_orientations = self.robot.data.root_quat_w #机器人当前朝向四元数，作为前进方向标记的朝向
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze() #绕up_dir轴旋转yaw角度的四元数，表示命令方向标记的朝向

        loc = self.marker_locations + self.marker_offset #标记的最终位置,为机器人位置加上偏移量
        loc = torch.vstack((loc, loc)) #创建两倍于环境数量的位置张量，以匹配两个标记的数量(2*num_envs, 3)---标记位置
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations)) #将两个标记的朝向四元数堆叠在一起---标记朝向

        all_envs = torch.arange(self.cfg.scene.num_envs) #从0到num_envs-1的环境索引张量
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs))) #为每个环境创建标记索引，前半部分为0作为forward前进的标记，后半部分为1作为command指令的标记），(2*num_envs)
        self.visualization_markers.visualize(loc, rots, marker_indices=indices) #调用可视化标记函数，传入位置、朝向和标记索引

    #预物理步函数
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone() #克隆动作张量，防止修改原始数据
        self._visualize_markers() #调用标记可视化函数，使箭头标记可见

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self._dof_idx) #设置机器人左右轮关节的速度目标

    def _get_observations(self) -> dict:
        # self.velocity = self.robot.data.root_com_vel_w #获取机器人质心线速度、角速度
        # self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B) #将机器人本体前进方向向量转换到世界坐标系
        # obs = torch.hstack((self.velocity, self.commands)) #将质心线速度和命令向量水平拼接作为观测值
        # observations = {"policy": obs} #将观测值作为键"policy"对应策略网络需要的观测值返回，critic定义评价模型观测值，“Actor-Critic”
        # return observations

        #尽可能缩小观测空间维度，减少模型参数量
        # self.velocity = self.robot.data.root_com_vel_w #机器人质心线速度，世界坐标系表示
        # self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B) #root_link_quat_w：局部坐标系到世界坐标系的四元数转换，FORWARD_VEC_B：机器人本体前进方向向量(1,0,0)局部坐标系表示
        # dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True) #机器人前进方向与命令方向的点积，衡量两者的一致性
        # cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1) #机器人前进方向与命令方向的叉积在z轴方向的分量，衡量两者的偏离方向
        # # cross = torch.cross(self.commands, self.forwards, dim=-1)[:,-1].reshape(-1,1)
        # forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1) #机器人质心线速度在本体x轴方向的分量
        # obs = torch.hstack((dot, cross, forward_speed)) #机器人前进方向与命令方向的点积、机器人前进方向与命令方向的叉积在z轴方向的分量、机器人质心线速度在本体x轴方向的分量
        # observations = {"policy": obs}

        #my_observation
        self.lin_vel = self.robot.data.root_lin_vel_b #获取机器人质心线速度（含方向、大小），机器人本体坐标系表示
        # self.lin_vel = self.robot.data.root_lin_vel_w #获取机器人质心线速度（含方向、大小），世界坐标系表示，不能作为self.forwards！！！！！！详情见飞书文档
        self.forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B) #将机器人本体前进方向向量（单位向量，仅方向）转换到世界坐标系
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1) 
        obs = torch.hstack((dot, cross, self.lin_vel))
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # cos_theta = torch.sum(self.forwards[:, :2] * self.commands[:, :2], dim=-1) #commands已归一化，[-1,1]，世界坐标系
        # forward_reward = (cos_theta + 1.0) / 2.0 #将cos_theta从[-1,1]映射到[0,1]作为前进奖励

        forward_velocity = torch.sum(self.robot.data.root_lin_vel_w[:, :2] * self.commands[:, :2], dim=-1) #在命令方向的速度投影
        velocity_reward = torch.clamp(forward_velocity, 0, 1.0)
        # velocity_reward = torch.tanh(velocity_reward)

        # cmd_yaw = torch.atan2(self.commands[:,1], self.commands[:,0])
        # robot_yaw = torch.atan2(self.forwards[:,1], self.forwards[:,0])
        # yaw_error = cmd_yaw - robot_yaw
        # yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        # heading = torch.cos(yaw_error) #使用余弦函数计算偏航奖励，范围[-1,1]
        # heading_reward = (heading + 1.0) / 2.0 #将偏航奖励从[-1,1]映射到[0,1]

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1) 
        yaw_error = torch.atan2(cross, dot) #使用反正切函数计算偏航误差，范围[-π, π]
        yaw_reward = torch.exp(-3*torch.abs(yaw_error)).squeeze(-1)

        rewards = yaw_reward * (9*velocity_reward+1.0)
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out #是否达到终止状态（False），是否超时

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        #命令重置
        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda() #为重置的环境生成新的随机命令向量
        self.commands[env_ids,-1] = 0.0 #将命令向量的z分量设为0，确保在水平面内运动
        self.commands[env_ids] = self.commands[env_ids] / torch.linalg.norm(self.commands[env_ids], dim=-1, keepdim=True) #归一化命令向量

        #计算重置后命令的偏航角yaw
        ratio = self.commands[env_ids,1]/(self.commands[env_ids,0]+1e-8) #计算反正切以获得偏航角
        gzero = torch.where(self.commands[env_ids, :2] > 0, True, False)
        lzero = torch.where(self.commands[env_ids, :2] < 0, True, False)
        # ratio = self._commands[env_ids,1]/(self._commands[env_ids,0]+1e-8) #计算反正切以获得偏航角
        # gzero = torch.where(self._commands[env_ids] > 0, True, False)
        # lzero = torch.where(self._commands[env_ids] < 0, True, False)
        plus = lzero[:,0] * gzero[:,1] #第二象限 lzero=[TRue, False], gzero=[False, True], plus=True，其余象限为False
        minus = lzero[:,0] * lzero[:,1] #第三象限 lzero=[True, True], minus=True，其余象限为False
        offsets = torch.pi*plus - torch.pi*minus #第二象限加π，第三象限减π，其余象限不变
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1) #计算命令的偏航角，并加上象限修正，区分一与三、二与四象限，确保偏航角在[-π, π]范围内

        #重置机器人位置
        default_root_state = self.robot.data.default_root_state[env_ids] #获取默认根状态，位置和姿态
        # default_root_state[:, :3] += self.scene.env_origins[env_ids] #将默认根状态的位置部分加上环境原点位置，确保机器人在各自环境中的正确位置
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)#将更新后的根状态写入仿真环境，完成重置

        self._visualize_markers() #重置时可视化标记

def define_markers() -> VisualizationMarkers:
    markers_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5), #缩放尺寸
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)), #定义箭头颜色为青色
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), #定义箭头颜色为红色
            ),
        },
    )
    return VisualizationMarkers(cfg=markers_cfg)

# self.cfg.terrain.class_type(self.cfg.terrain) 在初始化地形对象时，会根据 num_envs 和 env_spacing 参数来生成每个环境的地形原点（env_origins）和分布。
# 如果这两个参数没有提前设置为场景实际的环境数量和间距，地形对象就会按照默认值（通常为1或很小的数）来分配原点数组，导致后续访问（如 env_origins[env_ids]）时索引超出范围，出现越界错误。