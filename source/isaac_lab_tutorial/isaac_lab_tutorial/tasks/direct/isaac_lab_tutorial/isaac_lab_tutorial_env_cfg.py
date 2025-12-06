# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG #导入机器人配置
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCasterCfg, patterns


@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2 #控制仿真与控制频率的比值，decimation=2表示仿真频率为120Hz，控制频率为60Hz
    episode_length_s = 20
    # - spaces definition
    action_space = 2
    # observation_space = 9 #世界速度：线速度（vx, vy, vz)和角速度（wx, wy, wz）；命令向量（cx, cy, cw）
    # observation_space = 3 #机器人前进方向与命令方向的点积、机器人前进方向与命令方向的叉积在z轴方向的分量、机器人质心线速度在本体x轴方向的分量
    observation_space = 5
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground", #USD stage中创建地形的根路径
    #     terrain_type="generator", #“程序生成器”生成多个子地形网络
    #     terrain_generator=ROUGH_TERRAINS_CFG, #每个子地形(size)8*8m，子地形行列数(num_rows, num_cols)10*20，子地形边界外延(border_width)20m
    #     max_init_terrain_level=5, #课程难度层级，为None，默认max_init_level=num_rows-1；设置为5意味着初始难度层级会从0～5之间随机抽取（若 num_rows > 6）
    #     collision_group=-1, #被设置为“与环境实例发生碰撞”的全局路径（例如 ground 要与所有 env 中的机器人发生碰撞，因此常设为 -1）
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply", #两个接触体有不同摩擦系数时，合成策略采用乘法（常见选项：average, min, max, multiply 等）。multiply 会把两个摩擦值相乘，结果通常更小/更大取决于值
    #         restitution_combine_mode="multiply", #弹性系数合成策略采用乘法
    #         static_friction=1.0, #静摩擦系数
    #         dynamic_friction=1.0, #动摩擦系数
    #     ),
    #     debug_vis=False, #是否创建并显示 terrain origins（子地形原点 / env spawn 点）等调试标记
    # )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    dof_name = ["left_wheel_joint", "right_wheel_joint"]
