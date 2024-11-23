# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import class_uncertainty_ik_abs_env_cfg, search_object_ik_abs_env_cfg



##
# Inverse Kinematics - Absolute Pose Control
##



gym.register(
    id="Isaac-Lift-Franka-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": class_uncertainty_ik_abs_env_cfg.FrankaLiftEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Franka-IK-Abs-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": search_object_ik_abs_env_cfg.FrankaLiftEnvCfg,
    },
    disable_env_checker=True,
)

