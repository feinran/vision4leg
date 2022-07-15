# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from vision4leg.envs.locomotion_gym_env_with_rich_information import LocomotionGymEnv
from scheduler import Scheduler

class BaseTask():
    """Default task."""

    def __init__(self):
        """Initializes the task."""
        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()

    def done(self, env):
        """Checks if the episode is over.

        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        
        return rot_mat[-1] < 0.85

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        return self.current_base_pos[0] - self.last_base_pos[0]
    
    @property
    def move_reward(self):
        return None
    
    @property
    def align_reward(self):
        return None
    
    @property
    def speed_reward(self):
        return None
    
    @property
    def energy_reward(self):
        return None
    
    @property
    def l_move(self):
        return None

    @property
    def l_align(self):
        return None

    @property
    def l_speed(self):
        return None


class EnergyTask(BaseTask):
    """Penalize energy consumption"""
    def __init__(self, target_speed, alpha_1, alpha_2, alpha_3):
        super().__init__()
        self.energy_consumption = 0
        self.base_velocity = np.zeros(3)
        self.rpy = (0, 0, 0)

        self.target_speed = target_speed
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alive = target_speed * alpha_3

    def reset(self, env):
        super().reset(env)
        self.energy_consumption = 0
        self.base_velocity = np.zeros(3)
        self.rpy = (0, 0, 0)

    def update(self, env):
        super().update(env)
        self.energy_consumption = env.robot.GetEnergyConsumptionPerControlStep()
        self.base_velocity = env.robot.GetBaseVelocity()
        self.rpy = env.robot.GetTrueBaseRollPitchYaw()

    def reward(self, env):
        """Get the reward without side effects."""
        del env

        forward = -self.alpha_2 * abs(self.target_speed - self.base_velocity[0])
        forward -= self.base_velocity[1] * self.base_velocity[1]
        forward -= self.rpy[2] * self.rpy[2]
        return forward + self.alpha_1 * self.energy_consumption + self.alive


class DirectionTask(BaseTask):
    """Returns reward depending on the direction"""
    def __init__(self, 
                 energy_enable: bool = False,
                 similarity_func_name: str = "dot", 
                 l_move: float = 0.6,
                 l_align: float = 0.6,
                 move_decay: float = 1,
                 align_decay: float = 1,
                 l_move_limit: float = 1,
                 l_align_limit: float = 1,
                 w_move: float = 1,
                 w_align: float = 1,
                 w_energy: float = 1,
                 w_alive: float = 0):
        super().__init__()
        self.similarity_func_name = similarity_func_name
        
        # weights
        self._w_move = float(w_move)
        self._w_align = float(w_align)
        self._w_energy = float(w_energy)
        self._w_alive = float(w_alive)
        
        # log data
        self._move_dot = 0 
        self._align_dot = 0
        self._energy = 0
        
        # additional energy option
        self._energy_enable = energy_enable
        
        self._l_move_scheduler = Scheduler(len_history=8,
                                            value_init=l_move,
                                            decay=move_decay,
                                            upper_limit=l_move_limit,
                                            lower_limit=-l_move_limit)
        self._l_align_scheduler = Scheduler(len_history=8,
                                            value_init=l_align,
                                            decay=align_decay,
                                            upper_limit=l_align_limit,
                                            lower_limit=-l_align_limit)
        
    
    def similarity_func(self, v1, v2, l):
        if self.similarity_func_name.lower() == "rbf":
            rbf_value = np.exp2(- np.linalg.norm(v2 - v1)**2 / (2 * l**2))
            return 2 * rbf_value -1  # to make it comparable to dot_prod

        elif self.similarity_func_name.lower() == "ln":
            return -np.log(6 * np.abs(np.linalg.norm(v2 - v1)) + 0.05) / np.log(20) + 0.5

        elif self.similarity_func_name.lower() == "dot":
            return np.dot(v1, v2)
        
    def reset(self, env):
        self._l_move_scheduler.roll_out_end()
        self._l_align_scheduler.roll_out_end()
        
    def reward(self, env: LocomotionGymEnv):
        """
        Get the reward without side effects.
        """

        # get sensor data
        direction_sensor = env.sensor_by_name("Direction")
        dir = direction_sensor.direction
        
        # how far the robot has moved
        change = np.array(self.current_base_pos[:2]) - np.array(self.last_base_pos[:2])
        magnitude = np.linalg.norm(change)
        change = change / magnitude  # normalized move direction

        rot_quat = env.robot.GetTrueBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        forward = np.array([rot_mat[i] for i in [0, 3]])  # direction where the robot is looking at
        
        # energy consumption
        energy_consumption = None
        if self._energy_enable:
            energy_consumption = env.robot.GetEnergyConsumptionPerControlStep()
        
        # render debug lines
        if env.rendering_enabled:
            debug_lines(self.current_base_pos, env, forward, None, direction_sensor.target_direction)

        dir = dir / np.linalg.norm(dir)  # normalized target direction
        movement_dot = self.similarity_func(dir, change, self._l_move_scheduler.value)
        movement_reward = movement_dot * magnitude
        alignment_dot = self.similarity_func(dir, forward, self._l_align_scheduler.value)
        alignment_reward = alignment_dot * magnitude
        energy_reward = - energy_consumption if energy_consumption is not None else 0
        
        # provide logging data
        self._move_dot = movement_dot
        self._align_dot = alignment_dot
        self._energy = energy_consumption
        
        # update scheduler
        self._l_move_scheduler.update(last_performace=movement_dot)
        self._l_align_scheduler.update(last_performace=alignment_dot)
        
        return self._w_move * movement_reward + \
                self._w_align * alignment_reward + \
                self._w_energy * energy_reward + \
                self._w_alive
    
    @property
    def move_reward(self):
        return self._move_dot
    
    @property
    def align_reward(self):
        return self._align_dot
    
    @property
    def energy_reward(self):
        return self._energy    
    
    @property
    def l_move(self):
        return self._l_move_scheduler.value

    @property
    def l_align(self):
        return self._l_align_scheduler.value


class DirectionSpeedTask(BaseTask):
    """Returns reward depending on the direction"""
    def __init__(self, 
                 energy_enable: bool = False,
                 similarity_func_name: str = "rbf", 
                 l_move: float = 0.6, 
                 l_align: float = 0.6, 
                 l_speed: float = 0.6,
                 move_decay: float = 0.96,
                 align_decay: float = 0.96,
                 speed_decay: float = 0.96,
                 l_move_limit: float = 0.90,
                 l_align_limit: float = 0.90,
                 l_speed_limit: float = 0.90,
                 w_move: float = 1,
                 w_align: float = 1,
                 w_speed: float = 1,
                 w_energy: float = 0.1,
                 w_motor_limit_penalty: float = 0,      # doesn't work here yet
                 w_nofoot_contacts_penalty: float = 0,  # doesn't work here yet
                 w_alive: float = 0,
                 motor_limit_epsilon: float = -0.07):
        super().__init__()
        self.similarity_func_name = similarity_func_name
       
        # weights
        self._w_move = float(w_move)
        self._w_align = float(w_align)
        self._w_speed = float(w_speed)
        self._w_energy = float(w_energy)
        self._w_motor_limit_penalty = -float(w_motor_limit_penalty)
        self._w_nofoot_contacts_penalty = -float(w_nofoot_contacts_penalty)
        self._w_alive = float(w_alive)
        
        # log data
        self._move_reward = 0
        self._align_reward = 0
        self._speed_reward = 0
        self._energy = 0
        
        # additional energy option
        self._energy_enable = energy_enable
        
        history_length = 15
        self._l_move_scheduler = Scheduler(len_history=history_length,
                                            value_init=l_move,
                                            decay=move_decay,
                                            upper_limit=l_move_limit,
                                            lower_limit=-l_move_limit)
        self._l_align_scheduler = Scheduler(len_history=history_length,
                                            value_init=l_align,
                                            decay=align_decay,
                                            upper_limit=l_align_limit,
                                            lower_limit=-l_align_limit)
        self._l_speed_scheduler = Scheduler(len_history=history_length,
                                            value_init=l_speed,
                                            decay=speed_decay,
                                            upper_limit=l_speed_limit,
                                            lower_limit=-l_speed_limit)
        
        self.motor_limit_epsilon = motor_limit_epsilon
        
    def similarity_func(self, v1, v2, l):
        if self.similarity_func_name.lower() == "rbf":
            rbf_value = np.exp2(- np.linalg.norm(v2 - v1)**2 / (2 * l**2))  # returns are in range [0, 1]
            return 2 * rbf_value -1  # to make it comparable to dot_prod

        elif self.similarity_func_name.lower() == "ln":
            return -np.log(6 * np.abs(np.linalg.norm(v2 - v1)) + 0.05) / np.log(20) + 0.5

        elif self.similarity_func_name.lower() == "dot":
            return np.dot(v1, v2)
    
    @staticmethod
    def get_limit_violations(joint_angles, upper_limit, lower_limit):
        upper_bool = joint_angles > upper_limit
        lower_bool = joint_angles < lower_limit
        
        return sum(upper_bool + lower_bool)
    
    def reset(self, env):
        self._l_move_scheduler.roll_out_end()
        self._l_align_scheduler.roll_out_end()
        self._l_speed_scheduler.roll_out_end()
         
    def reward(self, env: LocomotionGymEnv):
        """
        Get the reward without side effects.
        """

        ## MOVED! get direction sensor data
        # direction_sensor = env.sensor_by_name("Direction")
        # dir = direction_sensor.direction
        
        ## MOVED! get speed sensor data
        # speed_sensor = env.sensor_by_name("Speed")
        # current_speed = speed_sensor.current_speed
        # target_speed = speed_sensor.target_speed
        
        # get rot matrix to convert from global coordinates to local coordinates
        rot_quat = env.robot.GetTrueBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        forward = np.array([rot_mat[i] for i in [0, 3]])  # direction where the robot is looking at

        # get direction sensor data
        dir = np.array([forward[0], -forward[1]])

        # DISABLED! get number of contacts that are not foot contacts
        nofoot_contacts = 0  # env.robot.nofoot_contacts

        ## DISABLED! calculate number of hip motors that exceed ther position limit  
        # epsilon = 0.07
        # motor_angles = np.array(env.robot.GetTrueMotorAngles())
        
        # hip_motor_angles = motor_angles[::3]
        # upper_limit = UPPER_THRESHOLD_HIP - epsilon
        # lower_limit = LOWER_THRESHOLD_HIP + epsilon
        # hip_motor_limits = self.get_limit_violations(hip_motor_angles, upper_limit, lower_limit)
        
        # upper_motor_angles = motor_angles[1::3]
        # upper_limit = UPPER_THRESHOLD_UPPER - epsilon
        # lower_limit = LOWER_THRESHOLD_UPPER + epsilon
        # upper_motor_limits = self.get_limit_violations(upper_motor_angles, upper_limit, lower_limit)
        
        # lower_motor_angles = motor_angles[2::3]
        # upper_limit = UPPER_THRESHOLD_LOWER - epsilon
        # lower_limit = LOWER_THRESHOLD_LOWER + epsilon
        # lower_motor_limits = self.get_limit_violations(lower_motor_angles, upper_limit, lower_limit)
        
        all_motor_violations = 0  # hip_motor_limits + upper_motor_limits + lower_motor_limits
        
        # print("hitted limits: ", np.array([hip_motor_limits, upper_motor_limits, lower_motor_limits]))
        
        # energy consumption
        energy_consumption = None
        if self._energy_enable:
            energy_consumption = env.robot.GetEnergyConsumptionPerControlStep()
        
        ## render debug lines 
        # if env.rendering_enabled:
        #     debug_lines(self.current_base_pos, env, forward, None, direction_sensor.target_direction)

        # get base velocity in local frame
        rot_mat = np.reshape(np.array(rot_mat), (3, 3))
        global_velocity = np.array(env.robot.GetBaseVelocity())
        local_velocity = np.matmul(rot_mat, global_velocity)[:-1]
        local_movement_dir = local_velocity / np.linalg.norm(local_velocity)

        # get speed sensor data
        current_speed = np.linalg.norm(global_velocity)
        target_speed = 1
        
        # compute rewards
        movement_dot = self.similarity_func(dir, local_movement_dir, self._l_move_scheduler.value)
        alignment_dot = self.similarity_func(dir, np.array([1, 0]), self._l_align_scheduler.value)
        speed_reward = self.similarity_func(current_speed, target_speed, self._l_speed_scheduler.value)
        energy_reward = - energy_consumption if energy_consumption is not None else 0
        
        # provide logging data
        self._move_dot = movement_dot
        self._align_dot = alignment_dot
        self._speed_dot = speed_reward
        self._energy = energy_consumption
        
        # update scheduler
        self._l_move_scheduler.update(last_performace=movement_dot)
        self._l_align_scheduler.update(last_performace=alignment_dot)
        self._l_speed_scheduler.update(last_performace=speed_reward)
        
        return self._w_move * movement_dot + \
                self._w_align * alignment_dot + \
                self._w_speed * speed_reward + \
                self._w_energy * energy_reward + \
                self._w_motor_limit_penalty * all_motor_violations + \
                self._w_nofoot_contacts_penalty * nofoot_contacts + \
                self._w_alive
    
    @property
    def move_reward(self):
        return self._move_dot
    
    @property
    def align_reward(self):
        return self._align_dot
    
    @property
    def speed_reward(self):
        return self._speed_dot
    
    @property
    def energy_reward(self):
        return self._energy

    @property
    def l_move(self):
        return self._l_move_scheduler.value

    @property
    def l_align(self):
        return self._l_align_scheduler.value

    @property
    def l_speed(self):
        return self._l_speed_scheduler.value


class DirectionSpeedTaskOld(BaseTask):
    def __init__(self):
        super().__init__()
        self.base_velocity = np.zeros(3)

    def reset(self, env):
        super().reset(env)
        self.base_velocity = env.robot.GetBaseVelocity()

    def update(self, env):
        super().update(env)
        self.base_velocity = env.robot.GetBaseVelocity()

    """Returns reward depending on the direction"""
    def reward(self, env: LocomotionGymEnv):
        """
        Get the reward without side effects.
        """

        # get sensor data
        direction_sensor = env.sensor_by_name("Direction")
        dir = direction_sensor.direction 
        speed = direction_sensor.speed
        velocity = np.zeros(3)
        
        # how far the robot has moved
        change = np.array(self.current_base_pos[:2]) - np.array(self.last_base_pos[:2])
        magnitude = np.linalg.norm(change)
        change = change / magnitude  # normalized move direction

        rot_quat = env.robot.GetTrueBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        forward = np.array([rot_mat[i] for i in [0, 3]])  # direction where the robot is looking at

        if speed is not None:
            velocity = np.array(env.robot.GetBaseVelocity()[:2])
            magnitude = speed - np.linalg.norm(dir - velocity)
            
        if env.rendering_enabled:
            self.debug_lines(self.current_base_pos, env, forward, velocity, direction_sensor.target_direction)

        dir = dir / np.linalg.norm(dir)  # normalized target direction
        movement_dot = np.dot(dir, change)
        movement_reward = np.sign(movement_dot) * magnitude * movement_dot * movement_dot
        alignment_dot = np.dot(dir, forward)
        alignment_reward = np.sign(alignment_dot) * magnitude * alignment_dot * alignment_dot

        if speed is not None and magnitude < 0:
            movement_reward = -abs(movement_reward)
            alignment_reward = -abs(alignment_reward)

        return movement_reward + alignment_reward
    


def debug_lines(current_base_pos, env, forward, velocity = None, dir = None):
    env.pybullet_client.addUserDebugLine(
        current_base_pos,
        current_base_pos + np.append(dir, 0) * 2,
        lineColorRGB=[0, 0, 1],
        lineWidth=2.0,
        lifeTime=0.005)

    env.pybullet_client.addUserDebugLine(
        current_base_pos,
        current_base_pos + np.append(forward, 0) * 2,
        lineColorRGB=[1, 0, 0],
        lineWidth=2.0,
        lifeTime=0.005)

    if velocity is not None:
        env.pybullet_client.addUserDebugLine(
            current_base_pos,
            current_base_pos + np.append(velocity, 0) * 2,
            lineColorRGB=[0, 1, 0],
            lineWidth=2.0,
            lifeTime=0.005)

