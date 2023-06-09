#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.registration import register
import gym

# Relax the robot
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'kitchen_relax-v1' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

register(
    id='kitchen_relax-v1',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxV1',
    max_episode_steps=280,
)