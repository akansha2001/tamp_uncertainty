# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/search_object.py --num_envs 1

"""


########################  SIMULATION  #################################
      
"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# add arguments to record videos
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import os
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractBelief,
    ActionSchema,
    StreamSchema,
    AliasStore,
    Belief,
    NoOp,
    Predicate,
    State,
    effect_from_execute_fn,
    Observation
)
import logging 
from tampura.symbolic import OBJ, Or, Atom, ForAll, Exists, OneOf, And, Not, Eq, Imply, negate
from tampura.policies.tampura_policy import TampuraPolicy
from tampura.config.config import get_default_config, setup_logger

import torch
import time

LOOK_CONFIDENCE = 0.8
GRASPS = [[0,1,0,0],[0,0.7,0.7,0]]
REGIONS={"r1":[0.5, 0, 0.055],"r2":[0.7, 0.1, 0.055],"r3":[0.5, 0.2, 0.055]}#,"robot_gripper":[0.0,0.0,0.0],"goal":[0.75, -0.15, 0.055]}
CONFIGURATIONS={"q1":[0.5, 0, 0.055],"q2":[0.7, 0.1, 0.055],"q3":[0.5, 0.2, 0.055]}

OBJECTS = {"object1":{"region":"r1","size":10,"pick_successes":[0.8,0.3]},"object2":{"region":"r2","size":4,"pick_successes":[0.3,0.8]},"object3":{"region":"r3","size":4,"pick_successes":[0.3,0.8]}}
ROBOT = {"configuration":"q1"}
TARGET_OBJECT = {"name":"object4","region":"r1","size":2,"pick_successes":[0.01,0.9]}


# Observation space
# Observation space
@dataclass
class EnvState(State):
    
    sim_env: Any
    desired_orientation: Any 
    pick_place_sm: Any 
    actions: Any

@dataclass 
class ObjectObservation(Observation):
    
    at: Dict[str,str]=field(default_factory=lambda: {})
    looked: List[str]=field(default_factory=lambda: [])
    

# Belief space
class ObjectsBelief(Belief):
    def __init__(self, at={}, looked=[]):
        
        self.at=at
        self.looked=looked

  

    def update(self, a, o, s):
        
        at=self.at
        looked=self.looked
        
        if a.name=="look":
            at=o.at
            looked=o.looked
        elif a.name=="pick"or a.name=="drop":
            at=o.at
        
        return ObjectsBelief(at=at,looked=looked) 

    def abstract(self, store: AliasStore):
        
        atoms = []
        
        
        
        for o in self.at.keys():
            atoms.append(Atom("at",[o,self.at[o]]))
        
        for q in self.looked:
            atoms.append(Atom("looked",[q]))
            
            
        return AbstractBelief(atoms)
    

    def vectorize(self):
        pass
    
# Sample function for stream schema
    
def sample_grasp(input_sym, store):
    
    o = input_sym[0]
    num_samp = input_sym[1]
    
    g = GRASPS[store.get(num_samp)]
    
    # TODO: make more sophisticated
    
    return g

    
# action simulators
def isClose(loc1,loc2,thresh=0.0):
    
    # if isinstance(loc1,list) and isinstance(loc2,list):
    #     loc1=np.array(loc1)
    #     loc2=np.array(loc2)
    #     return np.linalg.norm(loc1-loc2)<=thresh

    return torch.all(torch.linalg.norm(loc1[:,:3]-loc2[:,:3])<=thresh)

def forward_kinematics(q,store):
    
    for k,v in store.als_type.items():
        if v=="region":
            if torch.all(store.get(k) == store.get(q)):
                return k
            
    return "None"
    
# Action simulators
def look_execute_fn(a, b, s, store):
    
    if s is None: 
    
        q = a.args[0]
        r = a.args[1]
        
        at=b.at.copy()
        looked=b.looked.copy()
        
        size=0
        obj=None
        for o in OBJECTS.keys():
            if at.get(o) is None: # not already located, meaning it is at initial location
                
                if isClose(store.get(OBJECTS[o]["region"]),store.get(r)) and OBJECTS[o]["size"]>size:
                    obj=o
                    size=OBJECTS[o]["size"]
                    
        
                    
        if isClose(store.get(TARGET_OBJECT["region"]),store.get(r)) and TARGET_OBJECT["size"]>size:
                    obj=TARGET_OBJECT["name"]
                    size=TARGET_OBJECT["size"]
              
        # objects for which location is not known yet      
        unknown_locations = [o for o in OBJECTS.keys() if at.get(o) is None]
        
        if at.get(TARGET_OBJECT["name"]) is None:
            unknown_locations.append(TARGET_OBJECT["name"])
        # looking at a location, any of the yet unknown objects may be seen or NO object may be seen   
        obj=random.choice(unknown_locations+[None]) # to learn the MDP 
          
        
        if obj is None:
            looked = list(set(looked+[q]))
        else:
            at[obj]=r
        
                
        return State(), ObjectObservation(at=at,looked=looked)
    
    else: # execute fn
        
        env = s.sim_env
        pick_place_sm = s.pick_place_sm
        desired_orientation = s.desired_orientation
        actions = s.actions
    
        print("Looking in region ",a.args[1]," corresponding to robot configuration ",a.args[0])
        count = 0
        while True:
            dones = env.step(actions)[-2]
            
            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            
            # -- target object frame
            # desired_position = torch.tensor([store.get(a.args[0])],device=env.unwrapped.device)
            desired_position = store.get(a.args[1])
            desired_position_look = desired_position.clone()
            desired_position_look[:,2] += 0.2 # stop at some height above object location
            
            # advance state machine
            actions = pick_place_sm.compute_pick(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),                            
            )
            
            
            # print(pick_place_sm.sm_state)
            if torch.all(pick_place_sm.sm_state == 5.0*torch.ones_like(pick_place_sm.sm_state)):
                count += 1
                if count > 15: # TODO: check
                    break

            
        print("Look complete!")
        # rest
        pick_place_sm.rest()
        
        # actual look will require camera info
        
        at = b.at.copy()
        looked = b.looked.copy()
        
        # in the simulation, cannot be sure object is not knocked over to a new location
        # but problem was not designed to include this scenario!! 
        # to avoid knocking over the hidden object or dropping a cylindrical object (which cannot be picked up again)
        # the desired location for the pick is set to be slightly above the object
        # size=0
        # obj=None
        # for o in OBJECTS.keys():
        #     if at.get(o) is None: # not already located, meaning it is at initial location
                
        #         # -- object frame
        #         object_data: RigidObjectData = env.unwrapped.scene[o].data
        #         object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
                
        #         if isClose(object_position,desired_position,thresh=0.1) and OBJECTS[o]["size"]>size:
        #             obj=o
        #             size=OBJECTS[o]["size"]
                    
        # # -- object frame
        # object_data: RigidObjectData = env.unwrapped.scene[TARGET_OBJECT["name"]].data
        # object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
        
        # if isClose(object_position,desired_position,thresh=0.1) and TARGET_OBJECT["size"]>size:
            
        #     obj=TARGET_OBJECT["name"]
        #     size=TARGET_OBJECT["size"]
                    
                    
        # if obj is None:
        #     looked = list(set(looked+[a.args[0]])) # location is empty (may not be empty if object is knocked to this region in simulation)
        # else:
        #     at[obj]=a.args[1]
            
        # # ALTERNATELY 1
        # if obj is not None:
        #     at[obj]=a.args[1] # may have been knocked over to some other place - "looked"- cannot be guaranteed to be empty - but this scenario is not covered in the design of the problem!!!
        
        # ALTERNATELY 2
        # not a good way to do it!!!   but works  (takes care of object getting displaced)
        q = a.args[0]
        r = a.args[1]
        
        size=0
        obj=None
        for o in OBJECTS.keys():
            if at.get(o) is None: # not already located, meaning it is at initial location
                
                if isClose(store.get(OBJECTS[o]["region"]),store.get(r)) and OBJECTS[o]["size"]>size:
                    obj=o
                    size=OBJECTS[o]["size"]
                    
        
                    
        if isClose(store.get(TARGET_OBJECT["region"]),store.get(r)) and TARGET_OBJECT["size"]>size:
                    obj=TARGET_OBJECT["name"]
                    size=TARGET_OBJECT["size"]
                    
                    
        if obj is None:
            looked = list(set(looked+[q]))
        else:
            at[obj]=r
        
                
        return EnvState(sim_env=env,desired_orientation=desired_orientation,pick_place_sm=pick_place_sm,actions=actions), ObjectObservation(at=at,looked=looked)
        
        

def pick_execute_fn(a, b, s, store):
    
    if s is None: # effect fn
        
        o = a.args[0]
        g = a.args[1]
        
        at=b.at.copy()

        if o in OBJECTS.keys():
            pick_successes=OBJECTS[o]["pick_successes"]
        elif TARGET_OBJECT["name"]==o:
            pick_successes=TARGET_OBJECT["pick_successes"]
        
        # verify_effects
        pick_success = pick_successes[GRASPS.index(store.get(g))]
        if random.random() < pick_success: #TODO: modify with grasps
            at[o]="robot_gripper"
        
        return State(), ObjectObservation(at=at)
        
    else: # execute fn
        
        env = s.sim_env
        pick_place_sm = s.pick_place_sm
        desired_orientation = s.desired_orientation
        actions = s.actions
        
        print("Picking ",a.args[0])
        count = 0
        desired_orientation_pick=torch.tensor([store.get(a.args[1])]*env.num_envs, dtype=torch.float32,device=env.unwrapped.device) # grasp
        while True:
            dones = env.step(actions)[-2]
            
            
            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            
            # store approach stops above object : does not work: have to use current object position information
            # object_position=store.get(a.args[2]) 
            # alternately
            object_data: RigidObjectData = env.unwrapped.scene[a.args[0]].data # alternately use store.get(a.args[1]) to access location information
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            object_orientation = object_data.root_quat_w
            
            # -- target object frame
            # desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            desired_position = object_position.clone()
            desired_position[:,2] += 0.1
            
            # advance state machine
            actions = pick_place_sm.compute_pick(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation_pick], dim=-1),
                torch.cat([desired_position, desired_orientation_pick], dim=-1),                            
            )
            
            
            if torch.all(pick_place_sm.sm_state == 8.0*torch.ones_like(pick_place_sm.sm_state)):
                count += 1
                if count > 30:
                    break

            
        print(a.args[0]+" picked!")
        # rest
        pick_place_sm.rest()
        
        o = a.args[0]
        g = a.args[1]
        
        at=b.at.copy()
    
    
        if isClose(object_position,tcp_rest_position,thresh=0.1):
            at[o]="robot_gripper"

        return EnvState(sim_env=env,desired_orientation=desired_orientation,pick_place_sm=pick_place_sm,actions=actions), ObjectObservation(at=at)
    


def drop_execute_fn(a, b, s, store):
    
    if s is None: # effect fn
        
        o = a.args[0]
        g = a.args[1]

        at=b.at.copy()
        
        
        # verify_effects
        if o in OBJECTS.keys():
            pick_successes=OBJECTS[o]["pick_successes"]
        elif TARGET_OBJECT["name"]==o:
            pick_successes=TARGET_OBJECT["pick_successes"]
        
        # verify_effects
        pick_success = pick_successes[GRASPS.index(store.get(g))]
        if random.random() < pick_success: #TODO: modify with grasps
            at[o]="goal"
        
        return State(), ObjectObservation(at=at)
        
    else: # execute fn
        
        env = s.sim_env
        pick_place_sm = s.pick_place_sm
        desired_orientation = s.desired_orientation
        actions = s.actions
        
        
        print("Dropping ",a.args[0])
        count = 0
        desired_orientation_drop=torch.tensor([store.get(a.args[1])]*env.num_envs, dtype=torch.float32, device=env.unwrapped.device)
        while True:
            dones = env.step(actions)[-2]
            
            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- location
            location_position =store.get("goal")
           
            # advance state machine
            actions = pick_place_sm.compute_drop(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([location_position, desired_orientation_drop], dim=-1),
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
            )
                                    
            if torch.all(pick_place_sm.sm_state == 4.0*torch.ones_like(pick_place_sm.sm_state)):
                count += 1
                if count > 30:
                    break
            

        print(a.args[0]+" dropped!")
        # rest
        pick_place_sm.rest()

        o = a.args[0]
        g = a.args[1]
        
        at=b.at.copy()
    

    
        if isClose(location_position,tcp_rest_position,thresh=0.3): # why is tcp rest position pos+ori
            at[o]="goal"

        return EnvState(sim_env=env,desired_orientation=desired_orientation,pick_place_sm=pick_place_sm,actions=actions), ObjectObservation(at=at)
    
    
    


    
# closed world assumption: not explicitly True predicates are false

# Set up environment dynamics
class ToyDiscrete(TampuraEnv):
    def initialize(self,sim_env,state):
        
        self.state=state
        
        store = AliasStore()
        
        desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
        desired_orientation[:, 1] = 1.0
        
        # initialize store
        # set all object regions and robot configurations
        
        ridx=1
        for o in OBJECTS.keys():
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene[o].data
            object_position = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins
            flag=True # unique region
            for r in store.als_type.keys():
                if store.als_type[r]=="region":
                    if isClose(store.get(r),object_position,thresh=0.05):
                        OBJECTS[o]["region"]=r # modifying global variable!
                        flag=False
                        break
            if flag:                
                store.set("r"+str(ridx), object_position, "region")
                store.set("q"+str(ridx), object_position, "robot_configuration")
                reg=forward_kinematics("q"+str(ridx),store)
                store.certified.append(Atom("points_to",["q"+str(ridx),reg]))
                OBJECTS[o]["region"]="r"+str(ridx) # modifying global variable
                ridx+=1
                
        print(store)            

            
        # -- end-effector frame
        ee_frame_sensor = sim_env.unwrapped.scene["ee_frame"]
        tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins
        tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
        
        store.set("robot_gripper", tcp_rest_position, "region")
        
        goal_data: RigidObjectData = sim_env.unwrapped.scene["goal"].data
        goal_position = goal_data.root_pos_w - sim_env.unwrapped.scene.env_origins
        goal_position[0][2] += 0.2
        goal = torch.cat([goal_position, tcp_rest_orientation], dim=-1)
            
        store.set("goal",goal,"region")

            
        # object we are looking for
        store.set(TARGET_OBJECT["name"], TARGET_OBJECT["name"], "physical")
        
        # set objects 
        for o in OBJECTS.keys():
            store.set(o, o, "physical")
            
        # set number of times for sampling
        for i in range(len(GRASPS)):
            store.set("num_"+str(i),i,"num")
            
        
                   
        return ObjectsBelief(), store
    

    def get_problem_spec(self) -> ProblemSpec:
        predicates = [ 
            Predicate("at",["physical","region"]),
            Predicate("grasped",["physical","grasp"]),
            Predicate("points_to", ["robot_configuration","region"]),
            Predicate("looked", ["robot_configuration"]),
        ] 
        
        stream_schemas = [
         
            StreamSchema(
                name="sample-grasp",
                inputs=["?o1","?n1"],
                input_types=["physical","num"],
                output="?g1",
                output_type="grasp",
                certified=[Atom("grasped",["?o1","?g1"])], # generates a valid grasp g1 for o1
                sample_fn=sample_grasp                
            ),
          
        ]
        
        action_schemas = [
            
            ActionSchema(
                name="look",
                inputs=["?q1","?r1"],
                input_types=["robot_configuration","region"],
                preconditions=[Not(Exists(Atom("at",["?o","robot_gripper"]),["?o"],["physical"])), # robot gripper is free
                               Atom("points_to",["?q1","?r1"]), # ?q1 corresponds to ?r1
                               Not(Exists(Atom("at",["?o","?r1"]),["?o"],["physical"])), # no known object is in the region
                               Not(Atom("looked",["?q1"])), # has not been inspected already
                               ],
                
                verify_effects=[OneOf([Atom("at",[o,"?r1"])for o in OBJECTS.keys()]+[Atom("at",[TARGET_OBJECT["name"],"?r1"]),Atom("looked",["?q1"])])], 
                execute_fn=look_execute_fn,
                effects_fn=effect_from_execute_fn(look_execute_fn),
            
            ),

            ActionSchema(
                name="pick",
                inputs=["?o1","?g1","?r1"],
                input_types=["physical","grasp","region"],
                preconditions=[Not(Atom("at",["?o1","goal"])),
                               Atom("at",["?o1","?r1"]), # location of ?o1 is known
                               Atom("grasped",["?o1","?g1"]), # ?g1 is a valid grasp for ?o1
                               Not(Exists(Atom("at",["?o","robot_gripper"]),["?o"],["physical"])), # robot gripper is free
                               ], 
                verify_effects=[Atom("at",["?o1","robot_gripper"]),Not(Atom("at",["?o1","?r1"]))],
                execute_fn=pick_execute_fn,
                effects_fn=effect_from_execute_fn(pick_execute_fn),
            ),
            
            ActionSchema(
                name="drop",
                inputs=["?o1","?g1"],
                input_types=["physical","grasp"],
                preconditions=[
                               Atom("at",["?o1","robot_gripper"]), # robot is holding ?o1
                               Atom("grasped",["?o1","?g1"]), # ?g1 is a valid grasp for ?o1
                               Not(Atom("at",["?o1","goal"])), # current location is not the same as goal location
                               ], 
                verify_effects=[Atom("at",["?o1","goal"]),Not(Atom("at",["?o1","robot_gripper"]))], 
                execute_fn=drop_execute_fn,
                effects_fn=effect_from_execute_fn(drop_execute_fn),
            ),
            NoOp(),
        ]
        
        reward = Atom("at",[TARGET_OBJECT["name"],"robot_gripper"])
        
        spec = ProblemSpec(
            predicates=predicates,
            stream_schemas=stream_schemas,
            action_schemas=action_schemas,
            reward=reward,
        )

        return spec

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

    
class PickPlaceSmState:
    """States for the pick place state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_LOCATION = wp.constant(1)
    APPROACH_LOCATION = wp.constant(2)
    RELEASE_OBJECT = wp.constant(3)
    LIFT_EE = wp.constant(4)
    APPROACH_ABOVE_OBJECT = wp.constant(5)
    APPROACH_OBJECT = wp.constant(6)
    GRASP_OBJECT = wp.constant(7)
    LIFT_OBJECT = wp.constant(8)


class PickPlaceSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_LOCATION = wp.constant(0.5)
    APPROACH_LOCATION = wp.constant(0.6)
    RELEASE_OBJECT = wp.constant(0.3)
    LIFT_EE = wp.constant(1.0)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)



    
@wp.kernel
def infer_pick_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickPlaceSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
    
@wp.kernel
def infer_drop_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    location_pose: wp.array(dtype=wp.transform),
    des_final_ee_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickPlaceSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.APPROACH_ABOVE_LOCATION
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.APPROACH_ABOVE_LOCATION:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], location_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_LOCATION:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.APPROACH_LOCATION
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.APPROACH_LOCATION:
        des_ee_pose[tid] = location_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_LOCATION:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.RELEASE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = location_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.RELEASE_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.LIFT_EE
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceSmState.LIFT_EE:
        des_ee_pose[tid] = des_final_ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.LIFT_EE:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceSmState.LIFT_EE
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


    
class PickPlaceSm:
    """A simple state machine in a robot's task space to pick or place an object and lift the end effector.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST
    2. APPROACH_ABOVE_LOCATION 
    3. APPROACH_LOCATION 
    4. RELEASE_OBJECT 
    5. LIFT_EE 
    6. APPROACH_ABOVE_OBJECT
    7. APPROACH_OBJECT
    8. GRASP_OBJECT
    9. LIFT_OBJECT
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        
    def rest(self):
        # reset state machine
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        
        # convert to warp
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        
        

    
    def compute_pick(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_pick_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

    def compute_drop(self, ee_pose: torch.Tensor, location_pose: torch.Tensor, des_final_ee_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        location_pose = location_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_final_ee_pose = des_final_ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        location_pose_wp = wp.from_torch(location_pose.contiguous(), wp.transform)
        des_final_ee_pose_wp = wp.from_torch(des_final_ee_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_drop_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                location_pose_wp,
                des_final_ee_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
      
    log_dir = "/home/am/Videos/"  
    
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Franka-IK-Abs-v1",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Lift-Franka-IK-Abs-v1", cfg=env_cfg,render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "IsaacLab"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": False,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # reset environment at start
    env.reset()

     ############################  TAMPURA  ###################################
    # Initialize environment
    cfg = get_default_config(save_dir=os.getcwd())

    # Set some print options to print out abstract belief, action, observation, and reward
    cfg["print_options"] = "ab,a,o,r"
    cfg["flat_sample"] = True # disable progressive widening
    cfg["vis_graph"] = True
    cfg["batch_size"] = 10
    cfg["num_samples"] = 100  
    cfg["max_steps"] = 40
    cfg["num_skeletons"]=10
    
 
    

    ############################  TAMPURA  ###################################
    
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machines
    pick_place_sm = PickPlaceSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    # initial ee position (and final ee position after place)
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    ee_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    ee_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
    
    tampura_env = ToyDiscrete(config=cfg)
    
    state=EnvState(sim_env=env,desired_orientation=desired_orientation,pick_place_sm=pick_place_sm,actions=actions)
    
    b, store = tampura_env.initialize(sim_env=env,state=state)

    # Set up logger to print info
    setup_logger(cfg["save_dir"], logging.INFO)
    
    
    # Initialize the policy
    planner = TampuraPolicy(config = cfg, problem_spec = tampura_env.problem_spec)
    
    step=0
    time0 = time.time()
    time_taken=[]
    while simulation_app.is_running():
        with torch.inference_mode():
            
            while True:
                start_time=time.time()
                a_b = b.abstract(store)
                reward = tampura_env.problem_spec.get_reward(a_b, store)
                if reward > 0.0:
                    print("Goal state reached!")
                    break
                if step == cfg["max_steps"]:
                    print("FAILURE")
                    break

                logging.info("\n" + ("=" * 10) + "t=" + str(step) + ("=" * 10))
                if "s" in planner.print_options:
                    logging.info("State: " + str(s))
                if "b" in planner.print_options:
                    logging.info("Belief: " + str(b))
                if "ab" in planner.print_options:
                    logging.info("Abstract Belief: " + str(a_b))
                if "r" in planner.print_options:
                    logging.info("Reward: " + str(reward))

                action, info, store = planner.get_action(b, store)
                
                print("Time taken to plan for step "+str(step)+" : "+str(time.time()-start_time)+" seconds")

                if "a" in planner.print_options:
                    logging.info("Action: " + str(action))

                if action.name == "no-op":
                    bp = b
                    observation = None
                else:
                    observation = tampura_env.step(action, b, store)
                    bp = b.update(action, observation, store)

                    if planner.config["vis"]:
                        tampura_env.vis_updated_belief(bp, store)

                a_bp = bp.abstract(store)

                reward = tampura_env.problem_spec.get_reward(a_bp, store)
                if "o" in planner.print_options:
                    logging.info("Observation: " + str(observation))
                if "sp" in planner.print_options:
                    logging.info("Next State: " + str(tampura_env.state))
                if "bp" in planner.print_options:
                    logging.info("Next Belief: " + str(bp))
                if "abp" in planner.print_options:
                    logging.info("Next Abstract Belief: " + str(a_bp))
                if "rp" in planner.print_options:
                    logging.info("Next Reward: " + str(reward))

                # update the belief
                b = bp
                step += 1
                end_time=time.time()
                print("Time taken for planning and simulation for step "+str(step)+" : "+str(end_time-start_time)+" seconds")

            logging.info("=" * 20)

            tampura_env.wrapup()
            
        break
    
    total_time=time.time()-time0
    print("Total time: "+str(total_time)+" seconds")                   
    print("Plan executed!")
    env.close()
    
                
                
                



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()