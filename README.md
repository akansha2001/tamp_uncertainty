# Uncertainty and Risk Aware Task and Motion Planning
While Integrated Task and Motion Planning (TAMP) (Garrett et. al., [2020](https://arxiv.org/pdf/2010.01083)) offers a valuable approach to generalizable long-horizon robotic manipulation and navigation problems, the typical TAMP formulation assumes full observability and deterministic action effects. These assumptions limit the ability of the planner to gather inforamtion and make decisions that are risk-aware. Curtis et. al. [2024](https://arxiv.org/pdf/2403.10454) present TAMP with Uncertainty and Risk Awareness (TAMPURA) as an efficient way to solve long-horizon planning problems with initial-state and action outcome uncertainty. TAMPURA uses sequences of controllers for short-time manipulation tasks for long-horizon planning in a deterministic fashion. It extends TAMP with partial observability, uncertainty and a coarse knowledge of the controllers' preconditions and effects. The main algorithm is illustrated below.
![TAMPURA](/media/TAMPURA_fig3.png)
Using “templates” that define preconditions and effects for each operator, the symbolic planner solves for multiple all-outcomes determinized plans to the goal abstract belief state. These plans are then evaluated using a mental simulation to approximately learn the transition and reward models for the MDP. The MDP is then solved to obtain a policy which takes belief states as inputs and outputs a suitable controller.

## Status
Two TAMPURA scenarios from Curtis et. al. [2024](https://arxiv.org/pdf/2403.10454) have been implemented and simulated on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html). The simulations are shown below.
### Class Uncertainty
A [robot arm](https://robodk.com/robot/Franka/Emika-Panda) is mounted to a table with 4 objects placed in front of it, with at least one bowl in the scene. The robot must place all objects of a certain class (here, green blocks) in the bowl. Classification noise is added to ground truth labels to mimic the confidence scores typically returned by object
The agent can gain more certainty about an object category by inspecting the object more closely with a wrist mounted camera (the use of a camera is ommitted in this implementation and the agent gets the class from the environment). A reasonable
strategy is to closely inspect objects and stably grasp and place them in the bowl. The planner has access to the following controllers:
Pick(?o ?g ?r), Drop(?o ?g ?r), Inspect(?o),
for objects o, grasps g, and regions on the table r. 
This is done in a closed loop fashion, i.e., if the agent believes that the object has fallen from the gripper, it will repeat the Pick action.

*(In this example, the stability of grasps is ommitted. It is implemented in the search object scenario.)*

<video controls width="640">
  <source src="https://akansha2001.github.io/tamp_uncertainty/media/class_uncertainty1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


### Searching for an object in a cluttered environment 

## Installation notes

Git clone the following repositories in root of this repository: isaac lab, tampura.
    
```bash
git clone https://github.com/isaac-sim/IsaacLab.git 
git clone https://github.com/aidan-curtis/tampura.git
```

You may decide to keep the tampura source code as part of this repo if you plan to make changes to it.

Build the symbolic solver for tampura. 
    
```bash
cd tampura/third_party/symk
python build.py
```

Create the virtual environment and install the dependencies. This is automatically done by using [uv python](https://docs.astral.sh/uv/getting-started/installation/) package manager.

```bash
cd <path to repository root>
uv venv
source .venv/bin/activate
uv sync
```

This should have installed isaac sim, tampura and isaac lab. Isaac lab is specially problematic, so consider installing it manually instead of using uv sync if it gives problems (using the instructions in the isaac lab repository for python env installation).

To add more dependencies, just run `uv add <package>` (instead of `pip install <package>`). You can also add them by hand in the `pyproject.toml`. 
More information about dependencies specification can be found [here](https://docs.astral.sh/uv/concepts/dependencies/).

## TAMPURA files

The python files required for the two TAMPURA task scenarios are available in the `tamp_uncertainty` directory. These must be placed in the appropriate location in the local `IsaacLab` directory to run them. The following instructions are for the class uncertainty task and are similar to the search for object in clutter task with the file names changed.

In the terminal, working from the `tamp_uncertainty` directory, set the following environment variables

```bash
SM_PATH=../IsaacLab/source/standalone/environments/state_machine/
FRANKA_CFG_PATH=../IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/lift/config/franka/
ENV_CFG_PATH=../IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/lift/
```

Now, copy the files to the suitable locations

```bash
cp __init__.py ${FRANKA_CFG_PATH}
cp class_uncertainty_ik_abs_env_cfg ${FRANKA_CFG_PATH}
cp class_uncertainty_joint_pos_env_cfg ${FRANKA_CFG_PATH}
cp class_uncertainty.py ${SM_PATH}
cp tampura_env_cfg.py ${ENV_CFG_PATH}
```

The scripts should be ready for execution. To run `class_uncertainty.py`, type in the following in the terminal

```bash
python /home/am/tamp_uncertainty/IsaacLab/source/standalone/environments/state_machine/class_uncertainty.py --num_envs 1 
```