# Uncertainty-aware TAMP
Short description of the project...
## Status
List example files, and what they do. What is the current state of the project? What are the TODOs. Some videos and demonstration of the current state. 

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