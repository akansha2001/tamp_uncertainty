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
uv sync
```

This should have installed isaac sim, tampura and isaac lab. Isaac lab is specially problematic, so consider installing it manually instead of using uv sync if it gives problems (using the instructions in the isaac lab repository for python env installation).

To add more dependencies, just run `uv add <package>` (instead of `pip install <package>`). You can also add them by hand in the `pyproject.toml`. 
More information about dependencies specification can be found [here](https://docs.astral.sh/uv/concepts/dependencies/).