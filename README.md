# ur3e_peg_in_hole

## Required Library

If manipulator-mujoco library is not installed, added it from the submodule
```bash
git submodule update --init --recursive
cd Manipulator-mujoco
pip install -e .
```

## Additional Installation
```bash
pip install mujoco
pip install spatialmath-python
pip install open3d
```

## Running the simulation
```bash
cd scripts
python ur3e_peg_in_hole_test.py
python ur3e_assembly_test.py
```
