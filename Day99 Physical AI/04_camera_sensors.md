# Camera နှင့် Sensors

## အခန်း ၁ - Camera ထည့်ခြင်း

### ၁.၁ RGB Camera ထည့်ရန်

File: `rgb_camera.py`

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils

# Setup simulation
cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

# Ground plane
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Cube (ကြည့်ဖို့ object တစ်ခု)
cfg_cube = sim_utils.CuboidCfg(
    size=(0.5, 0.5, 0.5),
    spawn=sim_utils.SpawnCfg(pos=(2.0, 0.0, 0.5)),
)
cfg_cube.func("/World/cube", cfg_cube)

# Camera configuration
camera_cfg = CameraCfg(
    prim_path="/World/camera",
    update_period=0.1,  # 0.1 second ကြာလျှင် update
    height=480,
    width=640,
    data_types=["rgb"],  # RGB data သာ
    spawn=sim_utils.SpawnCfg(
        pos=(0.0, 0.0, 1.0),  # Camera position
        rot=(1.0, 0.0, 0.0, 0.0),  # Camera rotation
    ),
)

# Create camera
camera = Camera(camera_cfg)

sim.reset()

# Simulation loop
for i in range(1000):
    sim.step()
    camera.update(sim.dt)
    
    # Get camera data
    if camera.data.output is not None:
        rgb_data = camera.data.output["rgb"]
        print(f"Step {i}: RGB shape = {rgb_data.shape}")

simulation_app.close()
```

---

## အခန်း ၂ - Depth Camera

### ၂.၁ Depth Data ရယူရန်

File: `depth_camera.py`

```python
import torch
import numpy as np
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Multiple cubes at different distances
for i in range(3):
    cfg_cube = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        spawn=sim_utils.SpawnCfg(pos=(1.0 + i * 0.5, 0.0, 0.5)),
    )
    cfg_cube.func(f"/World/cube_{i}", cfg_cube)

# Depth camera
camera_cfg = CameraCfg(
    prim_path="/World/depth_camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["distance_to_camera"],  # Depth data
    spawn=sim_utils.SpawnCfg(pos=(0.0, 0.0, 1.0)),
)

camera = Camera(camera_cfg)
sim.reset()

for i in range(500):
    sim.step()
    camera.update(sim.dt)
    
    if camera.data.output is not None:
        depth = camera.data.output["distance_to_camera"]
        print(f"Depth range: {depth.min():.2f} to {depth.max():.2f} meters")

simulation_app.close()
```

---

## အခန်း ၃ - Camera ကို Robot မှာ တပ်ရန်

### ၃.၁ Robot's Eye View

File: `robot_camera.py`

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Franka robot
robot_cfg = FRANKA_PANDA_CFG.copy()
robot_cfg.spawn.func("/World/Franka", robot_cfg.spawn)
robot = Articulation(robot_cfg)

# Object to look at
cfg_cube = sim_utils.CuboidCfg(
    size=(0.1, 0.1, 0.1),
    spawn=sim_utils.SpawnCfg(pos=(0.5, 0.0, 0.5)),
)
cfg_cube.func("/World/target_cube", cfg_cube)

# Camera attached to robot end-effector
camera_cfg = CameraCfg(
    prim_path="/World/Franka/panda_hand/camera",  # Robot hand မှာ တပ်ပါ
    update_period=0.1,
    height=240,
    width=320,
    data_types=["rgb"],
    spawn=sim_utils.SpawnCfg(
        pos=(0.0, 0.0, 0.1),  # Hand ရဲ့ အပေါ်မှာ
    ),
)

camera = Camera(camera_cfg)
sim.reset()

# Move robot and capture images
target_pos = torch.tensor([[0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.0]])

for i in range(1000):
    robot.set_joint_position_target(target_pos)
    
    sim.step()
    robot.update(sim.dt)
    camera.update(sim.dt)
    
    if i % 10 == 0 and camera.data.output is not None:
        print(f"Step {i}: Camera sees object from robot's perspective")

simulation_app.close()
```

---

## အခန်း ၄ - Save Images

### ၄.၁ Camera Images ကို Save လုပ်ရန်

File: `save_images.py`

```python
import torch
import numpy as np
from PIL import Image
import os
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils

# Create output directory
os.makedirs("camera_output", exist_ok=True)

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Colorful cubes
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue
for i, color in enumerate(colors):
    cfg_cube = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        spawn=sim_utils.SpawnCfg(pos=(1.5, -0.5 + i * 0.5, 0.3)),
    )
    cfg_cube.func(f"/World/cube_{i}", cfg_cube)

# Camera
camera_cfg = CameraCfg(
    prim_path="/World/camera",
    update_period=0.5,  # 0.5 second တစ်ခါ capture
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.SpawnCfg(pos=(0.0, 0.0, 1.0)),
)

camera = Camera(camera_cfg)
sim.reset()

image_count = 0

for i in range(500):
    sim.step()
    camera.update(sim.dt)
    
    # Save every 50 steps
    if i % 50 == 0 and camera.data.output is not None:
        rgb_data = camera.data.output["rgb"][0].cpu().numpy()
        
        # Convert to uint8
        rgb_image = (rgb_data * 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(rgb_image)
        img.save(f"camera_output/frame_{image_count:04d}.png")
        
        print(f"Saved image {image_count}")
        image_count += 1

simulation_app.close()
print(f"Total images saved: {image_count}")
```

**Run လုပ်ပြီးရင်:**
```bash
ls camera_output/  # Images တွေ ကြည့်ရန်
```

---

## အခန်း ၅ - IMU Sensor (ဂျိုင်ရို/အိုက်မီယူ)

### ၅.၁ Robot ရဲ့ Orientation ဖတ်ရန်

File: `imu_sensor.py`

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import Imu, ImuCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import ANYMAL_C_CFG  # Quadruped robot

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Quadruped robot (လေးခြေမြင်း)
robot_cfg = ANYMAL_C_CFG.copy()
robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn)
robot = Articulation(robot_cfg)

# IMU sensor
imu_cfg = ImuCfg(
    prim_path="/World/Robot/base",  # Robot body မှာ တပ်ပါ
    update_period=0.01,
)

imu = Imu(imu_cfg)
sim.reset()

for i in range(1000):
    sim.step()
    robot.update(sim.dt)
    imu.update(sim.dt)
    
    if i % 100 == 0:
        # Read IMU data
        acceleration = imu.data.lin_acc
        angular_vel = imu.data.ang_vel
        
        print(f"Step {i}:")
        print(f"  Acceleration: {acceleration}")
        print(f"  Angular velocity: {angular_vel}")

simulation_app.close()
```

---

## အခန်း ၆ - Multi-Camera Setup

### ၆.၁ ပတ်ဝန်းကျင် ၃၆၀ ဒီဂရီ ကြည့်ရန်

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils
import math

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Central object
cfg_cube = sim_utils.CuboidCfg(
    size=(0.5, 0.5, 0.5),
    spawn=sim_utils.SpawnCfg(pos=(0.0, 0.0, 0.5)),
)
cfg_cube.func("/World/center_cube", cfg_cube)

# 4 cameras around the object (4 ဘက်က ကင်မရာ)
cameras = []
num_cameras = 4

for i in range(num_cameras):
    angle = (i / num_cameras) * 2 * math.pi
    x = 2.0 * math.cos(angle)
    y = 2.0 * math.sin(angle)
    
    camera_cfg = CameraCfg(
        prim_path=f"/World/camera_{i}",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.SpawnCfg(pos=(x, y, 1.0)),
    )
    
    camera = Camera(camera_cfg)
    cameras.append(camera)
    print(f"Camera {i} at position ({x:.2f}, {y:.2f}, 1.0)")

sim.reset()

for i in range(500):
    sim.step()
    
    for cam_idx, camera in enumerate(cameras):
        camera.update(sim.dt)
        
        if i % 50 == 0 and camera.data.output is not None:
            print(f"Camera {cam_idx} capturing frame at step {i}")

simulation_app.close()
```

---

## အကျဉ်းချုပ်

ဒီ lesson မှာ လေ့လာခဲ့တာတွေ:
- ✅ RGB Camera ထည့်နည်း
- ✅ Depth Camera သုံးနည်း
- ✅ Robot မှာ Camera တပ်နည်း
- ✅ Images save လုပ်နည်း
- ✅ IMU sensor သုံးနည်း
- ✅ Multiple cameras setup လုပ်နည်း

**နောက်ထပ်:** `05_reinforcement_learning.md` မှာ AI ကို လေ့ကျင့်နည်း လေ့လာပါမယ်!
