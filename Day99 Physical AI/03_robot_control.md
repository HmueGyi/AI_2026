# Robot Control လုပ်နည်း

## အခန်း ၁ - Robot တစ်ခုကို ရွေးခြင်း

### ၁.၁ Franka Panda Robot သုံးမယ်

Franka Panda က လွယ်ကူပြီး လေ့လာဖို့ကောင်းပါတယ်။

```bash
# Demo ကြည့်ရန်
cd IsaacLab
./isaaclab.sh -p source/standalone/demos/arms.py
```

---

## အခန်း ၂ - Joint Control (အဆစ်များ ထိန်းချုပ်ခြင်း)

### ၂.၁ Joint Positions ဖတ်ရန်

File: `read_joints.py`

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils

# Simulation scene ဖန်တီးပါ
cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

# Ground plane
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Franka robot ထည့်ပါ
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

robot_cfg = FRANKA_PANDA_CFG.copy()
robot_cfg.spawn.func("/World/Franka", robot_cfg.spawn, translation=(0.0, 0.0, 0.0))
robot = Articulation(robot_cfg)

# Simulation reset လုပ်ပါ
sim.reset()

# Simulation loop
for i in range(1000):
    # Joint positions ဖတ်ပါ
    joint_pos = robot.data.joint_pos
    
    print(f"Step {i}: Joint positions = {joint_pos}")
    
    # Simulation step
    sim.step()
    robot.update(sim.dt)

simulation_app.close()
```

---

### ၂.၂ Joint Positions သတ်မှတ်ရန်

File: `move_joints.py`

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

# Setup simulation
cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

# Ground and robot
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

robot_cfg = FRANKA_PANDA_CFG.copy()
robot_cfg.spawn.func("/World/Franka", robot_cfg.spawn)
robot = Articulation(robot_cfg)

sim.reset()

# Target joint positions (radians)
target_positions = torch.tensor([
    [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]  # 7 joints
])

# Simulation loop
for i in range(2000):
    # Set joint position targets
    robot.set_joint_position_target(target_positions)
    
    # Step simulation
    sim.step()
    robot.update(sim.dt)

simulation_app.close()
```

**Run လုပ်ရန်:**
```bash
./isaaclab.sh -p move_joints.py
```

---

## အခန်း ၃ - Velocity Control (အမြန်နှုန်း ထိန်းချုပ်ခြင်း)

### ၃.၁ Joint Velocities သတ်မှတ်ရန်

File: `velocity_control.py`

```python
import torch
import math
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

# Setup
cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

robot_cfg = FRANKA_PANDA_CFG.copy()
robot_cfg.spawn.func("/World/Franka", robot_cfg.spawn)
robot = Articulation(robot_cfg)

sim.reset()

# Simulation loop - ပထမ joint ကို လှည့်ပါမယ်
for i in range(3000):
    # Joint 0 ကို 0.5 rad/s နှုန်းနဲ့ လှည့်ပါ
    velocity = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    robot.set_joint_velocity_target(velocity)
    
    sim.step()
    robot.update(sim.dt)

simulation_app.close()
```

---

## အခန်း ၄ - Sine Wave Motion (လှိုင်းပုံစံ ရွေ့လျားခြင်း)

### ၄.၁ ချောမွေ့တဲ့ Motion

File: `sine_motion.py`

```python
import torch
import math
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

robot_cfg = FRANKA_PANDA_CFG.copy()
robot_cfg.spawn.func("/World/Franka", robot_cfg.spawn)
robot = Articulation(robot_cfg)

sim.reset()

# Sine wave motion
count = 0
while simulation_app.is_running():
    # Sine wave လုပ်ပါ
    angle = math.sin(count * 0.01) * 1.0  # -1 to 1 radians
    
    target = torch.tensor([[angle, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]])
    robot.set_joint_position_target(target)
    
    sim.step()
    robot.update(sim.dt)
    count += 1

simulation_app.close()
```

---

## အခန်း ၅ - Keyboard Control

### ၅.၁ Keyboard နဲ့ Robot ကို Control လုပ်ရန်

File: `keyboard_control.py`

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

# Setup
cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

robot_cfg = FRANKA_PANDA_CFG.copy()
robot_cfg.spawn.func("/World/Franka", robot_cfg.spawn)
robot = Articulation(robot_cfg)

sim.reset()

# Current joint positions
current_pos = torch.zeros((1, 7))
step_size = 0.1

print("Controls:")
print("  1/2: Joint 0 +/-")
print("  3/4: Joint 1 +/-")
print("  Q: Quit")

while simulation_app.is_running():
    # ဒီမှာ keyboard input ဖတ်နိုင်ပါတယ် (advanced)
    # အခု demo မှာတော့ automatic motion လုပ်ပါမယ်
    
    robot.set_joint_position_target(current_pos)
    
    sim.step()
    robot.update(sim.dt)

simulation_app.close()
```

---

## အခန်း ၆ - Multiple Robots

### ၆.၁ Robot များစွာ တပြိုင်နက် Control လုပ်ရန်

```python
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(cfg)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Robot 1
robot_cfg1 = FRANKA_PANDA_CFG.copy()
robot_cfg1.spawn.func("/World/Franka1", robot_cfg1.spawn, translation=(-1.0, 0.0, 0.0))
robot1 = Articulation(robot_cfg1)

# Robot 2
robot_cfg2 = FRANKA_PANDA_CFG.copy()
robot_cfg2.spawn.func("/World/Franka2", robot_cfg2.spawn, translation=(1.0, 0.0, 0.0))
robot2 = Articulation(robot_cfg2)

sim.reset()

# Different positions for each robot
target1 = torch.tensor([[0.5, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]])
target2 = torch.tensor([[-0.5, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]])

for i in range(2000):
    robot1.set_joint_position_target(target1)
    robot2.set_joint_position_target(target2)
    
    sim.step()
    robot1.update(sim.dt)
    robot2.update(sim.dt)

simulation_app.close()
```

---

## အကျဉ်းချုပ်

ဒီ lesson မှာ လေ့လာခဲ့တာတွေ:
- ✅ Joint positions ဖတ်နည်း
- ✅ Position control လုပ်နည်း
- ✅ Velocity control လုပ်နည်း
- ✅ Smooth motion (sine wave) လုပ်နည်း
- ✅ Multiple robots control လုပ်နည်း

**နောက်ထပ်:** `04_camera_sensors.md` မှာ camera တွေ sensor တွေ သုံးနည်း လေ့လာပါမယ်!
