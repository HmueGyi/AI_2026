# ပထမဆုံး Simulation ဖန်တီးခြင်း

## လေ့ကျင့်ခန်း ၁ - Empty World ဖန်တီးရန်

### အဆင့် ၁ - Isaac Lab ဖွင့်ရန်

```bash
# Terminal ဖွင့်ပါ
cd IsaacLab
conda activate isaaclab

# Empty simulation ဖွင့်ပါ
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

---

## လေ့ကျင့်ခန်း ၂ - Robot ထည့်ရန်

### ၂.၁ Franka Robot ထည့်ရန်

```bash
# Franka Panda robot simulation
./isaaclab.sh -p source/standalone/demos/arms.py
```

**ဘာတွေမြင်ရမလဲ:**
- Robotic arm တစ်ခု မြင်ရပါမယ်
- Robot က random motion လုပ်နေပါမယ်

### ၂.၂ Mobile Robot ထည့်ရန်

```bash
# ANYmal quadruped robot
./isaaclab.sh -p source/standalone/demos/quadrupeds.py
```

---

## လေ့ကျင့်ခန်း ၃ - Control Keys

**Simulation ထဲမှာ အသုံးပြုနိုင်တဲ့ Keys:**

| Key | လုပ်ဆောင်ချက် |
|-----|-------------|
| `Space` | Pause/Play simulation |
| `R` | Reset simulation |
| `Q` | Quit simulation |
| `Mouse Scroll` | Zoom in/out |
| `Right Click + Drag` | Rotate camera |
| `Middle Click + Drag` | Pan camera |

---

## လေ့ကျင့်ခန်း ၄ - Python Script ဖန်တီးရန်

### ၄.၁ Simple Script တစ်ခု ရေးကြည့်ရန်

File အသစ်ဖန်တီးပါ: `my_first_sim.py`

```python
# my_first_sim.py
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.app import AppLauncher

# Launch the simulator
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Ground plane ထည့်ပါ
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)

# Simulation run လုပ်ပါ
while simulation_app.is_running():
    simulation_app.update()

# Close
simulation_app.close()
```

### ၄.၂ Script Run လုပ်ရန်

```bash
./isaaclab.sh -p my_first_sim.py
```

---

## လေ့ကျင့်ခန်း ၅ - Object ထည့်ရန်

### ၅.၁ Cube ထည့်ရန်

```python
# cube_sim.py
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Ground plane
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Cube ထည့်ပါ
cfg_cube = sim_utils.CuboidCfg(
    size=(0.5, 0.5, 0.5),  # အရွယ်အစား
    spawn=sim_utils.SpawnCfg(
        pos=(0.0, 0.0, 1.0),  # အနေအထား (x, y, z)
    ),
)
cfg_cube.func("/World/cube", cfg_cube)

# Run simulation
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
```

---

## လေ့ကျင့်ခန်း ၆ - Physics ထည့်ရန်

### ၆.၁ လဲကျတဲ့ Object

```python
# falling_cube.py
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Ground plane
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/ground", cfg_ground)

# Rigid body cube (physics ပါတဲ့ cube)
cfg_cube = sim_utils.CuboidCfg(
    size=(0.3, 0.3, 0.3),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Physics ထည့်ပါ
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # အလေးချိန်
    spawn=sim_utils.SpawnCfg(pos=(0.0, 0.0, 5.0)),  # အမြင့်က ကြွပါစေ
)
cfg_cube.func("/World/falling_cube", cfg_cube)

# Run simulation - Cube လဲကျတာကို မြင်ရပါမယ်
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
```

---

## အကျဉ်းချုပ်

ဒီ lesson မှာ သင်ယူခဲ့တာတွေ:
- ✅ Empty simulation ဖွင့်နည်း
- ✅ Robot models တွေ load လုပ်နည်း
- ✅ Control keys တွေ
- ✅ Python script ရေးနည်း
- ✅ Objects တွေ ထည့်နည်း
- ✅ Physics simulation လုပ်နည်း

**နောက်ထပ်:** `03_robot_control.md` မှာ robot ကို control လုပ်နည်း လေ့လာပါမယ်!
