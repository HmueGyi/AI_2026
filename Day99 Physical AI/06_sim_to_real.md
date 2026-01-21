# Simulation မှ Real Robot သို့ Transfer လုပ်ခြင်း

## အခန်း ၁ - Sim-to-Real Gap

### ၁.၁ Problem ဘာလဲ?

**Simulation မှာ:**
- ပြီးပြည့်စုံတဲ့ physics
- Noise မရှိ sensors
- အမြဲတမ်း တိကျတဲ့ measurements

**Real World မှာ:**
- Imperfect physics
- Noisy sensors
- Delays and latencies
- Unexpected disturbances

**ဒါကြောင့်** simulation မှာ အလုပ်လုပ်တဲ့ policy က real world မှာ မအလုပ်လုပ်နိုင်ပါ။

---

## အခန်း ၂ - Domain Randomization

### ၂.၁ Randomization အခြေခံ

Training လုပ်တဲ့အခါ parameters တွေကို random လုပ်ပါ:

```python
# Domain randomization configuration
from omni.isaac.lab.utils import configclass

@configclass
class RandomizationCfg:
    """Simulation parameters randomization"""
    
    # Physics properties
    mass_range = (0.8, 1.2)  # ±20% mass variation
    friction_range = (0.5, 1.5)  # Friction coefficient
    
    # Sensor noise
    camera_noise_std = 0.01  # RGB noise
    imu_noise_std = 0.1      # IMU noise
    
    # Actuator properties
    motor_strength_range = (0.9, 1.1)  # Motor power variation
    
    # Environmental factors
    lighting_range = (0.7, 1.3)  # Lighting changes
    
    # Delays
    control_delay_range = (0.0, 0.02)  # 0-20ms delay
```

---

### ၂.၂ Domain Randomization အသုံးပြုနည်း

File: `domain_randomization.py`

```python
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.envs import ManagerBasedRLEnv
import torch
import random

# Environment with randomization
from omni.isaac.lab_tasks.manager_based.manipulation.reach import ReachEnvCfg

env_cfg = ReachEnvCfg()
env_cfg.scene.num_envs = 64

# Add randomization
env_cfg.randomization = {
    "robot_mass": {
        "operation": "scale",
        "range": (0.8, 1.2),
    },
    "target_position": {
        "operation": "uniform",
        "range": [(-0.2, -0.2, 0.5), (0.2, 0.2, 0.8)],
    },
}

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

# Training loop with randomization
for step in range(1000):
    # Random actions (အရင်ဥပမာ)
    actions = torch.randn(env.num_envs, env.action_space.shape[0])
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    if step % 100 == 0:
        print(f"Step {step}: Mean reward = {rewards.mean():.2f}")

env.close()
simulation_app.close()
```

---

## အခန်း ၃ - Sensor Noise ထည့်ခြင်း

### ၃.၁ Camera Noise

```python
import torch
from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils

# Camera with noise
camera_cfg = CameraCfg(
    prim_path="/World/camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.SpawnCfg(pos=(1.0, 0.0, 1.0)),
)

camera = Camera(camera_cfg)

# ယခု camera data ကို noise ထည့်ပါမယ်
def add_camera_noise(rgb_data, noise_level=0.02):
    """Add Gaussian noise to RGB images"""
    noise = torch.randn_like(rgb_data) * noise_level
    noisy_rgb = torch.clamp(rgb_data + noise, 0.0, 1.0)
    return noisy_rgb

# Simulation loop
for i in range(500):
    sim.step()
    camera.update(sim.dt)
    
    if camera.data.output is not None:
        clean_rgb = camera.data.output["rgb"]
        noisy_rgb = add_camera_noise(clean_rgb)
        
        # ယခု noisy_rgb ကို သုံးပါ
```

---

### ၃.၂ IMU Noise

```python
def add_imu_noise(acceleration, angular_velocity, noise_std=0.1):
    """Add noise to IMU readings"""
    acc_noise = torch.randn_like(acceleration) * noise_std
    gyro_noise = torch.randn_like(angular_velocity) * noise_std * 0.01
    
    noisy_acc = acceleration + acc_noise
    noisy_gyro = angular_velocity + gyro_noise
    
    return noisy_acc, noisy_gyro

# Usage
clean_acc = imu.data.lin_acc
clean_gyro = imu.data.ang_vel

noisy_acc, noisy_gyro = add_imu_noise(clean_acc, clean_gyro)
```

---

## အခန်း ၄ - System Identification

### ၄.၁ Real Robot Parameters ရှာခြင်း

Real robot မှာ run ပြီး actual parameters တွေ တိုင်းတာပါ:

```python
# sys_id.py - Real robot မှာ run ရမယ်
import numpy as np

def measure_motor_response():
    """Motor က ဘယ်လောက် powerful ရှိလဲ တိုင်းရန်"""
    commanded_torque = 1.0  # Newton-meters
    
    # Send command
    robot.set_joint_torque([commanded_torque, 0, 0, 0, 0, 0, 0])
    
    # Measure actual response
    time.sleep(0.1)
    actual_velocity = robot.get_joint_velocity()[0]
    
    # Calculate actual motor constant
    motor_constant = actual_velocity / commanded_torque
    
    return motor_constant

def measure_friction():
    """Friction coefficient တိုင်းရန်"""
    # Move joint slowly
    robot.set_joint_velocity([0.1, 0, 0, 0, 0, 0, 0])
    
    # Measure required torque
    required_torque = robot.get_joint_torque()[0]
    
    # Estimate friction
    friction = required_torque / 0.1
    
    return friction

# Measurements run ပါ
motor_k = measure_motor_response()
friction_coef = measure_friction()

print(f"Motor constant: {motor_k}")
print(f"Friction coefficient: {friction_coef}")

# ဒီ values တွေကို simulation မှာ သုံးပါ
```

---

## အခန်း ၅ - Control Frequency Matching

### ၅.၁ Real Robot နဲ့ တူအောင် လုပ်ရန်

```python
# Simulation မှာ control frequency သတ်မှတ်ပါ
from omni.isaac.lab.app import AppLauncher

# Real robot က 50Hz control frequency ရှိရင်
control_dt = 1.0 / 50.0  # 20ms per control step

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils

cfg = sim_utils.SimulationCfg(
    dt=0.005,  # Physics timestep (5ms)
    substeps=1,
)

sim = sim_utils.SimulationContext(cfg)

# Control loop
control_counter = 0
control_decimation = 4  # Every 4 physics steps = 20ms

for step in range(10000):
    # Physics step
    sim.step()
    
    # Control update (every 20ms only)
    if control_counter % control_decimation == 0:
        # Compute and send new control command
        action = compute_control_action(state)
        robot.set_joint_position_target(action)
    
    control_counter += 1
```

---

## အခန်း ၆ - Latency Simulation

### ၆.၁ Communication Delays

Real robot communication မှာ delays ရှိပါတယ်:

```python
import collections

class DelayBuffer:
    """Simulate communication latency"""
    
    def __init__(self, delay_steps=2):
        self.delay_steps = delay_steps
        self.buffer = collections.deque(maxlen=delay_steps)
        
        # Initialize with zeros
        for _ in range(delay_steps):
            self.buffer.append(None)
    
    def add(self, data):
        """Add new data"""
        self.buffer.append(data)
    
    def get(self):
        """Get delayed data"""
        return self.buffer[0]

# Usage
observation_buffer = DelayBuffer(delay_steps=3)  # 3 steps delay
action_buffer = DelayBuffer(delay_steps=2)       # 2 steps delay

for step in range(1000):
    # Get delayed observation
    delayed_obs = observation_buffer.get()
    
    if delayed_obs is not None:
        # Compute action based on delayed observation
        action = policy(delayed_obs)
        
        # Add action to delay buffer
        action_buffer.add(action)
    
    # Get delayed action to send to robot
    delayed_action = action_buffer.get()
    
    if delayed_action is not None:
        robot.set_joint_position_target(delayed_action)
    
    # Get new observation and add to buffer
    new_obs = get_observation()
    observation_buffer.add(new_obs)
    
    sim.step()
```

---

## အခန်း ၇ - Gradual Transfer Strategy

### ၇.၁ အဆင့်ဆင့် Transfer လုပ်ခြင်း

**Stage 1: Pure Simulation**
```bash
# Simulation မှာ train ပါ (no randomization)
./isaaclab.sh -p train.py --task ReachTask --num_envs 512
```

**Stage 2: Light Randomization**
```bash
# အနည်းငယ် randomization ထည့်ပါ
./isaaclab.sh -p train.py --task ReachTask-Randomized-v1 --num_envs 512
```

**Stage 3: Heavy Randomization**
```bash
# Realistic randomization
./isaaclab.sh -p train.py --task ReachTask-Randomized-v2 --num_envs 512
```

**Stage 4: Real Robot Testing**
```bash
# Real robot မှာ test လုပ်ပါ
python deploy_to_robot.py --checkpoint model.pt
```

---

## အခန်း ၈ - Safety Considerations

### ၈.၁ Real Robot မှာ အန္တရာယ်ကင်းအောင်

```python
class SafetyWrapper:
    """Safety checks for real robot deployment"""
    
    def __init__(self, robot):
        self.robot = robot
        
        # Safety limits
        self.max_joint_velocity = 2.0  # rad/s
        self.max_joint_torque = 10.0   # Nm
        self.workspace_limits = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 1.0),
        }
    
    def check_joint_limits(self, joint_positions):
        """Check if joints are within safe range"""
        for i, pos in enumerate(joint_positions):
            if not (self.robot.joint_limits[i][0] <= pos <= self.robot.joint_limits[i][1]):
                print(f"WARNING: Joint {i} out of range!")
                return False
        return True
    
    def check_velocity(self, joint_velocities):
        """Check if velocities are safe"""
        for i, vel in enumerate(joint_velocities):
            if abs(vel) > self.max_joint_velocity:
                print(f"WARNING: Joint {i} velocity too high!")
                return False
        return True
    
    def emergency_stop(self):
        """Stop robot immediately"""
        print("EMERGENCY STOP!")
        self.robot.set_joint_velocity([0] * 7)
        self.robot.disable_motors()
    
    def safe_action(self, action):
        """Filter unsafe actions"""
        # Clip to safe range
        safe_action = np.clip(
            action,
            self.robot.action_space.low,
            self.robot.action_space.high
        )
        
        return safe_action

# Usage
safety = SafetyWrapper(robot)

while True:
    obs = get_observation()
    action = policy(obs)
    
    # Check safety
    safe_action = safety.safe_action(action)
    
    if safety.check_joint_limits(safe_action):
        robot.apply_action(safe_action)
    else:
        safety.emergency_stop()
        break
```

---

## အခန်း ၉ - Real World Testing Checklist

### ၉.၁ စစ်ဆေးရမည့် အချက်များ

**Before deploying to real robot:**

- [ ] Model trained with domain randomization
- [ ] Tested with sensor noise
- [ ] Tested with latency simulation
- [ ] Control frequency matches real robot
- [ ] Safety limits implemented
- [ ] Emergency stop working
- [ ] Workspace limits configured
- [ ] Collision detection active

**During initial testing:**

- [ ] Start with slow motions
- [ ] Human supervisor present
- [ ] Emergency stop accessible
- [ ] Record all attempts
- [ ] Gradually increase speed
- [ ] Test in safe environment

---

## အခန်း ၁၀ - ROS Integration (Optional)

### ၁၀.၁ ROS နဲ့ ချိတ်ဆက်ခြင်း

```python
# ros_interface.py
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class ROSRobotInterface:
    """Interface between trained policy and ROS robot"""
    
    def __init__(self):
        rospy.init_node('isaac_policy_node')
        
        # Subscribers
        self.joint_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self.joint_callback
        )
        
        # Publishers
        self.cmd_pub = rospy.Publisher(
            '/joint_commands',
            Float64MultiArray,
            queue_size=10
        )
        
        self.current_state = None
    
    def joint_callback(self, msg):
        """Receive joint states"""
        self.current_state = {
            'position': msg.position,
            'velocity': msg.velocity,
        }
    
    def send_command(self, action):
        """Send action to robot"""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = action.tolist()
        self.cmd_pub.publish(cmd_msg)
    
    def run(self, policy):
        """Main control loop"""
        rate = rospy.Rate(50)  # 50 Hz
        
        while not rospy.is_shutdown():
            if self.current_state is not None:
                # Get action from policy
                action = policy(self.current_state)
                
                # Send to robot
                self.send_command(action)
            
            rate.sleep()

# Usage
interface = ROSRobotInterface()
interface.run(trained_policy)
```

---

## အကျဉ်းချုပ်

ဒီ lesson မှာ လေ့လာခဲ့တာတွေ:
- ✅ Sim-to-Real Gap နဲ့ ပြဿနာများ
- ✅ Domain Randomization အသုံးပြုနည်း
- ✅ Sensor noise simulation
- ✅ System identification လုပ်နည်း
- ✅ Control frequency matching
- ✅ Latency simulation
- ✅ Gradual transfer strategy
- ✅ Safety considerations
- ✅ Real world testing checklist
- ✅ ROS integration

**အဆုံး**: ယခု သင့်မှာ NVIDIA Physical AI အတွက် အခြေခံ knowledge အပြည့်အစုံ ရှိပြီ ဖြစ်ပါတယ်! 🎉

---

## နောက်ထပ် Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67)
- [Research Papers on Sim-to-Real](https://arxiv.org/search/?query=sim+to+real&searchtype=all)
