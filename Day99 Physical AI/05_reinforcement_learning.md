# Reinforcement Learning ဖြင့် Robot လေ့ကျင့်ခြင်း

## အခန်း ၁ - RL အခြေခံများ

### ၁.၁ Reinforcement Learning ဆိုတာ ဘာလဲ?

**ရိုးရှင်းသောဥပမာ:**
- Robot က **action** (လုပ်ဆောင်ချက်) လုပ်ပါတယ်
- Environment က **state** (အခြေအနေ) ပေးပါတယ်
- ကောင်းရင် **reward** (+) ရပါတယ်
- မကောင်းရင် **penalty** (-) ရပါတယ်
- Robot က reward များဖို့ သင်ယူပါတယ်

---

## အခန်း ၂ - Cartpole ဥပမာ (အရိုးရှင်းဆုံး)

### ၂.၁ Cartpole Environment Setup

File: `cartpole_rl.py`

```python
from omni.isaac.lab.app import AppLauncher

# Launch with training mode
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.envs import ManagerBasedRLEnv
import torch

# Cartpole က pole တစ်ခုကို balance လုပ်ဖို့ cart ကို လှုပ်ပါတယ်
# Goal: Pole မလဲအောင် ထားရန်

print("Loading Cartpole environment...")

# Environment config
from omni.isaac.lab_tasks.manager_based.classic.cartpole import CartpoleEnvCfg

env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = 4  # 4 parallel environments

env = ManagerBasedRLEnv(cfg=env_cfg)

# Reset environment
obs, _ = env.reset()
print(f"Observation shape: {obs['policy'].shape}")

# Random actions (မသင်ယူရသေးတဲ့အခါ)
for step in range(500):
    # Random action
    actions = torch.rand(env.num_envs, env.action_space.shape[0]) * 2 - 1
    
    # Step environment
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    if step % 50 == 0:
        print(f"Step {step}, Reward: {rewards.mean().item():.2f}")

env.close()
simulation_app.close()
```

---

## အခန်း ၃ - PPO Algorithm နဲ့ လေ့ကျင့်ခြင်း

### ၃.၁ Training Script

Isaac Lab မှာ training လုပ်ဖို့ built-in scripts တွေ ရှိပါတယ်။

```bash
# Cartpole ကို train လုပ်ရန်
cd IsaacLab

# Training start လုပ်ပါ
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 512 \
  --headless
```

**Parameters ရှင်းလင်းချက်:**
- `--task`: လေ့ကျင့်မည့် task
- `--num_envs`: parallel environments အရေအတွက် (များလေ လေ့လာမြန်လေ)
- `--headless`: GUI မပါဘဲ run မယ် (မြန်ပါတယ်)

---

### ၃.၂ Training Progress ကြည့်ရန်

```bash
# Training running နေစဉ် terminal အသစ်ဖွင့်ပြီး:
cd IsaacLab/logs

# Tensorboard ဖွင့်ပါ
tensorboard --logdir .
```

Browser မှာ `http://localhost:6006` ဖွင့်ပါ။
- Reward graph
- Loss graph
- Training progress တွေ မြင်ရပါမယ်

---

## အခန်း ၄ - Trained Model Test လုပ်ခြင်း

### ၄.၁ Trained Policy Run ခြင်း

```bash
# Training ပြီးရင် model ကို test လုပ်ပါ
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/cartpole/model.pt
```

ယခု cartpole တွေက balance ကောင်းကောင်းလုပ်နိုင်ပြီ!

---

## အခန်း ၅ - Robot Arm Reaching Task

### ၅.၁ Reach Task လေ့ကျင့်ခြင်း

Robot arm က target position ကို ရောက်အောင် လေ့လာမယ်။

```bash
# Franka arm reaching task train လုပ်ရန်
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Reach-Franka-v0 \
  --num_envs 256 \
  --headless
```

---

### ၅.၂ Custom Reward Function

File: `custom_reach_reward.py`

```python
import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import RewardTermCfg

def distance_to_target_reward(env, asset_cfg: SceneEntityCfg):
    """Target နဲ့ အနီးနားရောက်ရင် reward"""
    # End-effector position
    ee_pos = env.scene["robot"].data.body_pos_w[:, asset_cfg.body_ids[0], :]
    
    # Target position
    target_pos = env.scene["target"].data.root_pos_w
    
    # Distance တွက်ပါ
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # Distance နည်းလေ reward များလေ
    reward = 1.0 / (1.0 + distance)
    
    return reward

def reached_target_bonus(env, asset_cfg: SceneEntityCfg, threshold: float = 0.05):
    """Target ရောက်သွားရင် bonus"""
    ee_pos = env.scene["robot"].data.body_pos_w[:, asset_cfg.body_ids[0], :]
    target_pos = env.scene["target"].data.root_pos_w
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # Threshold အောက်ရောက်ရင် bonus
    bonus = (distance < threshold).float() * 10.0
    
    return bonus

# Reward configuration
rewards = {
    "distance_reward": RewardTermCfg(
        func=distance_to_target_reward,
        weight=1.0,
    ),
    "reached_bonus": RewardTermCfg(
        func=reached_target_bonus,
        weight=1.0,
        params={"threshold": 0.05},
    ),
}
```

---

## အခန်း ၆ - Locomotion (လမ်းလျှောက်ခြင်း)

### ၆.၁ ANYmal Quadruped လေ့ကျင့်ခြင်း

လေးခြေသတ္တဝါ robot ကို လမ်းလျှောက်ဖို့ လေ့ကျင့်မယ်။

```bash
# ANYmal robot ကို လမ်းလျှောက်ခိုင်းပါမယ်
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 \
  --num_envs 4096 \
  --headless
```

**ဒီ task မှာ:**
- Robot က ပေးထားတဲ့ velocity နဲ့ လျှောက်ရမယ်
- မလဲအောင် balance လုပ်ရမယ်
- Smooth ဖြစ်အောင် လုပ်ရမယ်

---

### ၆.၂ Training Time

| Task | Environments | Training Time |
|------|-------------|---------------|
| Cartpole | 512 | ~5 minutes |
| Reach (Franka) | 256 | ~20 minutes |
| Locomotion (ANYmal) | 4096 | ~2-3 hours |

*GPU dependent - RTX 3090 기준

---

## အခန်း ၇ - Curriculum Learning

### ၇.၁ တဖြည်းဖြည်း ခက်အောင် လုပ်ခြင်း

Robot ကို အလွယ်ကစပြီး ဖြည်းဖြည်း ခက်အောင် သင်ပေးမယ်။

```python
# Curriculum config example
curriculum = {
    "stage_1": {
        "max_velocity": 0.5,  # ဖြည်းဖြည်း လျှောက်ပါ
        "terrain": "flat",     # ညီညာတဲ့မြေ
        "duration": 1000000,   # steps
    },
    "stage_2": {
        "max_velocity": 1.0,   # ပိုမြန်အောင်
        "terrain": "flat",
        "duration": 2000000,
    },
    "stage_3": {
        "max_velocity": 1.5,
        "terrain": "rough",    # ကြမ်းတမ်းတဲ့မြေ
        "duration": 3000000,
    },
}
```

---

## အခန်း ၈ - Hyperparameter Tuning

### ၈.၁ အရေးကြီးတဲ့ Parameters

```python
# Training configuration
train_cfg = {
    # Learning rate - သင်ယူမြန်နှုန်း
    "learning_rate": 1e-3,  # များရင် unstable, နည်းရင် နှေး
    
    # Batch size - တစ်ခါ သင်ယူတဲ့ data အရေအတွက်
    "num_steps_per_env": 24,
    "num_minibatches": 4,
    
    # Discount factor - အနာဂတ် reward ကို တန်ဖိုးထား
    "gamma": 0.99,
    
    # Clip range - policy update မကြီးလွန်းအောင်
    "clip_param": 0.2,
    
    # Entropy bonus - exploration အတွက်
    "entropy_coef": 0.01,
}
```

---

## အခန်း ၉ - Multi-Task Learning

### ၉.၁ Task များစွာ တပြိုင်နက် လေ့ကျင့်ခြင်း

```bash
# Multiple tasks train လုပ်ရန်
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Reach-Franka-v0 Isaac-Lift-Franka-v0 \
  --num_envs 128
```

---

## အကျဉ်းချုပ်

ဒီ lesson မှာ လေ့လာခဲ့တာတွေ:
- ✅ Reinforcement Learning အခြေခံ
- ✅ Cartpole လေ့ကျင့်နည်း
- ✅ PPO algorithm သုံးနည်း
- ✅ Training progress ကြည့်နည်း (Tensorboard)
- ✅ Custom reward functions ရေးနည်း
- ✅ Locomotion tasks
- ✅ Curriculum learning
- ✅ Hyperparameter tuning

**နောက်ထပ်:** `06_sim_to_real.md` မှာ simulation ကနေ real robot ကို transfer လုပ်နည်း လေ့လာပါမယ်!
