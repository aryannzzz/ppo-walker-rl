# Train a Robot to Walk
## iBot Mini Project 1: Reinforcement Learning for Bipedal Locomotion

This project teaches you to train a virtual robot to walk using **Proximal Policy Optimization (PPO)**, a modern deep reinforcement learning algorithm. The robot lives in a physics simulation (PyBullet) and you control it only through a reward function that you design.

By the end of this project you will have:
- A robot that walks in a simulation
- Hands-on experience with reward engineering
- Results from your own ablation experiments
- A short written report and a demo video

---

## Repository Layout

```
iBot-MP1/
|
|-- env/
|   |-- __init__.py           Package entry point
|   |-- walker_env.py         Walker2D Gymnasium environment
|   |-- reward_functions.py   All reward functions (sparse, dense, ...)
|
|-- Week1/
|   |-- guide.tex             Week 1 reading guide (compile to PDF)
|   |-- 01_rl_concepts.ipynb  Solved examples: GridWorld and value iteration
|   |-- 02_cartpole_ppo.ipynb Solved examples: CartPole with PPO
|   |-- starter_week1.ipynb   Your starting point for Week 1
|
|-- Week2/
|   |-- guide.tex             Week 2 reading guide
|   |-- 01_reward_shaping.ipynb  Solved examples: reward shaping in GridWorld
|   |-- starter_week2.ipynb   Your starting point for Week 2
|
|-- Week3/
|   |-- guide.tex             Week 3 reading guide
|   |-- 01_hyperparameter_effects.ipynb  Solved examples: how LR and net size matter
|   |-- starter_week3.ipynb   Your starting point for Week 3
|
|-- Week4/
|   |-- guide.tex             Week 4 reading guide
|   |-- starter_week4.ipynb   Final agent, video, and report
|
|-- train.py                  Command-line training script
|-- evaluate.py               Evaluation + video recording script
|-- requirements.txt          Python package list
|-- README.md                 This file
```

---

## Quick Start

### 1. Set up the Python environment

We recommend Python 3.10 or 3.11. Create a virtual environment first.

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the baseline training (Week 1)

```bash
python train.py --reward dense --total_steps 1000000
```

This saves logs and model checkpoints under `runs/`.

### 3. Watch training in TensorBoard

Open a new terminal, then run:

```bash
tensorboard --logdir runs/
```

Open the link shown (usually `http://localhost:6006`) in your browser.

### 4. Evaluate the trained agent

```bash
python evaluate.py --run_dir runs/dense_YYYYMMDD_HHMMSS --n_episodes 20 --plot
```

Replace the date/time part with the actual folder name created during training.

### 5. Record a video

```bash
python evaluate.py --run_dir runs/dense_YYYYMMDD_HHMMSS --video_out demo.mp4
```

---

## Weekly Tasks

| Week | Main Goal | Key Files |
|------|-----------|-----------|
| 1 | Set up the environment. Train a baseline PPO agent. Observe TensorBoard. | `Week1/` |
| 2 | Implement four reward functions. Compare their effects on behavior. | `Week2/` |
| 3 | Run ablation experiments over five or more hyperparameter settings. | `Week3/` |
| 4 | Train your best agent to 1M steps. Record a video. Write a short report. | `Week4/` |

Read the `guide.tex` file inside each week's folder before starting that week's notebook.

---

## Using the Training Script

```
python train.py [options]

--reward         sparse | dense | velocity_only | heavy_energy  (default: dense)
--total_steps    total environment steps (default: 1000000)
--n_envs         parallel environments (default: 4)
--lr             learning rate (default: 3e-4)
--n_steps        rollout steps per environment per update (default: 2048)
--batch_size     mini-batch size (default: 64)
--gamma          discount factor (default: 0.99)
--net_arch       hidden layer sizes e.g. "64 64" or "256 256" (default: "256 256")
--run_name       name for this run (auto-generated if not given)
--out_dir        output directory (default: runs)
```

---

## Compiling the LaTeX Guides

The guide files are in LaTeX format (`.tex`). To compile them to PDF:

```bash
pdflatex Week1/guide.tex
```

Or open the `.tex` file in a LaTeX editor such as TeXstudio, Overleaf, or VS Code with the LaTeX Workshop extension.

---

## Packages Used

| Package | What it does |
|---------|--------------|
| `pybullet` | Physics simulation of the robot |
| `stable-baselines3` | PPO implementation |
| `gymnasium` | Standard environment API |
| `tensorboard` | Training curve visualization |
| `matplotlib` | Plots for your report |
| `imageio` / `imageio-ffmpeg` | Video recording |
| `torch` | Deep learning backend (required by SB3) |

---

## References

All the papers mentioned in the guides are freely available online. Links are included in the LaTeX documents.

- Schulman et al. (2017). Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
- Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. https://arxiv.org/abs/1506.02438
- Hwangbo et al. (2019). Learning agile and dynamic motor skills for legged robots. Science Robotics.

---

*Robotics and ML Club -- DC Training Program*
