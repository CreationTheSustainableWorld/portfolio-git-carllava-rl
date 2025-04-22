# 🧠 CarLLaVA-RL: Vision-Language Guided Reinforcement Learning with LoRA

This project demonstrates how to use a LoRA-fine-tuned vision-language model (Git) in the style of CarLLaVA to guide reinforcement learning (PPO) in the [CarRacing-v2](https://www.gymlibrary.dev/environments/box2d/car_racing/) environment.

---

## 🚗 Overview

We leverage the Git vision-language model and fine-tune it using a small dataset of driving scenarios paired with natural language action descriptions like:

> What should the vehicle do in this scenario? → "Turn left" / "Accelerate" / "Brake"

Once trained, the model is used as a policy in a PPO loop to learn autonomous driving behavior in a simulated environment.

---

## 🧱 Project Structure

carllava-rl/ ├── train_git_lora_carllava_style.py # LoRA fine-tuning with fixed prompts ├── ppo_git_with_prompt.py # Reinforcement Learning with PPO ├── GitText2ActionWithPrompt.py # Model class: Git + Prompt + Policy head ├── CarRacing-Data/ # Collected caption dataset ├── lora_git_caption_model_carllava/ # Fine-tuned Git model ├── results/ # Training graphs and videos └── README.md

yaml
コピーする
編集する

---

## 📊 Performance

| Epoch | Avg Loss (LoRA) | Total Reward (PPO) |
|-------|------------------|--------------------|
| 1     | 8.59             | -28.3              |
| 2     | 8.11             | -11.8              |
| 3     | 8.06             | +22.6 🚀           |

### Reward Curve

![Reward Curve](results/reward_curve.png)

---

## 🎮 Demo (Video)

> Below is the video of the trained agent using the Git model with natural language prompts:

https://user-images.githubusercontent.com/your-username/video_episode_10.mp4

---

## 🔧 Key Features

- 🔁 **Caption Collection**: From trained policy (image → natural language)
- 🧠 **LoRA Fine-tuning**: Git model fine-tuned on small caption dataset
- 🤖 **Prompt-guided RL**: "What should the vehicle do in this scenario?" as fixed input prompt
- 🏎️ **Control Output**: Logits for discrete driving actions (no-op, accelerate, left, right, brake)

---

## 🚀 Performance Improvements

The following changes improved the training performance significantly:

- ✅ Return normalization
- ✅ Dropout & hidden layer expansion in policy head
- ✅ Clipped rewards for stability
- ✅ Multiple PPO epochs (`K_EPOCHS = 5`)

---

## 💾 Setup

You can install dependencies using conda:

```bash
conda env create -f environment.yml
conda activate carllava-rl
Or use requirements.txt.

📚 References
CarLLaVA (arXiv)

Git Model (Hugging Face)

LoRA (PEFT)
