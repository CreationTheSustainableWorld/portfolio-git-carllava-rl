import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from PIL import Image
from transformers import set_seed
from git_rl_carllava_model import GitRLCarllavaModel

# === ハイパーパラメータ ===
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 10
GAMMA = 0.99
LR = 1e-4
EPS_CLIP = 0.2
K_EPOCHS = 3
POLICY_SAVE_PATH = "policy_head_rl_latest.pth"

# === 環境とモデルの初期化 ===
env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
model = GitRLCarllavaModel(model_path="lora_git_caption_model_carllava").to(device)

# === 以前のpolicy_headが存在すれば読み込む ===
if os.path.exists(POLICY_SAVE_PATH):
    model.policy_head.load_state_dict(torch.load(POLICY_SAVE_PATH))
    print(f"📥 前回の policy_head を読み込みました: {POLICY_SAVE_PATH}")
else:
    print("🆕 新しい policy_head から学習開始")

optimizer = optim.Adam(model.policy_head.parameters(), lr=LR)

# === エピソード学習ループ ===
for episode in range(EPISODES):
    obs, _ = env.reset()
    done = False
    rewards = []
    log_probs = []
    actions = []
    states = []

    while not done:
        image = Image.fromarray(obs).convert("RGB")

        logits = model(image)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        rewards.append(reward)
        log_probs.append(log_prob.detach())
        actions.append(action.detach())
        states.append(image)

    # === 割引報酬の計算 ===
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns).float().to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # === PPO更新 ===
    model.train()
    for _ in range(K_EPOCHS):
        for i in range(len(states)):
            logits = model(states[i])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(actions[i])

            ratio = torch.exp(new_log_prob - log_probs[i])
            advantage = returns[i]

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"✅ Episode {episode+1}, Total Reward: {sum(rewards):.2f}")

    # === policy_head の保存 ===
    torch.save(model.policy_head.state_dict(), POLICY_SAVE_PATH)
    print(f"💾 policy_head を保存しました: {POLICY_SAVE_PATH}")

env.close()
