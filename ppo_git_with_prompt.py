import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
from GitTextActionWithPrompt import GitText2ActionWithPrompt
from PIL import Image
from transformers import set_seed

# ✅ 勾配の問題を検出しやすくする（必要なら一時ON）
torch.autograd.set_detect_anomaly(True)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ハイパーパラメータ ===
EPISODES = 10
GAMMA = 0.99
LR = 1e-4
EPS_CLIP = 0.2
K_EPOCHS = 3

# === 環境とモデル ===
env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
model = GitText2ActionWithPrompt().to(device)
optimizer = optim.Adam(model.policy_head.parameters(), lr=LR)

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
        log_probs.append(log_prob.detach())  # detachして保存
        actions.append(action.detach())
        states.append(image)

    # === 割引報酬の計算 ===
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns).float().to(device)

    # === PPO更新 ===
    model.train()
    for _ in range(K_EPOCHS):
        for i in range(len(states)):
            logits = model(states[i])  # 再度forward

            # 明示的に分離したdist（安全）
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(actions[i])

            ratio = torch.exp(new_log_prob - log_probs[i])
            advantage = returns[i] - returns.mean()

            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
            loss = -torch.min(surrogate1, surrogate2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"✅ Episode {episode+1}, Total Reward: {sum(rewards):.2f}")

env.close()
