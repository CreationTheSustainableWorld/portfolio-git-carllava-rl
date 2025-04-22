import os
import csv
import gymnasium as gym
import torch
import numpy as np
import cv2
from network import cnn
from datetime import datetime

# モデル準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn().to(device)
model.load_state_dict(torch.load("runs/model_best.pth", map_location=device))
model.eval()

# 行動ラベルのテキスト対応
action_text_map = {
    0: "何もしない",
    1: "アクセルを踏んで",
    2: "左に曲がって",
    3: "右に曲がって",
    4: "ブレーキを踏んで"
}

# 保存ディレクトリの作成
base_dir = "CarRacing-Data"
image_dir = os.path.join(base_dir, "images")
os.makedirs(image_dir, exist_ok=True)
csv_path = os.path.join(base_dir, "actions.csv")

# 環境準備
env = gym.make('CarRacing-v2', render_mode="rgb_array", continuous=False)
state, _ = env.reset()
step_count = 0
data = []

def preprocess(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    gray = gray / 255.0
    tensor = torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

def select_action(state_tensor):
    with torch.no_grad():
        logits = model(state_tensor).squeeze()
        return logits.argmax().item()

try:
    for ep in range(10):  # 5エピソード分収集
        print(f"Episode {ep+1}")
        state, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            image_path = os.path.join(image_dir, f"{step_count:04d}.png")
            cv2.imwrite(image_path, cv2.cvtColor(state, cv2.COLOR_RGB2BGR))

            state_tensor = preprocess(state)
            action = select_action(state_tensor)

            data.append([f"{step_count:04d}.png", action_text_map[action]])

            state, reward, terminated, truncated, _ = env.step(action)
            step_count += 1

finally:
    env.close()
    print("環境終了。データを保存中...")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "action_text"])
        writer.writerows(data)

    print(f"保存完了！画像: {len(data)}枚, CSV: {csv_path}")
