# 🚗 CarLLaVA-Style Vision-to-Action Model for Autonomous Driving

This project demonstrates a vision-language-to-action learning framework  
inspired by [CarLLaVA](https://arxiv.org/abs/2406.10165),  
combining Git (Generative Image-to-Text Transformer), LoRA fine-tuning, and Reinforcement Learning (PPO)  
to train an agent that understands visual scenes and outputs driving actions.

---

## 📌 Overview

- 🔍 **Input**: RGB image of driving environment + fixed natural language prompt
- 🧠 **Backbone**: `microsoft/git-base` (Vision-Language model)
- 🛠 **Fine-Tuning**: LoRA (Low-Rank Adaptation) on image-action descriptions
- 🎯 **Action Prediction**: Lightweight `policy_head` trained with PPO
- 🚙 **Environment**: `CarRacing-v2` from OpenAI Gymnasium

---

## 🗂 Project Structure

| File/Dir                          | Description |
|----------------------------------|-------------|
| `train_caption_lora.py`          | LoRA-based caption fine-tuning on Git |
| `git_rl_carllava_model.py`       | Git model with added policy head (action predictor) |
| `train_policy_head.py`           | PPO training on policy head using CarRacing-v2 |
| `train_data_collect.py`          | Generates image + action caption pairs for pretraining |
| `train_data_collector_model.py`  | Model used for data collection |
| `convert_to_jsonl.py`            | Converts CSV to JSONL for caption training |
| `dataset.py`                     | Hugging Face Dataset loader for caption training |
| `policy_head_rl_latest.pth`      | Trained RL policy head (PPO) |
| `lora_git_caption_model_carllava/` | Saved LoRA fine-tuned Git model |
| `CarRacing-Data/`                | Collected training data (images + actions) |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Prepare training data (if not already collected)
bash
コピーする
編集する
python train_data_collect.py
python convert_to_jsonl.py
3. Fine-tune the captioning model with LoRA
bash
コピーする
編集する
python train_caption_lora.py
4. Train the policy head with reinforcement learning
bash
コピーする
編集する
python train_policy_head.py
🧠 Model Architecture
scss
コピーする
編集する
[RGB Image] + [Prompt]
         ↓
   Git Vision-Language Encoder
         ↓
   [CLS] Token Embedding
         ↓
     Policy Head (MLP)
         ↓
  Discrete Action (0–4)
Action labels

ID	Action
0	Nothing
1	Accelerate
2	Turn Left
3	Turn Right
4	Brake
📈 Highlights
✅ Gitを活用した視覚・言語融合による状況理解

✅ LoRAを使った軽量なファインチューニング

✅ 強化学習（PPO）による行動選択最適化

✅ 自動運転のようなビジョン→アクションへの流れを再現

🔗 Related Work
CarLLaVA (2024)

LoRA: Low-Rank Adaptation

Git by Microsoft

👤 Author
Created by Daiki Matsuba
GitHub: github.com/CreationTheSustainableWorld
Portfolio: https://sites.google.com/view/job-application-portfolio

📝 License
This project is licensed under the MIT License.
