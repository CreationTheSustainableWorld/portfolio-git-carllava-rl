import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from PIL import Image

# === ハイパーパラメータ ===
MODEL_NAME = "microsoft/git-base"
JSONL_PATH = "CarRacing-Data/caption_data.jsonl"
PROMPT = "What should the vehicle do in this scenario?"  # 固定プロンプト
BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 3
OUTPUT_DIR = "lora_git_caption_model_carllava"

# === デバイス設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === モデルとLoRA構成 ===
print("🔧 モデルとLoRAを準備中...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)
model.to(device)

# === Processor（画像＋テキスト用） ===
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# === JSONL読み込み（image, text） ===
from datasets import load_dataset
raw_dataset = load_dataset("json", data_files=JSONL_PATH, split="train")

# === 前処理関数 ===
def preprocess(example):
    image_path = example["image"]
    image = Image.open(image_path).convert("RGB")
    full_text = f"{PROMPT} {example['text']}"
    inputs = processor(images=image, text=full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

processed = raw_dataset.map(preprocess)

# === PyTorch Dataset に変換 ===
class GitCaptionTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "pixel_values": torch.tensor(item["pixel_values"]),
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"]),
        }

torch_dataset = GitCaptionTorchDataset(processed)

# === DataLoader ===
dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Optimizer & Scheduler ===
optimizer = AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0,
                              num_training_steps=EPOCHS * len(dataloader))

# === 学習ループ ===
print("🚀 学習開始！")
model.train()
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"📉 Epoch {epoch+1} 完了 - 平均Loss: {avg_loss:.4f}")

# === モデル保存 ===
print("💾 LoRAモデル保存中...")
model.save_pretrained(OUTPUT_DIR)
print(f"✅ 保存完了：{OUTPUT_DIR}")
