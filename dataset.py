import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

class GitCaptionDataset(Dataset):
    def __init__(self, jsonl_path, processor_name="microsoft/git-base"):
        self.data = []
        self.processor = AutoProcessor.from_pretrained(processor_name)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.data.append({
                    "image_path": entry["image"],
                    "text": entry["text"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        text = entry["text"]

        encoding = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True)
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }
