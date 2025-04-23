import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM

class GitRLCarllavaModel(nn.Module):
    def __init__(self, model_path="lora_git_caption_model", action_dim=5, prompt="What should the vehicle do in this scenario?"):
        super().__init__()
        self.vision_language_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.prompt = prompt

        self.policy_head = nn.Sequential(
            nn.Linear(self.vision_language_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, images):
        # images: PIL.Image or numpy RGB image
        inputs = self.processor(
            images=images,
            text=self.prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).to(next(self.parameters()).device)

        # 推論（last_hidden_state 取得）
        outputs = self.vision_language_model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )

        # [CLS]相当の最初のトークンを使う
        cls_token = outputs.hidden_states[-1][:, 0]  # 最終層の[CLS]
        return self.policy_head(cls_token)
