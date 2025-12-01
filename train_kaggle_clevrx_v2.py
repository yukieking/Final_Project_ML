import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import csv

# ================= 配置参数 =================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TRAIN_CSV = "custom_dataset/train_labels.csv"
IMAGE_ROOT_TRAIN = "custom_dataset/train/"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 5            
LEARNING_RATE = 2e-4  
LORA_RANK = 32        
LORA_ALPHA = 64       
PATIENCE = 2          
# ====================================================

# ================= 手动工具函数 (防止报错) =================
def process_vision_info(messages):
    image_inputs = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image":
                    image_inputs.append(item["image"])
    return image_inputs, None
# ========================================================

# 1. 定义数据集
class CLEVRXDataset(Dataset):
    def __init__(self, csv_path, image_root, processor, mode="train"):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, row['file'])
        image = Image.open(image_path).convert("RGB")
        
        question = row['question']
        
        if self.mode == "train":
            answer = row['answer']
            explanation = row['explanation']
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question}\nPlease provide the answer and a detailed explanation in the format:\nAnswer: ...\nExplanation: ..."}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"Answer: {answer}\nExplanation: {explanation}"}
                ]}
            ]
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question}\nPlease provide the answer and a detailed explanation in the format:\nAnswer: ...\nExplanation: ..."}
                ]}
            ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(self.mode != "train")
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        new_inputs = {}
        for k, v in inputs.items():
            if k == 'image_grid_thw':
                new_inputs[k] = v 
            else:
                new_inputs[k] = v.squeeze(0)
        
        return new_inputs

# 2. 初始化模型和处理器
def get_model_and_processor():
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # 【提分关键】增大 LoRA Rank
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,        # 增大到 32
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,  #稍微降低 dropout
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 【激进策略】全模块微调，显存不够的话改回 ["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, processor

# 3. 训练循环 (带 Scheduler 和 Acc 返回)
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_acc = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = torch.nn.utils.rnn.pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=0).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0).to(device)
        pixel_values = torch.cat([b['pixel_values'] for b in batch]).to(device)
        image_grid_thw = torch.cat([b['image_grid_thw'] for b in batch]).to(device)
        
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )
        
        loss = outputs.loss
        
        # --- 计算准确率 ---
        with torch.no_grad():
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            _, preds = torch.max(shift_logits, dim=-1)
            mask = shift_labels != 0 
            correct = (preds == shift_labels) & mask
            num_valid_tokens = mask.sum()
            if num_valid_tokens > 0:
                acc = correct.sum().float() / num_valid_tokens.float()
            else:
                acc = torch.tensor(0.0).to(device)
        # ------------------

        loss.backward()
        optimizer.step()
        scheduler.step() # 【提分关键】更新学习率
        
        total_loss += loss.item()
        total_acc += acc.item()
        
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "acc": f"{acc.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    print(f"Epoch Result -> Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

# 主程序
if __name__ == "__main__":
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = get_model_and_processor()
        
        train_dataset = CLEVRXDataset(TRAIN_CSV, IMAGE_ROOT_TRAIN, processor, mode="train")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x, num_workers=0) 
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # 定义学习率调度器
        num_training_steps = len(train_loader) * EPOCHS
        num_warmup_steps = int(0.1 * num_training_steps) 
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        
        # === 早停与最佳模型记录 ===
        best_acc = 0.0
        patience_counter = 0
        best_model_path = "clevrx_lora_best"
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            
            loss, acc = train(model, train_loader, optimizer, scheduler, device)
            
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0 # 重置计数器
                print(f"发现新高分 (Acc: {best_acc:.4f})！保存模型到 '{best_model_path}'...")
                model.save_pretrained(best_model_path)
                processor.save_pretrained(best_model_path)
            else:
                patience_counter += 1
                print(f"性能未提升 (Best: {best_acc:.4f}). 耐心值: {patience_counter}/{PATIENCE}")
            
            # 早停检查
            if patience_counter >= PATIENCE:
                print(f"触发早停机制！在 Epoch {epoch+1} 停止训练。")
                break
            
        print(f"\nTraining Finished! 最佳模型已保存在: {best_model_path}")
        print("请修改 inference.py 中的 LORA_PATH = 'clevrx_lora_best' 进行推理。")