import os
import csv
import torch
import pandas as pd
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from tqdm import tqdm

# =================配置参数=================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = "clevrx_lora_best"
TEST_CSV = "custom_dataset/test_non_labels.csv"
IMAGE_ROOT_TEST = "custom_dataset/test/"
OUTPUT_FILE = "submission2.csv"
# =========================================

# ================= 工具函数 =================
def process_vision_info(messages):
    image_inputs = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image":
                    image_inputs.append(item["image"])
    return image_inputs, None
# ============================================

class CLEVRXTestDataset(Dataset):
    def __init__(self, csv_path, image_root, processor):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, row['file'])
        image = Image.open(image_path).convert("RGB")
        question = row['question']
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{question}\nPlease provide the answer and a detailed explanation in the format:\nAnswer: ...\nExplanation: ..."}
            ]}
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        new_inputs = {}
        for k, v in inputs.items():
            if k == 'image_grid_thw':
                new_inputs[k] = v 
            else:
                new_inputs[k] = v.squeeze(0)
                
        return new_inputs, row['id']

def generate_csv():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not os.path.exists(LORA_PATH):
        print(f"Error: 找不到权重文件夹 {LORA_PATH}")
        return

    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    test_dataset = CLEVRXTestDataset(TEST_CSV, IMAGE_ROOT_TEST, processor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0)

    print(f"Starting inference -> {OUTPUT_FILE}")
    
    # 使用 QUOTE_MINIMAL：这是最标准的 CSV 格式
    # 只有当 Explanation 里包含逗号时，它才会自动加上引号，和 Sample 文件逻辑一致。
    with open(OUTPUT_FILE, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "answer", "explanation"])
        
        for batch in tqdm(test_loader):
            inputs, sample_id = batch[0]
            
            input_ids = inputs['input_ids'].unsqueeze(0).to(device)
            attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
            pixel_values = inputs['pixel_values'].to(device)
            image_grid_thw = inputs['image_grid_thw'].to(device)
            if image_grid_thw.dim() == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=128,
                    repetition_penalty=1.1  # 略微降低惩罚，避免句子过短
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 解析 Answer / Explanation
            ans_part = "unknown"
            exp_part = output_text
            try:
                if "Explanation:" in output_text:
                    parts = output_text.split("Explanation:")
                    ans_part = parts[0].replace("Answer:", "").strip()
                    exp_part = parts[1].strip()
                elif "Answer:" in output_text:
                    ans_part = output_text.replace("Answer:", "").strip()
            except:
                pass
            
            # 清洗列表格式 ['...'] -> 纯文本
            try:
                if exp_part.strip().startswith("[") and exp_part.strip().endswith("]"):
                    parsed = ast.literal_eval(exp_part)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        exp_part = str(parsed[0])
            except:
                pass
            
            # 写入
            writer.writerow([sample_id, ans_part, exp_part])

    print("Done!")

if __name__ == "__main__":
    generate_csv()