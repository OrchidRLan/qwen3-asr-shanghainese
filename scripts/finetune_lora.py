from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "Qwen/Qwen3-ASR"

print("Loading model...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

print("LoRA applied.")

# ===== Placeholder dataset =====
# TODO: replace with real dataset
train_dataset = []

training_args = TrainingArguments(
    output_dir="results/lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Start training...")
trainer.train()