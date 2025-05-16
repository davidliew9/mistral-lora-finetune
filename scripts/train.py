import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Check device (MPS or CPU fallback)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Load dataset
dataset = load_dataset("json", data_files="data/custom_alpaca.json")["train"]

# Format the input for T5
def format(example):
    input_text = f"{example['instruction']} {example['input']}"
    target_text = example['output']
    return {"input_text": input_text, "target_text": target_text}

dataset = dataset.map(format)

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# Tokenize
def tokenize(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    tokenized = tokenizer(prompt, truncation=True,
                        padding="max_length", max_length=512)
    labels = tokenizer(example['output'], truncation=True,
                    padding="max_length", max_length=512)
    tokenized["labels"] = labels["input_ids"]
    return tokenized


tokenized_dataset = dataset.map(tokenize)
print(tokenized_dataset[0])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/lora_t5_finetuned",
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    logging_strategy="epoch",
    report_to="none",
)

# Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
