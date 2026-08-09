from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")
print(dataset)
print(dataset.column_names)

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype='auto')


training_args = TrainingArguments(
    output_dir="./qwen3-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    # gradient_accumulation_steps=8,
    # gradient_checkpointing=False,
    # fp16=True,
    # learning_rate=2e-5,
    logging_steps=10,
    # eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    load_best_model_at_end=True,
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

res = trainer.train()

print(res)