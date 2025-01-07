from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

model_name = 'AI-Growth-Lab/PatentSBERTa'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        examples['text_b'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Map the dataset through the preprocess function
encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)

# Important: set the format of the dataset so it can be fed into the Trainer
encoded_train_dataset = encoded_train_dataset.rename_column("label", "labels")
encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# -------------------------------------------------
# 5) Define training arguments
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert-finetuned",
    evaluation_strategy="no",   # "no", "steps", or "epoch"
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=100,
    logging_steps=50,
    learning_rate=1e-5
)

# -------------------------------------------------
# 6) Create Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    tokenizer=tokenizer
)

# -------------------------------------------------
# 7) Train
# -------------------------------------------------
trainer.train()

# -------------------------------------------------
# 8) (Optional) Predict or evaluate
# -------------------------------------------------
# Once trained, you can run predictions on new data.
# For demonstration, let's predict on the training set itself:
predictions = trainer.predict(encoded_train_dataset)
predicted_logits = predictions.predictions
predicted_labels = torch.argmax(torch.tensor(predicted_logits), dim=1).tolist()

print("Predicted labels:", predicted_labels)
print("True labels:     ", df['label'].tolist())