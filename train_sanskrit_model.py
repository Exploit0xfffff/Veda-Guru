import os
import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load the pre-trained BERT model and tokenizer from local files
model_dir = "./"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForMaskedLM.from_pretrained(model_dir)

# Load the preprocessed Vedic texts
preprocessed_dir = "./preprocessed"
files = [os.path.join(preprocessed_dir, f) for f in os.listdir(preprocessed_dir) if f.endswith(".txt")]

# Read the preprocessed texts into a list
texts = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        texts.append(f.read())

# Create a dataset from the texts
dataset = Dataset.from_dict({"text": texts})

# Split the dataset into training and validation sets
train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    # Create labels for MLM
    labels = torch.tensor(tokenized_inputs["input_ids"]).clone()
    # Mask 15% of the tokens for MLM
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
