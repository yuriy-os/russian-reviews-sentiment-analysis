from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased-conversational", num_labels=3
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        "DeepPavlov/rubert-base-cased-conversational"
    )

    finetune_dataset = load_dataset("csv", data_files="./dataset.csv", split=["train"])[0].train_test_split()
    train_dataset, test_dataset = finetune_dataset["train"], finetune_dataset["test"]

    train_dataset = train_dataset.map(
        tokenize, batched=True, batch_size=len(train_dataset)
    )
    test_dataset = test_dataset.map(
        tokenize, batched=True, batch_size=len(train_dataset)
    )
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_gpu_train_batch_size=16,
        per_gpu_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        evaluate_during_training=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    print(trainer.evaluate())
