
import argparse
import os
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)
from transformers.trainer_utils import set_seed
from sklearn.metrics import accuracy_score
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--compare_runs", nargs=2, help="Compare best and worst run directories")
    return parser.parse_args()


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc}


def preprocess_data(dataset, tokenizer, max_samples, pad=True):
    def tokenize(example):
        return tokenizer(
            example["sentence1"], 
            example["sentence2"], 
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length" if pad else False
        )

    if max_samples != -1:
        dataset = dataset.select(range(max_samples))

    return dataset.map(tokenize, batched=True)


def compare_runs(best_dir, worst_dir, val_dataset):
    def load_val_preds(path):
        with open(path) as f:
            return [line.strip().split("###") for line in f.readlines()]

    best = load_val_preds(os.path.join(best_dir, "val_predictions.txt"))
    worst = load_val_preds(os.path.join(worst_dir, "val_predictions.txt"))

    table = wandb.Table(columns=["sentence1", "sentence2", "label", "best_pred", "worst_pred"])

    for b, w in zip(best, worst):
        s1_b, s2_b, label_b, pred_b = b
        s1_w, s2_w, label_w, pred_w = w
        if pred_b == label_b and pred_w != label_w:
            table.add_data(s1_b, s2_b, label_b, pred_b, pred_w)

    wandb.log({"qualitative_hard_cases": table})


def main():
    args = parse_args()
    wandb.init(project="anlp_ex1_mrpc", config=vars(args))
    run = wandb.init(
        entity="me_myself_and_I",
        project="Advanced_NLP_HW1",
        config=vars(args)
    )

    set_seed(42)

    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = preprocess_data(dataset["train"], tokenizer, args.max_train_samples)
    eval_dataset = preprocess_data(dataset["validation"], tokenizer, args.max_eval_samples)

    data_collator = DataCollatorWithPadding(tokenizer)

    output_dir = f'saved_models/{wandb.run.name}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to=["wandb"],
        logging_steps=1,
        save_strategy="no",
        run_name=f"lr_{args.lr}_bs_{args.batch_size}_ep_{args.num_train_epochs}",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.do_train:

        trainer.train()

        eval_result = trainer.evaluate()
        eval_acc = eval_result["eval_accuracy"]
        config_str = f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {eval_acc:.4f}"

        with open(f"{output_dir}/res.txt", "a") as f:
            f.write(config_str + "\n")

        val_preds = trainer.predict(eval_dataset)
        val_labels = val_preds.label_ids
        val_outputs = np.argmax(val_preds.predictions, axis=1)

        with open(f"{output_dir}/val_predictions.txt", "w") as f:
            for i, pred in enumerate(val_outputs):
                label = val_labels[i]
                s1 = dataset["validation"][i]["sentence1"]
                s2 = dataset["validation"][i]["sentence2"]
                f.write(f"{s1}###{s2}###{label}###{pred}\n")

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    if args.do_predict:
        model.eval()

        test_dataset = preprocess_data(dataset["test"], tokenizer, args.max_predict_samples, pad=False)
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        with open(f"{output_dir}/predictions.txt", "w") as f:
            for i, pred in enumerate(preds):
                s1 = dataset["test"][i]["sentence1"]
                s2 = dataset["test"][i]["sentence2"]
                f.write(f"{s1}###{s2}###{pred}\n")

        # Optional: Log to wandb
        table = wandb.Table(columns=["sentence1", "sentence2", "prediction"])
        for i, pred in enumerate(preds):
            s1 = dataset["test"][i]["sentence1"]
            s2 = dataset["test"][i]["sentence2"]
            table.add_data(s1, s2, int(pred))
        wandb.log({"test_predictions": table})

    if args.compare_runs:
        best_dir, worst_dir = args.compare_runs
        val_dataset = dataset["validation"]
        compare_runs(best_dir, worst_dir, val_dataset)

if __name__ == "__main__":
    main()
