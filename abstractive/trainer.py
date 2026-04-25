import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import os


class SummarizationTrainer:
    def __init__(self, model_checkpoint="t5-small", output_dir="models/fine_tuned_t5"):
        self.model_checkpoint = model_checkpoint
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    def preprocess_function(self, examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)

        # Fixed: as_target_tokenizer() is deprecated — use text_target parameter instead
        labels = self.tokenizer(
            text_target=examples["highlights"],
            max_length=128,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, tokenized_datasets, epochs=3, batch_size=8):
        """Standard training loop using Seq2SeqTrainer."""
        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",          # Fixed: evaluation_strategy -> eval_strategy
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none"
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        print("Starting training...")
        trainer.train()

        print(f"Saving fine-tuned model to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
