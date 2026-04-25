import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AbstractiveSummarizer:
    def __init__(self, model_path="models/t5-small"):
        """
        Loads the transformer model from a local directory.
        If the model doesn't exist, it provides instructions.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Resolve to absolute path so it works regardless of working directory
        self.model_path = os.path.abspath(model_path)

        if not os.path.exists(self.model_path):
            print(f"Warning: Local model not found at {self.model_path}")
            print("Please run: python setup_offline.py")
            self.tokenizer = None
            self.model = None
            return

        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True).to(self.device)
        print("Model loaded successfully.")

    def summarize(self, text, max_length=150, min_length=40, num_beams=4):
        """Generates summary using beam search."""
        if self.model is None:
            return "Error: Model not loaded. Please run setup_offline.py first."

        # T5 requires "summarize: " prefix
        input_text = "summarize: " + text

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=num_beams,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
