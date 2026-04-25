import os
import sys
import time
import argparse

# Fix: ensure imports work from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from extractive.text_rank import TextRankSummarizer
from abstractive.summarizer import AbstractiveSummarizer
from evaluation.metrics import EvaluationMetrics


def main():
    parser = argparse.ArgumentParser(description="Offline Text Summarization Tool")
    parser.add_argument("--type", choices=["extractive", "abstractive", "both"], default="both",
                        help="Type of summarization to perform")
    parser.add_argument("--sentences", type=int, default=3, help="Number of sentences for extractive summary")
    parser.add_argument("--max_len", type=int, default=150, help="Max length for abstractive summary")

    args = parser.parse_args()

    print("\n--- Offline Text Summarizer CLI ---")
    print("Paste your text below (Press Ctrl+D on Linux/Mac or Ctrl+Z then Enter on Windows to finish):")

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        text = "\n".join(lines).strip()

    if not text:
        print("No text provided. Exiting.")
        return

    # Initialize models
    ext_model = TextRankSummarizer()
    abs_model = AbstractiveSummarizer(model_path=os.path.join(BASE_DIR, "models", "t5-small"))
    eval_tool = EvaluationMetrics()

    results = {}

    # 1. Extractive Summarization
    if args.type in ["extractive", "both"]:
        print("\n[Running Extractive Summarization (TextRank)...]")
        start_time = time.time()
        ext_summary = ext_model.summarize(text, num_sentences=args.sentences)
        ext_time = time.time() - start_time
        print("\nExtractive Summary:")
        print("-" * 30)
        print(ext_summary)
        print("-" * 30)
        print(f"Time Taken: {ext_time:.4f} seconds")
        results['extractive'] = (ext_summary, ext_time)

    # 2. Abstractive Summarization
    if args.type in ["abstractive", "both"]:
        print("\n[Running Abstractive Summarization (Transformer)...]")
        start_time = time.time()
        abs_summary = abs_model.summarize(text, max_length=args.max_len)
        abs_time = time.time() - start_time
        print("\nAbstractive Summary:")
        print("-" * 30)
        print(abs_summary)
        print("-" * 30)
        print(f"Time Taken: {abs_time:.4f} seconds")
        results['abstractive'] = (abs_summary, abs_time)

    # Comparison Table
    if args.type == "both":
        print("\n--- PERFORMANCE COMPARISON ---")
        print(f"{'Method':<15} | {'Execution Time':<15}")
        print("-" * 35)
        print(f"{'Extractive':<15} | {results['extractive'][1]:.4f}s")
        print(f"{'Abstractive':<15} | {results['abstractive'][1]:.4f}s")


if __name__ == "__main__":
    main()
