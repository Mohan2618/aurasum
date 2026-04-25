import os
import sys
import nltk

# Ensure we run from the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def download_models():
    """
    Downloads models/data and saves them locally for offline use.
    Run this ONCE while connected to the internet before using the project offline.
    """
    # 1. Download NLTK data
    print("Downloading NLTK data (punkt, punkt_tab, stopwords)...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    print("NLTK data downloaded.\n")

    # 2. Download Transformer weights
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    model_name = "t5-small"
    save_path = os.path.join(BASE_DIR, "models", model_name)

    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"Model already exists at {save_path}. Skipping download.")
    else:
        os.makedirs(save_path, exist_ok=True)
        print(f"Downloading {model_name} tokenizer and weights (this may take a few minutes)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print(f"Saving to {save_path}...")
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

    print("\n✅ Setup Complete! You can now run the project offline.")
    print(f"   Model saved at: {os.path.abspath(save_path)}")
    print("\nNext steps:")
    print("  Web UI:  python server.py")
    print("  CLI:     python main.py")


if __name__ == "__main__":
    download_models()
