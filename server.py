import os
import sys
import time
from flask import Flask, render_template, request, jsonify

# Ensure imports work regardless of where the script is launched from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from extractive.text_rank import TextRankSummarizer
from abstractive.summarizer import AbstractiveSummarizer

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# 10 MB upload limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

print("Initializing NLP Models... This may take a moment.")
ext_model = TextRankSummarizer()
abs_model = AbstractiveSummarizer(model_path=os.path.join(BASE_DIR, "models", "t5-small"))

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_storage):
    filename = file_storage.filename
    ext = filename.rsplit('.', 1)[1].lower()

    if ext == 'txt':
        raw = file_storage.read()
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1')

    elif ext == 'pdf':
        import io
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required. Run: pip install pypdf")
        reader = PdfReader(io.BytesIO(file_storage.read()))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t.strip())
        return "\n".join(pages_text)

    elif ext == 'docx':
        import io
        try:
            import docx as python_docx
        except ImportError:
            raise ImportError("python-docx is required. Run: pip install python-docx")
        doc = python_docx.Document(io.BytesIO(file_storage.read()))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    else:
        raise ValueError(f"Unsupported file type: .{ext}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    sum_type = data.get('type', 'both')

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    response = {}

    if sum_type in ['extractive', 'both']:
        start = time.time()
        res = ext_model.summarize(text)
        response['extractive'] = {"text": res, "time": f"{time.time() - start:.3f}s"}

    if sum_type in ['abstractive', 'both']:
        start = time.time()
        res = abs_model.summarize(text)
        response['abstractive'] = {"text": res, "time": f"{time.time() - start:.3f}s"}

    return jsonify(response)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported type. Upload PDF, DOCX, or TXT."}), 400

    try:
        extracted = extract_text_from_file(file)
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    if not extracted.strip():
        return jsonify({"error": "File is empty or has no readable text."}), 400

    word_count = len(extracted.split())
    return jsonify({"text": extracted, "word_count": word_count})


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  OFFLINE SUMMARIZER DASHBOARD STARTED")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
