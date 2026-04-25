AuraSum – Offline Text Summarization System
Overview

AuraSum is a web-based text summarization system that works completely offline. It generates both extractive and abstractive summaries and supports multiple input formats including typed text and uploaded documents.

Features
Extractive summarization using TextRank (implemented from scratch)
Abstractive summarization using T5-Small transformer
Supports PDF, DOCX, and TXT file uploads
Works fully offline
Displays processing time for both methods
Simple and user-friendly web interface

Project Structure:
aurasum/
│
├── server.py
├── setup_offline.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── models/ (downloaded locally)

Installation & Setup:
1. Install dependencies
pip install -r requirements.txt
2. Download model (one-time setup)
python setup_offline.py
3. Download model (one-time setup)
python setup_offline.py
4. Run the application
python server.py
5. Open in browser
http://localhost:5000
Usage
Option 1: Type or paste text
Option 2: Upload document (PDF, DOCX, TXT)

Then:

Select summarization method
Click Summarize

The system generates:

Extractive summary
Abstractive summary
Technologies Used
Python
Flask
NLTK
Hugging Face Transformers
pypdf
python-docx

Notes:
The model is not included in the repository due to size limitations
Run setup_offline.py to download it locally
