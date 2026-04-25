import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import Counter


class TextRankSummarizer:
    def __init__(self):
        # Download required NLTK data if not present
        for resource in ['punkt', 'punkt_tab', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource != 'stopwords' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

        self.stop_words = set(stopwords.words('english'))

    def _clean_text(self, text):
        """Basic cleaning of text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def _get_tf_idf_vector(self, sentence, all_words):
        """Calculates a simple frequency vector for a sentence."""
        words = word_tokenize(self._clean_text(sentence))
        words = [w for w in words if w not in self.stop_words]
        word_counts = Counter(words)

        vector = np.zeros(len(all_words))
        for i, word in enumerate(all_words):
            vector[i] = word_counts.get(word, 0)
        return vector

    def _cosine_similarity(self, v1, v2):
        """Calculates cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if not norm_v1 or not norm_v2:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def summarize(self, text, num_sentences=3):
        """
        Main TextRank algorithm:
        1. Tokenize into sentences.
        2. Build similarity matrix.
        3. Run PageRank.
        4. Return top ranked sentences.
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        # Extract all unique words for vectorization
        all_words = set()
        for sent in sentences:
            words = [w for w in word_tokenize(self._clean_text(sent)) if w not in self.stop_words]
            all_words.update(words)

        all_words = sorted(list(all_words))

        # Build Similarity Matrix
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))

        # Pre-calculate vectors to save time
        vectors = [self._get_tf_idf_vector(sent, all_words) for sent in sentences]

        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._cosine_similarity(vectors[i], vectors[j])

        # PageRank algorithm from scratch
        d = 0.85  # damping factor
        scores = np.ones(n)
        threshold = 0.0001

        for _ in range(100):  # max iterations
            prev_scores = np.copy(scores)
            for i in range(n):
                sum_val = 0
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        out_degree = np.sum(similarity_matrix[j])
                        if out_degree > 0:
                            sum_val += (similarity_matrix[j][i] / out_degree) * prev_scores[j]
                scores[i] = (1 - d) + d * sum_val

            if np.linalg.norm(scores - prev_scores) < threshold:
                break

        # Get top ranked sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Return top N sentences in original order
        top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: sentences.index(x[1]))
        return " ".join([s[1] for s in top_sentences])
