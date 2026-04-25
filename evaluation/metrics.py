from rouge_score import rouge_scorer

class EvaluationMetrics:
    def __init__(self):
        # We'll use ROUGE-1, ROUGE-2, and ROUGE-L
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge(self, reference, generated):
        """Calculates ROUGE scores between a reference summary and a generated summary."""
        scores = self.scorer.score(reference, generated)
        
        # Format the output for easier reading
        results = {
            "ROUGE-1": scores['rouge1'].fmeasure,
            "ROUGE-2": scores['rouge2'].fmeasure,
            "ROUGE-L": scores['rougeL'].fmeasure
        }
        return results

    def compare_summaries(self, reference, extractive_summary, abstractive_summary):
        """Compares extractive and abstractive results against a reference."""
        ext_scores = self.calculate_rouge(reference, extractive_summary)
        abs_scores = self.calculate_rouge(reference, abstractive_summary)
        
        return {
            "extractive": ext_scores,
            "abstractive": abs_scores
        }
