"""Stub module for t5 (has Windows-incompatible dependencies).

This provides minimal implementations to allow the code to load.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import functools


# ============================================================================
# t5.data module stubs
# ============================================================================

class Feature:
    """Stub for t5.data.Feature."""
    
    def __init__(self, vocabulary=None, add_eos: bool = True, required: bool = True, dtype: str = "int32"):
        self.vocabulary = vocabulary
        self.add_eos = add_eos
        self.required = required
        self.dtype = dtype


def rate_num_examples(task=None, maximum: int = None, temperature: float = 1.0, scale: float = 1.0):
    """Stub for t5.data.rate_num_examples."""
    return 1.0


class preprocessors:
    """Stub for t5.data.preprocessors."""
    
    @staticmethod
    def rekey(dataset, key_map: Dict[str, Optional[str]]):
        """Stub rekey preprocessor."""
        return dataset
    
    @staticmethod
    def tokenize(dataset, output_features, **kwargs):
        """Stub tokenize preprocessor."""
        return dataset
    
    @staticmethod
    def append_eos_after_trim(dataset, output_features, **kwargs):
        """Stub append_eos_after_trim preprocessor."""
        return dataset
    
    @staticmethod
    def span_corruption(dataset, **kwargs):
        """Stub span_corruption preprocessor."""
        return dataset
    
    @staticmethod
    def select_random_chunk(dataset, **kwargs):
        """Stub select_random_chunk preprocessor."""
        return dataset
    
    @staticmethod
    def reduce_concat_tokens(dataset, **kwargs):
        """Stub reduce_concat_tokens preprocessor."""
        return dataset
    
    @staticmethod
    def split_tokens(dataset, **kwargs):
        """Stub split_tokens preprocessor."""
        return dataset
    
    @staticmethod
    def denoise(dataset, **kwargs):
        """Stub denoise preprocessor."""
        return dataset
    
    @staticmethod
    def glue(dataset, benchmark_name=None, label_names=None, **kwargs):
        """Stub glue preprocessor."""
        return dataset
    
    @staticmethod
    def summarize(dataset, article_key=None, summary_key=None, **kwargs):
        """Stub summarize preprocessor."""
        return dataset
    
    @staticmethod
    def definite_pronoun_resolution_simple(dataset, **kwargs):
        """Stub definite_pronoun_resolution_simple preprocessor."""
        return dataset
    
    @staticmethod
    def wsc_simple(dataset, correct_referent_only=True, **kwargs):
        """Stub wsc_simple preprocessor."""
        return dataset


class postprocessors:
    """Stub for t5.data.postprocessors."""
    
    @staticmethod
    def string_label_to_class_id(string_label, label_classes=None, **kwargs):
        """Stub string_label_to_class_id postprocessor."""
        if label_classes and string_label in label_classes:
            return label_classes.index(string_label)
        return -1
    
    @staticmethod
    def qa(answer, **kwargs):
        """Stub qa postprocessor."""
        return answer
    
    @staticmethod
    def lower_text(text, **kwargs):
        """Stub lower_text postprocessor."""
        return text.lower() if isinstance(text, str) else text
    
    @staticmethod
    def wsc_simple(output, **kwargs):
        """Stub wsc_simple postprocessor."""
        return output


class tasks:
    """Stub for t5.data.tasks - empty placeholder."""
    pass


class glue_utils:
    """Stub for t5.data.glue_utils."""
    
    @staticmethod
    def get_glue_text_preprocessor(benchmark):
        """Stub get_glue_text_preprocessor."""
        def preprocessor(dataset, **kwargs):
            return dataset
        return preprocessor
    
    @staticmethod
    def get_glue_metric(benchmark_name):
        """Stub get_glue_metric."""
        return [metrics.accuracy]
    
    @staticmethod
    def get_super_glue_metric(benchmark_name):
        """Stub get_super_glue_metric."""
        return [metrics.accuracy]
    
    @staticmethod
    def get_glue_postprocess_fn(benchmark):
        """Stub get_glue_postprocess_fn."""
        def postprocess(output, **kwargs):
            return output
        return postprocess
    
    @staticmethod
    def get_glue_weight_mapping():
        """Stub get_glue_weight_mapping."""
        return {}


class data:
    """Namespace for t5.data."""
    Feature = Feature
    rate_num_examples = staticmethod(rate_num_examples)
    preprocessors = preprocessors
    postprocessors = postprocessors
    glue_utils = glue_utils
    tasks = tasks
    
    @staticmethod
    def get_super_glue_weight_mapping():
        """Stub get_super_glue_weight_mapping."""
        return {}
    
    @staticmethod
    def get_glue_weight_mapping():
        """Stub get_glue_weight_mapping."""
        return {}


# ============================================================================
# t5.evaluation module stubs  
# ============================================================================

class qa_utils:
    """Stub for t5.evaluation.qa_utils."""
    
    @staticmethod
    def normalize_squad(text: str) -> str:
        """Basic text normalization."""
        import re
        import string
        text = text.lower()
        text = ''.join(c for c in text if c not in string.punctuation)
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def qa_metrics(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Compute basic QA metrics (F1 and EM)."""
        from collections import Counter
        
        def get_tokens(s):
            return s.split()
        
        def compute_f1(pred_tokens, gold_tokens):
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gold_tokens) if gold_tokens else 0
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        def compute_em(pred, gold):
            return float(pred.strip() == gold.strip())
        
        total_f1 = 0.0
        total_em = 0.0
        
        for target_list, pred in zip(targets, predictions):
            pred_tokens = get_tokens(pred)
            max_f1 = 0.0
            max_em = 0.0
            for target in target_list:
                target_tokens = get_tokens(target)
                f1 = compute_f1(pred_tokens, target_tokens)
                em = compute_em(pred, target)
                max_f1 = max(max_f1, f1)
                max_em = max(max_em, em)
            total_f1 += max_f1
            total_em += max_em
        
        n = len(predictions) if predictions else 1
        return {
            "f1": total_f1 / n * 100,
            "em": total_em / n * 100
        }


class metrics:
    """Stub for t5.evaluation.metrics."""
    
    @staticmethod
    def bleu(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Simple BLEU stub."""
        return {"bleu": 0.0}
    
    @staticmethod
    def rouge(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Simple ROUGE stub."""
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    @staticmethod
    def accuracy(targets: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute accuracy."""
        if not targets:
            return {"accuracy": 0.0}
        correct = sum(1 for t, p in zip(targets, predictions) if t.strip() == p.strip())
        return {"accuracy": correct / len(targets) * 100}
    
    @staticmethod
    def squad(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Compute SQuAD-style F1 and EM metrics."""
        return qa_utils.qa_metrics(targets, predictions)


class evaluation:
    """Namespace for t5.evaluation."""
    qa_utils = qa_utils
    metrics = metrics


print("[INFO] Using t5 stub module - full functionality requires Linux with tensorflow-text")
