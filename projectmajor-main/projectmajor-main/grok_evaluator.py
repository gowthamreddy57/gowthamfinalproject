# Copyright 2026 mT5 + Grok Integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Evaluation Runner using Grok API for mT5-style tasks.

This module provides tools to evaluate NLP models using Grok as a judge,
supporting all major mT5 benchmark tasks.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

try:
    from multilingual_t5.grok_client import (
        GrokClient, GrokConfig, GrokModel, TaskType, 
        EvaluationResult, create_client
    )
except ImportError:
    from grok_client import (
        GrokClient, GrokConfig, GrokModel, TaskType, 
        EvaluationResult, create_client
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    task_type: TaskType
    languages: List[str]
    output_dir: str = "./eval_results"
    save_individual: bool = True
    strict_mode: bool = True  # Harsh evaluation
    batch_size: int = 10
    max_samples: Optional[int] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    task_name: str
    timestamp: str
    total_samples: int
    overall_score: float
    accuracy: float
    metrics_by_language: Dict[str, Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    errors: List[Dict[str, Any]]
    config: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    def save(self, filepath: str):
        """Save report to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def print_summary(self):
        """Print a formatted summary of the evaluation."""
        print("\n" + "="*80)
        print(f"🔬 EVALUATION REPORT: {self.task_name}")
        print("="*80)
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"📊 Total Samples: {self.total_samples}")
        print(f"⭐ Overall Score: {self.overall_score:.2%}")
        print(f"✅ Accuracy: {self.accuracy:.2%}")
        print("-"*40)
        print("📈 Aggregate Metrics:")
        for metric, value in self.aggregate_metrics.items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"   {metric:20s}: {bar} {value:.2%}")
        print("-"*40)
        print("🌍 Metrics by Language:")
        for lang, metrics in self.metrics_by_language.items():
            avg = sum(metrics.values()) / len(metrics) if metrics else 0
            print(f"   {lang}: {avg:.2%}")
        if self.errors:
            print("-"*40)
            print(f"⚠️  Errors: {len(self.errors)} issues found")
        print("="*80 + "\n")


class EvaluationRunner:
    """Runner for evaluating NLP tasks using Grok API."""
    
    # Language names for reporting
    LANGUAGE_NAMES = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "ru": "Russian",
        "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "it": "Italian",
        "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
        "th": "Thai", "el": "Greek", "bg": "Bulgarian", "uk": "Ukrainian",
        "he": "Hebrew", "sw": "Swahili", "ur": "Urdu", "bn": "Bengali"
    }
    
    def __init__(
        self,
        grok_client: Optional[GrokClient] = None,
        api_key: Optional[str] = None
    ):
        """Initialize the evaluation runner.
        
        Args:
            grok_client: Optional pre-configured GrokClient.
            api_key: Optional API key (used if grok_client not provided).
        """
        self.client = grok_client or create_client(api_key)
        self.results_cache = []
    
    def _create_output_dir(self, output_dir: str) -> Path:
        """Create output directory if it doesn't exist."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # =========================================================================
    # XQuAD / MLQA Style QA Evaluation
    # =========================================================================
    
    def evaluate_qa(
        self,
        predictions: List[Dict[str, Any]],
        gold_answers: Optional[List[str]] = None,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationReport:
        """Evaluate QA predictions harshly.
        
        Args:
            predictions: List of dicts with keys:
                - question: str
                - context: str
                - predicted_answer: str
                - language: str (optional)
            gold_answers: Optional list of gold answers.
            config: Optional evaluation configuration.
            
        Returns:
            EvaluationReport with detailed results.
        """
        config = config or EvaluationConfig(
            task_type=TaskType.QA,
            languages=["en"]
        )
        
        results_by_lang = {}
        all_results = []
        errors = []
        
        for i, pred in enumerate(predictions):
            if config.max_samples and i >= config.max_samples:
                break
            
            lang = pred.get("language", "en")
            gold = gold_answers[i] if gold_answers and i < len(gold_answers) else None
            
            try:
                result = self.client.evaluate_qa_answer(
                    question=pred["question"],
                    context=pred["context"],
                    predicted_answer=pred["predicted_answer"],
                    gold_answer=gold
                )
                
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(result)
                all_results.append(result)
                
                logger.info(f"[{i+1}/{len(predictions)}] {lang}: Score={result.score:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                errors.append({"index": i, "error": str(e), "sample": pred})
        
        return self._create_report(
            task_name="Question Answering (XQuAD/MLQA)",
            results_by_lang=results_by_lang,
            all_results=all_results,
            errors=errors,
            config=config
        )
    
    # =========================================================================
    # XNLI Style NLI Evaluation
    # =========================================================================
    
    def evaluate_nli(
        self,
        predictions: List[Dict[str, Any]],
        gold_labels: Optional[List[str]] = None,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationReport:
        """Evaluate NLI predictions.
        
        Args:
            predictions: List of dicts with keys:
                - premise: str
                - hypothesis: str
                - predicted_label: str
                - language: str (optional)
            gold_labels: Optional list of gold labels.
            config: Optional evaluation configuration.
            
        Returns:
            EvaluationReport with detailed results.
        """
        config = config or EvaluationConfig(
            task_type=TaskType.NLI,
            languages=["en"]
        )
        
        results_by_lang = {}
        all_results = []
        errors = []
        
        for i, pred in enumerate(predictions):
            if config.max_samples and i >= config.max_samples:
                break
            
            lang = pred.get("language", "en")
            gold = gold_labels[i] if gold_labels and i < len(gold_labels) else None
            
            try:
                # Get Grok's classification
                response = self.client.natural_language_inference(
                    premise=pred["premise"],
                    hypothesis=pred["hypothesis"],
                    language=lang
                )
                
                grok_label = response.text.strip().lower()
                pred_label = pred["predicted_label"].strip().lower()
                
                # Check if prediction matches Grok's judgment
                is_correct = pred_label == grok_label
                if gold:
                    is_correct = pred_label == gold.strip().lower()
                
                result = EvaluationResult(
                    score=1.0 if is_correct else 0.0,
                    metrics={"accuracy": 1.0 if is_correct else 0.0},
                    feedback=f"Predicted: {pred_label}, Expected: {gold or grok_label}",
                    is_correct=is_correct,
                    details={
                        "predicted": pred_label,
                        "expected": gold or grok_label,
                        "grok_judgment": grok_label
                    }
                )
                
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(result)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                errors.append({"index": i, "error": str(e)})
        
        return self._create_report(
            task_name="Natural Language Inference (XNLI)",
            results_by_lang=results_by_lang,
            all_results=all_results,
            errors=errors,
            config=config
        )
    
    # =========================================================================
    # WikiANN Style NER Evaluation
    # =========================================================================
    
    def evaluate_ner(
        self,
        predictions: List[Dict[str, Any]],
        gold_entities: Optional[List[str]] = None,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationReport:
        """Evaluate NER predictions harshly.
        
        Args:
            predictions: List of dicts with keys:
                - text: str
                - predicted_entities: str (format: "TYPE: entity $$ ...")
                - language: str (optional)
            gold_entities: Optional list of gold entity strings.
            config: Optional evaluation configuration.
            
        Returns:
            EvaluationReport with detailed results.
        """
        config = config or EvaluationConfig(
            task_type=TaskType.NER,
            languages=["en"]
        )
        
        results_by_lang = {}
        all_results = []
        errors = []
        
        for i, pred in enumerate(predictions):
            if config.max_samples and i >= config.max_samples:
                break
            
            lang = pred.get("language", "en")
            gold = gold_entities[i] if gold_entities and i < len(gold_entities) else None
            
            try:
                result = self.client.evaluate_ner(
                    text=pred["text"],
                    predicted_entities=pred["predicted_entities"],
                    gold_entities=gold
                )
                
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(result)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                errors.append({"index": i, "error": str(e)})
        
        return self._create_report(
            task_name="Named Entity Recognition (WikiANN)",
            results_by_lang=results_by_lang,
            all_results=all_results,
            errors=errors,
            config=config
        )
    
    # =========================================================================
    # Translation Evaluation
    # =========================================================================
    
    def evaluate_translation(
        self,
        predictions: List[Dict[str, Any]],
        references: Optional[List[str]] = None,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationReport:
        """Evaluate translation quality harshly.
        
        Args:
            predictions: List of dicts with keys:
                - source_text: str
                - translation: str
                - source_lang: str
                - target_lang: str
            references: Optional list of reference translations.
            config: Optional evaluation configuration.
            
        Returns:
            EvaluationReport with detailed results.
        """
        config = config or EvaluationConfig(
            task_type=TaskType.TRANSLATION,
            languages=["en", "es"]
        )
        
        results_by_lang = {}
        all_results = []
        errors = []
        
        for i, pred in enumerate(predictions):
            if config.max_samples and i >= config.max_samples:
                break
            
            lang_pair = f"{pred.get('source_lang', 'en')}-{pred.get('target_lang', 'es')}"
            ref = references[i] if references and i < len(references) else None
            
            try:
                result = self.client.evaluate_translation(
                    source_text=pred["source_text"],
                    translation=pred["translation"],
                    reference=ref,
                    source_lang=pred.get("source_lang", "en"),
                    target_lang=pred.get("target_lang", "es")
                )
                
                if lang_pair not in results_by_lang:
                    results_by_lang[lang_pair] = []
                results_by_lang[lang_pair].append(result)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                errors.append({"index": i, "error": str(e)})
        
        return self._create_report(
            task_name="Translation Quality",
            results_by_lang=results_by_lang,
            all_results=all_results,
            errors=errors,
            config=config
        )
    
    # =========================================================================
    # PAWS-X Style Paraphrase Evaluation
    # =========================================================================
    
    def evaluate_paraphrase(
        self,
        predictions: List[Dict[str, Any]],
        gold_labels: Optional[List[bool]] = None,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationReport:
        """Evaluate paraphrase detection.
        
        Args:
            predictions: List of dicts with keys:
                - sentence1: str
                - sentence2: str
                - predicted_label: bool
                - language: str (optional)
            gold_labels: Optional list of gold labels.
            config: Optional evaluation configuration.
            
        Returns:
            EvaluationReport with detailed results.
        """
        config = config or EvaluationConfig(
            task_type=TaskType.PARAPHRASE,
            languages=["en"]
        )
        
        results_by_lang = {}
        all_results = []
        errors = []
        
        for i, pred in enumerate(predictions):
            if config.max_samples and i >= config.max_samples:
                break
            
            lang = pred.get("language", "en")
            gold = gold_labels[i] if gold_labels and i < len(gold_labels) else None
            
            try:
                response = self.client.paraphrase_detection(
                    sentence1=pred["sentence1"],
                    sentence2=pred["sentence2"],
                    language=lang
                )
                
                grok_is_paraphrase = "paraphrase" in response.text.lower() and "not" not in response.text.lower()
                pred_is_paraphrase = pred["predicted_label"]
                
                is_correct = pred_is_paraphrase == (gold if gold is not None else grok_is_paraphrase)
                
                result = EvaluationResult(
                    score=1.0 if is_correct else 0.0,
                    metrics={"accuracy": 1.0 if is_correct else 0.0},
                    feedback=f"Predicted: {pred_is_paraphrase}, Expected: {gold if gold is not None else grok_is_paraphrase}",
                    is_correct=is_correct,
                    details={"grok_judgment": grok_is_paraphrase}
                )
                
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(result)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                errors.append({"index": i, "error": str(e)})
        
        return self._create_report(
            task_name="Paraphrase Detection (PAWS-X)",
            results_by_lang=results_by_lang,
            all_results=all_results,
            errors=errors,
            config=config
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _create_report(
        self,
        task_name: str,
        results_by_lang: Dict[str, List[EvaluationResult]],
        all_results: List[EvaluationResult],
        errors: List[Dict[str, Any]],
        config: EvaluationConfig
    ) -> EvaluationReport:
        """Create an evaluation report from results."""
        
        # Calculate aggregate metrics
        if all_results:
            overall_score = sum(r.score for r in all_results) / len(all_results)
            accuracy = sum(1 for r in all_results if r.is_correct) / len(all_results)
            
            # Aggregate all metrics
            all_metrics = {}
            for result in all_results:
                for key, value in result.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
            aggregate_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        else:
            overall_score = 0.0
            accuracy = 0.0
            aggregate_metrics = {}
        
        # Calculate per-language metrics
        metrics_by_lang = {}
        for lang, results in results_by_lang.items():
            if results:
                lang_metrics = {}
                for result in results:
                    for key, value in result.metrics.items():
                        if key not in lang_metrics:
                            lang_metrics[key] = []
                        lang_metrics[key].append(value)
                metrics_by_lang[lang] = {k: sum(v) / len(v) for k, v in lang_metrics.items()}
        
        return EvaluationReport(
            task_name=task_name,
            timestamp=datetime.now().isoformat(),
            total_samples=len(all_results),
            overall_score=overall_score,
            accuracy=accuracy,
            metrics_by_language=metrics_by_lang,
            aggregate_metrics=aggregate_metrics,
            errors=errors,
            config=asdict(config) if hasattr(config, '__dict__') else {}
        )
    
    def run_full_evaluation(
        self,
        task_type: TaskType,
        data_path: str,
        output_dir: str = "./eval_results"
    ) -> EvaluationReport:
        """Run a full evaluation on a dataset file.
        
        Args:
            task_type: Type of task to evaluate.
            data_path: Path to JSON file with predictions.
            output_dir: Directory to save results.
            
        Returns:
            EvaluationReport with complete results.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = data.get("predictions", data)
        gold = data.get("gold", None)
        
        task_methods = {
            TaskType.QA: self.evaluate_qa,
            TaskType.NLI: self.evaluate_nli,
            TaskType.NER: self.evaluate_ner,
            TaskType.TRANSLATION: self.evaluate_translation,
            TaskType.PARAPHRASE: self.evaluate_paraphrase,
        }
        
        if task_type not in task_methods:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        config = EvaluationConfig(
            task_type=task_type,
            languages=list(set(p.get("language", "en") for p in predictions)),
            output_dir=output_dir
        )
        
        report = task_methods[task_type](predictions, gold, config)
        
        # Save report
        output_path = self._create_output_dir(output_dir)
        report_file = output_path / f"{task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.save(str(report_file))
        
        report.print_summary()
        logger.info(f"Report saved to: {report_file}")
        
        return report


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate NLP predictions using Grok API")
    parser.add_argument("--task", type=str, required=True,
                        choices=["qa", "nli", "ner", "translation", "paraphrase"],
                        help="Task type to evaluate")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to JSON file with predictions")
    parser.add_argument("--output", type=str, default="./eval_results",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    
    args = parser.parse_args()
    
    task_map = {
        "qa": TaskType.QA,
        "nli": TaskType.NLI,
        "ner": TaskType.NER,
        "translation": TaskType.TRANSLATION,
        "paraphrase": TaskType.PARAPHRASE
    }
    
    runner = EvaluationRunner()
    report = runner.run_full_evaluation(
        task_type=task_map[args.task],
        data_path=args.data,
        output_dir=args.output
    )
