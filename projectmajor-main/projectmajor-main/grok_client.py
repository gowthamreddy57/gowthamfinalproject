# Copyright 2026 mT5 + Grok Integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grok API Client for Multilingual T5 Tasks.

This module provides a client for interacting with the Grok API to perform
multilingual NLP tasks including:
- Text generation and completion
- Translation
- Question answering (XQuAD, MLQA, TyDiQA style)
- Natural Language Inference (XNLI)
- Paraphrase detection (PAWS-X)
- Named Entity Recognition (WikiANN)
- Text evaluation and scoring
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrokModel(Enum):
    """Available Grok models."""
    GROK_1 = "grok-1"
    GROK_2 = "grok-2"
    GROK_2_MINI = "grok-2-mini"
    GROK_BETA = "grok-beta"


class TaskType(Enum):
    """Supported mT5-style task types."""
    TRANSLATION = "translation"
    QA = "question_answering"
    NLI = "natural_language_inference"
    NER = "named_entity_recognition"
    PARAPHRASE = "paraphrase_detection"
    SUMMARIZATION = "summarization"
    TEXT_GENERATION = "text_generation"
    CUSTOM = "custom"


@dataclass
class GrokConfig:
    """Configuration for Grok API client."""
    api_key: str = field(default_factory=lambda: os.getenv("GROK_API_KEY", ""))
    base_url: str = "https://api.x.ai/v1"
    model: GrokModel = GrokModel.GROK_2
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60  # Requests per minute

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "GROK_API_KEY environment variable not set. "
                "Set it using: export GROK_API_KEY='your-api-key' "
                "or pass api_key to GrokConfig."
            )


@dataclass
class GrokResponse:
    """Response from Grok API."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Dict[str, Any]
    latency_ms: float


@dataclass
class EvaluationResult:
    """Result of evaluating model output."""
    score: float
    metrics: Dict[str, float]
    feedback: str
    is_correct: bool
    details: Dict[str, Any]


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = []
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded."""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [r for r in self.requests if now - r < 60]
        
        if len(self.requests) >= self.rpm:
            sleep_time = 60 - (now - self.requests[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.requests.append(time.time())


class GrokClient:
    """Client for interacting with Grok API for mT5-style tasks."""
    
    def __init__(self, config: Optional[GrokConfig] = None):
        """Initialize the Grok client.
        
        Args:
            config: GrokConfig object. If None, uses defaults with env var API key.
        """
        self.config = config or GrokConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_rpm)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> GrokResponse:
        """Make a request to the Grok API with retries.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            GrokResponse object with the API response.
        """
        self.rate_limiter.wait_if_needed()
        
        payload = {
            "model": self.config.model.value,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
            **kwargs
        }
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    timeout=self.config.timeout
                )
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 429:
                    # Rate limited, wait and retry
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return GrokResponse(
                    text=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=data["choices"][0].get("finish_reason", "unknown"),
                    raw_response=data,
                    latency_ms=latency_ms
                )
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"All retries failed. Last error: {last_error}")
    
    # =========================================================================
    # Core NLP Task Methods (mT5-style)
    # =========================================================================
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None
    ) -> GrokResponse:
        """Translate text from source to target language.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (e.g., 'en', 'zh', 'es').
            target_lang: Target language code.
            context: Optional context to improve translation quality.
            
        Returns:
            GrokResponse with translated text.
        """
        prompt = f"Translate from {source_lang} to {target_lang}:\n\n{text}"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        messages = [
            {"role": "system", "content": "You are an expert multilingual translator. Provide accurate, natural translations preserving the original meaning, tone, and style."},
            {"role": "user", "content": prompt}
        ]
        
        return self._make_request(messages, temperature=0.3)
    
    def question_answering(
        self,
        question: str,
        context: str,
        language: Optional[str] = None
    ) -> GrokResponse:
        """Answer a question based on context (XQuAD/MLQA style).
        
        Args:
            question: The question to answer.
            context: The context/passage containing the answer.
            language: Optional language code for the task.
            
        Returns:
            GrokResponse with the answer.
        """
        lang_instruction = f" Answer in {language}." if language else ""
        
        messages = [
            {"role": "system", "content": f"You are a precise question-answering system. Extract the answer from the given context. If the answer is not in the context, say 'unanswerable'.{lang_instruction}"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        return self._make_request(messages, temperature=0.1)
    
    def natural_language_inference(
        self,
        premise: str,
        hypothesis: str,
        language: Optional[str] = None
    ) -> GrokResponse:
        """Perform NLI classification (XNLI style).
        
        Args:
            premise: The premise statement.
            hypothesis: The hypothesis to evaluate.
            language: Optional language code.
            
        Returns:
            GrokResponse with classification (entailment/neutral/contradiction).
        """
        messages = [
            {"role": "system", "content": "You are an NLI classifier. Given a premise and hypothesis, classify the relationship as exactly one of: 'entailment', 'neutral', or 'contradiction'. Output only the classification label."},
            {"role": "user", "content": f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\nClassification:"}
        ]
        
        return self._make_request(messages, temperature=0.1, max_tokens=20)
    
    def named_entity_recognition(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> GrokResponse:
        """Extract named entities from text (WikiANN style).
        
        Args:
            text: Text to extract entities from.
            entity_types: Optional list of entity types to extract.
                         Defaults to ['PER', 'LOC', 'ORG'].
            
        Returns:
            GrokResponse with extracted entities in "TYPE: entity" format.
        """
        types = entity_types or ["PER", "LOC", "ORG"]
        types_str = ", ".join(types)
        
        messages = [
            {"role": "system", "content": f"You are a named entity recognition system. Extract entities of types: {types_str}. Format output as 'TYPE: entity $$ TYPE: entity' with $$ as separator. If no entities found, output 'NONE'."},
            {"role": "user", "content": f"tag: {text}"}
        ]
        
        return self._make_request(messages, temperature=0.1)
    
    def paraphrase_detection(
        self,
        sentence1: str,
        sentence2: str,
        language: Optional[str] = None
    ) -> GrokResponse:
        """Detect if two sentences are paraphrases (PAWS-X style).
        
        Args:
            sentence1: First sentence.
            sentence2: Second sentence.
            language: Optional language code.
            
        Returns:
            GrokResponse with 'paraphrase' or 'not_paraphrase'.
        """
        messages = [
            {"role": "system", "content": "You are a paraphrase detection system. Determine if two sentences have the same meaning. Output exactly 'paraphrase' or 'not_paraphrase'."},
            {"role": "user", "content": f"Sentence 1: {sentence1}\n\nSentence 2: {sentence2}\n\nAre these paraphrases?"}
        ]
        
        return self._make_request(messages, temperature=0.1, max_tokens=20)
    
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        language: Optional[str] = None
    ) -> GrokResponse:
        """Summarize text (XSum style).
        
        Args:
            text: Text to summarize.
            max_length: Optional maximum summary length in words.
            language: Optional language code for output.
            
        Returns:
            GrokResponse with summary.
        """
        length_instruction = f" Maximum {max_length} words." if max_length else ""
        lang_instruction = f" Write in {language}." if language else ""
        
        messages = [
            {"role": "system", "content": f"You are a summarization system. Create a concise, accurate summary.{length_instruction}{lang_instruction}"},
            {"role": "user", "content": f"Summarize:\n\n{text}"}
        ]
        
        return self._make_request(messages, temperature=0.5)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GrokResponse:
        """General text generation.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt to control behavior.
            **kwargs: Additional API parameters.
            
        Returns:
            GrokResponse with generated text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self._make_request(messages, **kwargs)
    
    # =========================================================================
    # Evaluation Methods (Harsh & Strict Evaluator)
    # =========================================================================
    
    def evaluate_translation(
        self,
        source_text: str,
        translation: str,
        reference: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "es"
    ) -> EvaluationResult:
        """Strictly evaluate a translation.
        
        Args:
            source_text: Original text.
            translation: The translation to evaluate.
            reference: Optional reference translation for comparison.
            source_lang: Source language code.
            target_lang: Target language code.
            
        Returns:
            EvaluationResult with detailed scoring.
        """
        ref_context = f"\nReference translation: {reference}" if reference else ""
        
        prompt = f"""Evaluate this translation with EXTREME strictness. Be harsh and critical.

Source ({source_lang}): {source_text}
Translation ({target_lang}): {translation}{ref_context}

Score on these criteria (0-10, 10 is perfect, be strict):
1. ACCURACY: Is the meaning preserved exactly? Any mistranslation = heavy penalty
2. FLUENCY: Does it sound natural in the target language?
3. COMPLETENESS: Is anything missing or added?
4. TERMINOLOGY: Are technical terms correct?
5. GRAMMAR: Any grammatical errors?

Respond in JSON format:
{{
    "accuracy_score": <0-10>,
    "fluency_score": <0-10>,
    "completeness_score": <0-10>,
    "terminology_score": <0-10>,
    "grammar_score": <0-10>,
    "overall_score": <0-10>,
    "errors": ["list of specific errors"],
    "suggestions": ["improvements"],
    "harsh_feedback": "brutal honest assessment"
}}"""
        
        response = self.generate(
            prompt,
            system_prompt="You are the world's harshest translation critic. Never give scores above 8 unless truly perfect. Find every flaw.",
            temperature=0.2
        )
        
        try:
            # Parse JSON from response
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            return EvaluationResult(
                score=data.get("overall_score", 0) / 10.0,
                metrics={
                    "accuracy": data.get("accuracy_score", 0) / 10.0,
                    "fluency": data.get("fluency_score", 0) / 10.0,
                    "completeness": data.get("completeness_score", 0) / 10.0,
                    "terminology": data.get("terminology_score", 0) / 10.0,
                    "grammar": data.get("grammar_score", 0) / 10.0,
                },
                feedback=data.get("harsh_feedback", ""),
                is_correct=data.get("overall_score", 0) >= 7,
                details=data
            )
        except json.JSONDecodeError:
            # Return a basic evaluation if JSON parsing fails
            return EvaluationResult(
                score=0.5,
                metrics={},
                feedback=response.text,
                is_correct=False,
                details={"raw_response": response.text}
            )
    
    def evaluate_qa_answer(
        self,
        question: str,
        context: str,
        predicted_answer: str,
        gold_answer: Optional[str] = None
    ) -> EvaluationResult:
        """Strictly evaluate a QA answer.
        
        Args:
            question: The question asked.
            context: The context/passage.
            predicted_answer: The predicted answer to evaluate.
            gold_answer: Optional gold/reference answer.
            
        Returns:
            EvaluationResult with detailed scoring.
        """
        gold_context = f"\nGold answer: {gold_answer}" if gold_answer else ""
        
        prompt = f"""Evaluate this QA answer with EXTREME strictness.

Context: {context}

Question: {question}

Predicted Answer: {predicted_answer}{gold_context}

Evaluate harshly on:
1. CORRECTNESS: Is the answer factually correct based on context?
2. COMPLETENESS: Does it fully answer the question?
3. PRECISION: Is there unnecessary information?
4. EXTRACTIVENESS: Can this answer be found in the context?

Respond in JSON:
{{
    "correctness_score": <0-10>,
    "completeness_score": <0-10>,
    "precision_score": <0-10>,
    "extractiveness_score": <0-10>,
    "overall_score": <0-10>,
    "is_correct": <true/false>,
    "errors": ["specific issues"],
    "harsh_feedback": "brutal assessment"
}}"""
        
        response = self.generate(
            prompt,
            system_prompt="You are the world's strictest QA evaluator. Partial matches get low scores. Find every flaw.",
            temperature=0.2
        )
        
        try:
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            return EvaluationResult(
                score=data.get("overall_score", 0) / 10.0,
                metrics={
                    "correctness": data.get("correctness_score", 0) / 10.0,
                    "completeness": data.get("completeness_score", 0) / 10.0,
                    "precision": data.get("precision_score", 0) / 10.0,
                    "extractiveness": data.get("extractiveness_score", 0) / 10.0,
                },
                feedback=data.get("harsh_feedback", ""),
                is_correct=data.get("is_correct", False),
                details=data
            )
        except json.JSONDecodeError:
            return EvaluationResult(
                score=0.5,
                metrics={},
                feedback=response.text,
                is_correct=False,
                details={"raw_response": response.text}
            )
    
    def evaluate_ner(
        self,
        text: str,
        predicted_entities: str,
        gold_entities: Optional[str] = None
    ) -> EvaluationResult:
        """Strictly evaluate NER output.
        
        Args:
            text: Original text.
            predicted_entities: Predicted entities in "TYPE: entity $$ ..." format.
            gold_entities: Optional gold entities for comparison.
            
        Returns:
            EvaluationResult with detailed scoring.
        """
        gold_context = f"\nGold entities: {gold_entities}" if gold_entities else ""
        
        prompt = f"""Evaluate this NER extraction with EXTREME strictness.

Text: {text}

Predicted entities: {predicted_entities}{gold_context}

Evaluate harshly on:
1. PRECISION: Are all predicted entities actually entities? (False positives = severe penalty)
2. RECALL: Are all entities in the text captured? (Missed entities = severe penalty)
3. TYPE_ACCURACY: Are entity types (PER/LOC/ORG) correct?
4. BOUNDARY_ACCURACY: Are entity boundaries exact?

Respond in JSON:
{{
    "precision_score": <0-10>,
    "recall_score": <0-10>,
    "type_accuracy_score": <0-10>,
    "boundary_accuracy_score": <0-10>,
    "overall_score": <0-10>,
    "false_positives": ["list of incorrectly identified entities"],
    "false_negatives": ["list of missed entities"],
    "type_errors": ["list of type misclassifications"],
    "harsh_feedback": "brutal assessment"
}}"""
        
        response = self.generate(
            prompt,
            system_prompt="You are the world's strictest NER evaluator. Any error is a major flaw.",
            temperature=0.2
        )
        
        try:
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            return EvaluationResult(
                score=data.get("overall_score", 0) / 10.0,
                metrics={
                    "precision": data.get("precision_score", 0) / 10.0,
                    "recall": data.get("recall_score", 0) / 10.0,
                    "type_accuracy": data.get("type_accuracy_score", 0) / 10.0,
                    "boundary_accuracy": data.get("boundary_accuracy_score", 0) / 10.0,
                },
                feedback=data.get("harsh_feedback", ""),
                is_correct=data.get("overall_score", 0) >= 8,
                details=data
            )
        except json.JSONDecodeError:
            return EvaluationResult(
                score=0.5,
                metrics={},
                feedback=response.text,
                is_correct=False,
                details={"raw_response": response.text}
            )
    
    # =========================================================================
    # Batch Processing Methods
    # =========================================================================
    
    def batch_process(
        self,
        items: List[Dict[str, Any]],
        task_fn: Callable,
        max_workers: int = 5
    ) -> List[GrokResponse]:
        """Process multiple items in parallel.
        
        Args:
            items: List of dicts with parameters for each task.
            task_fn: The task function to call (e.g., self.translate).
            max_workers: Maximum parallel workers.
            
        Returns:
            List of GrokResponse objects.
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task_fn, **item) for item in items]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Batch item failed: {e}")
                    results.append(None)
        
        return results
    
    def evaluate_batch(
        self,
        predictions: List[Dict[str, Any]],
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Evaluate a batch of predictions with aggregate metrics.
        
        Args:
            predictions: List of prediction dicts.
            task_type: Type of task being evaluated.
            
        Returns:
            Dict with aggregate metrics and individual results.
        """
        results = []
        
        for pred in predictions:
            if task_type == TaskType.TRANSLATION:
                result = self.evaluate_translation(**pred)
            elif task_type == TaskType.QA:
                result = self.evaluate_qa_answer(**pred)
            elif task_type == TaskType.NER:
                result = self.evaluate_ner(**pred)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            results.append(result)
        
        # Aggregate metrics
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            correct_count = sum(1 for r in results if r.is_correct)
            accuracy = correct_count / len(results)
            
            # Aggregate per-metric
            all_metrics = {}
            for result in results:
                for key, value in result.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
            
            avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        else:
            avg_score = 0
            accuracy = 0
            avg_metrics = {}
        
        return {
            "average_score": avg_score,
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct_samples": correct_count if results else 0,
            "metrics": avg_metrics,
            "individual_results": results
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(api_key: Optional[str] = None, **kwargs) -> GrokClient:
    """Create a Grok client with optional configuration.
    
    Args:
        api_key: Grok API key. If None, uses GROK_API_KEY env var.
        **kwargs: Additional GrokConfig parameters.
        
    Returns:
        Configured GrokClient instance.
    """
    config_kwargs = {"api_key": api_key} if api_key else {}
    config_kwargs.update(kwargs)
    config = GrokConfig(**config_kwargs)
    return GrokClient(config)


def quick_translate(text: str, source: str, target: str, api_key: Optional[str] = None) -> str:
    """Quick translation helper.
    
    Args:
        text: Text to translate.
        source: Source language code.
        target: Target language code.
        api_key: Optional API key.
        
    Returns:
        Translated text string.
    """
    client = create_client(api_key)
    response = client.translate(text, source, target)
    return response.text


def quick_qa(question: str, context: str, api_key: Optional[str] = None) -> str:
    """Quick QA helper.
    
    Args:
        question: Question to answer.
        context: Context containing the answer.
        api_key: Optional API key.
        
    Returns:
        Answer string.
    """
    client = create_client(api_key)
    response = client.question_answering(question, context)
    return response.text


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grok API Client for mT5 Tasks")
    parser.add_argument("--task", choices=["translate", "qa", "nli", "ner", "summarize", "generate"],
                        required=True, help="Task to perform")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--source-lang", type=str, default="en", help="Source language")
    parser.add_argument("--target-lang", type=str, default="es", help="Target language")
    parser.add_argument("--context", type=str, help="Context for QA")
    parser.add_argument("--question", type=str, help="Question for QA")
    parser.add_argument("--premise", type=str, help="Premise for NLI")
    parser.add_argument("--hypothesis", type=str, help="Hypothesis for NLI")
    parser.add_argument("--model", type=str, default="grok-2", help="Model to use")
    
    args = parser.parse_args()
    
    # Create client
    client = create_client()
    
    if args.task == "translate":
        if not args.text:
            parser.error("--text required for translation")
        response = client.translate(args.text, args.source_lang, args.target_lang)
        print(f"Translation: {response.text}")
        
    elif args.task == "qa":
        if not args.question or not args.context:
            parser.error("--question and --context required for QA")
        response = client.question_answering(args.question, args.context)
        print(f"Answer: {response.text}")
        
    elif args.task == "nli":
        if not args.premise or not args.hypothesis:
            parser.error("--premise and --hypothesis required for NLI")
        response = client.natural_language_inference(args.premise, args.hypothesis)
        print(f"Classification: {response.text}")
        
    elif args.task == "ner":
        if not args.text:
            parser.error("--text required for NER")
        response = client.named_entity_recognition(args.text)
        print(f"Entities: {response.text}")
        
    elif args.task == "summarize":
        if not args.text:
            parser.error("--text required for summarization")
        response = client.summarize(args.text)
        print(f"Summary: {response.text}")
        
    elif args.task == "generate":
        if not args.text:
            parser.error("--text required for generation")
        response = client.generate(args.text)
        print(f"Generated: {response.text}")
    
    print(f"\n[Latency: {response.latency_ms:.0f}ms | Tokens: {response.usage}]")
