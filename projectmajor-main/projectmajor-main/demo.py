#!/usr/bin/env python3
"""
Demo script for Multilingual T5 + Grok Integration

This script demonstrates the functionality of the mT5 Grok integration
without requiring an API key (uses mock mode).

Run: python demo.py
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grok_client import (
    GrokClient, GrokConfig, GrokModel, TaskType,
    EvaluationResult, create_client
)

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, value: str):
    """Print a formatted result."""
    print(f"  {label}: {value}")


def demo_translation():
    """Demonstrate translation capabilities."""
    print_header("🌍 TRANSLATION DEMO")
    
    examples = [
        ("Hello, how are you?", "en", "es"),
        ("The weather is beautiful today.", "en", "fr"),
        ("Bonjour le monde!", "fr", "en"),
    ]
    
    for text, src, tgt in examples:
        print(f"\n  [{src}→{tgt}] \"{text}\"")
        # In mock mode, we simulate the translation
        translations = {
            ("Hello, how are you?", "es"): "Hola, ¿cómo estás?",
            ("The weather is beautiful today.", "fr"): "Le temps est magnifique aujourd'hui.",
            ("Bonjour le monde!", "en"): "Hello world!",
        }
        result = translations.get((text, tgt), f"[Translation to {tgt}]")
        print(f"  → \"{result}\"")


def demo_question_answering():
    """Demonstrate QA capabilities."""
    print_header("❓ QUESTION ANSWERING DEMO")
    
    qa_pairs = [
        {
            "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889.",
            "question": "Where is the Eiffel Tower located?",
            "answer": "Paris, France"
        },
        {
            "context": "Python is a programming language created by Guido van Rossum and first released in 1991.",
            "question": "Who created Python?",
            "answer": "Guido van Rossum"
        },
    ]
    
    for qa in qa_pairs:
        print(f"\n  Context: \"{qa['context'][:60]}...\"")
        print(f"  Question: \"{qa['question']}\"")
        print(f"  → Answer: \"{qa['answer']}\"")


def demo_nli():
    """Demonstrate Natural Language Inference."""
    print_header("🧠 NATURAL LANGUAGE INFERENCE DEMO")
    
    examples = [
        ("A man is playing guitar.", "Someone is making music.", "entailment"),
        ("The cat is sleeping.", "The cat is running.", "contradiction"),
        ("A woman is reading.", "The woman is at the library.", "neutral"),
    ]
    
    for premise, hypothesis, label in examples:
        print(f"\n  Premise: \"{premise}\"")
        print(f"  Hypothesis: \"{hypothesis}\"")
        print(f"  → Classification: {label.upper()}")


def demo_ner():
    """Demonstrate Named Entity Recognition."""
    print_header("🏷️ NAMED ENTITY RECOGNITION DEMO")
    
    examples = [
        ("Barack Obama was born in Honolulu, Hawaii.", 
         [("Barack Obama", "PERSON"), ("Honolulu", "LOCATION"), ("Hawaii", "LOCATION")]),
        ("Apple Inc. was founded by Steve Jobs.", 
         [("Apple Inc.", "ORGANIZATION"), ("Steve Jobs", "PERSON")]),
    ]
    
    for text, entities in examples:
        print(f"\n  Text: \"{text}\"")
        print("  → Entities:")
        for entity, etype in entities:
            print(f"      • {entity} [{etype}]")


def demo_summarization():
    """Demonstrate text summarization."""
    print_header("📝 SUMMARIZATION DEMO")
    
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial 
    intelligence" had previously been used to describe machines that mimic and 
    display "human" cognitive skills that are associated with the human mind, 
    such as "learning" and "problem-solving".
    """
    
    summary = "AI is machine intelligence that studies intelligent agents perceiving environments and taking goal-oriented actions."
    
    print(f"\n  Original ({len(text.split())} words):")
    print(f"  \"{text.strip()[:100]}...\"")
    print(f"\n  → Summary ({len(summary.split())} words):")
    print(f"  \"{summary}\"")


def demo_metrics():
    """Demonstrate evaluation metrics."""
    print_header("📊 EVALUATION METRICS DEMO")
    
    # Load sample data
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    
    if os.path.exists(sample_dir):
        print("\n  Available sample datasets:")
        for f in os.listdir(sample_dir):
            if f.endswith('.json'):
                filepath = os.path.join(sample_dir, f)
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                print(f"    • {f}: {len(data)} samples")
    
    # Simple accuracy calculation
    predictions = ["Paris", "Shakespeare", "1945", "Everest"]
    references = ["Paris", "William Shakespeare", "1945", "Mount Everest"]
    
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.lower() in r.lower())
    accuracy = exact_matches / len(predictions) * 100
    
    print(f"\n  Sample QA Evaluation:")
    print(f"    Predictions: {predictions}")
    print(f"    References:  {references}")
    print(f"    → Accuracy: {accuracy:.1f}%")


def demo_multilingual():
    """Demonstrate multilingual capabilities."""
    print_header("🌐 MULTILINGUAL CAPABILITIES")
    
    languages = {
        "English": "Hello, world!",
        "Spanish": "¡Hola, mundo!",
        "French": "Bonjour le monde!",
        "German": "Hallo Welt!",
        "Chinese": "你好，世界！",
        "Japanese": "こんにちは世界！",
        "Arabic": "مرحبا بالعالم!",
        "Hindi": "नमस्ते दुनिया!",
    }
    
    print("\n  Supported languages for mT5 tasks:")
    for lang, greeting in languages.items():
        print(f"    • {lang:12s}: {greeting}")
    
    print(f"\n  Total: 101 languages supported by mT5")


def main():
    """Run all demos."""
    print("\n" + "🚀" * 30)
    print("  MULTILINGUAL T5 + GROK INTEGRATION DEMO")
    print("🚀" * 30)
    
    demo_translation()
    demo_question_answering()
    demo_nli()
    demo_ner()
    demo_summarization()
    demo_metrics()
    demo_multilingual()
    
    print_header("✅ DEMO COMPLETE")
    print("""
  This demo showed the capabilities of the mT5 + Grok integration.
  
  To use with the real Grok API:
    1. Set your API key: export GROK_API_KEY=your_key
    2. Run commands like:
       python grok_client.py --task translate --text "Hello" --target-lang es
       python grok_client.py --task qa --question "What is AI?" --context "..."
    
  For evaluation:
    python grok_evaluator.py --task qa --data sample_data/qa_samples.json
    """)


if __name__ == "__main__":
    main()
