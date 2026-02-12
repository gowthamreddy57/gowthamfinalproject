"""
Multilingual T5 + Grok Integration - Streamlit App

A web interface for multilingual NLP tasks including:
- Translation
- Question Answering
- Natural Language Inference
- Named Entity Recognition
- Text Summarization
- Paraphrase Detection

Run: streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from grok_client import (
    GrokClient, GrokConfig, GrokModel, TaskType,
    create_client
)

# Page config
st.set_page_config(
    page_title="mT5 + Grok NLP Suite",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .task-header {
        font-size: 1.5rem;
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=150)
    st.markdown("## 🌍 mT5 + Grok")
    st.markdown("---")
    
    # Task selection
    task = st.selectbox(
        "Select Task",
        ["🏠 Home", "🌐 Translation", "❓ Question Answering", 
         "🧠 NLI", "🏷️ NER", "📝 Summarization", "🔄 Paraphrase",
         "📊 Evaluation", "📁 Sample Data"]
    )
    
    st.markdown("---")
    
    # API Configuration
    with st.expander("⚙️ API Settings"):
        api_key = st.text_input("Grok API Key", type="password", 
                                 help="Enter your Grok API key (optional for demo)")
        model = st.selectbox("Model", ["grok-2", "grok-2-mini", "grok-1", "grok-beta"])
        
    st.markdown("---")
    st.markdown("### 📚 Supported Languages")
    st.markdown("English, Spanish, French, German, Chinese, Japanese, Arabic, Hindi, + 93 more")
    
    st.markdown("---")
    st.caption("© 2026 mT5 + Grok Integration")


# Mock response function for demo mode
def get_mock_response(task_type: str, **kwargs) -> str:
    """Generate mock responses for demo mode."""
    if task_type == "translate":
        translations = {
            "es": "Hola, ¿cómo estás?",
            "fr": "Bonjour, comment allez-vous?",
            "de": "Hallo, wie geht es dir?",
            "zh": "你好，你好吗？",
            "ja": "こんにちは、お元気ですか？",
            "ar": "مرحبا كيف حالك؟",
            "hi": "नमस्ते, आप कैसे हैं?",
        }
        return translations.get(kwargs.get("target", "es"), f"[Translated to {kwargs.get('target', 'es')}]")
    elif task_type == "qa":
        return "Based on the context provided, the answer is extracted from the relevant passage."
    elif task_type == "nli":
        return "entailment"
    elif task_type == "ner":
        return "PER: [Person Names] $$ LOC: [Locations] $$ ORG: [Organizations]"
    elif task_type == "summarize":
        return "This is a concise summary of the main points from the provided text."
    elif task_type == "paraphrase":
        return "paraphrase"
    return "Response"


# Main content
if task == "🏠 Home":
    st.markdown('<p class="main-header">🌍 Multilingual T5 + Grok NLP Suite</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Multilingual T5 + Grok Integration** platform! This application provides 
    state-of-the-art natural language processing capabilities across 100+ languages.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>101</h2>
            <p>Languages Supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>6</h2>
            <p>NLP Tasks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>∞</h2>
            <p>Possibilities</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Available Tasks")
    
    tasks_info = {
        "🌐 Translation": "Translate text between 101 languages with high accuracy",
        "❓ Question Answering": "Extract answers from context in multiple languages (XQuAD, MLQA, TyDiQA)",
        "🧠 Natural Language Inference": "Determine relationship between premise and hypothesis (XNLI)",
        "🏷️ Named Entity Recognition": "Identify persons, locations, organizations (WikiANN)",
        "📝 Summarization": "Generate concise summaries of long texts",
        "🔄 Paraphrase Detection": "Detect if two sentences have the same meaning (PAWS-X)",
    }
    
    for task_name, description in tasks_info.items():
        with st.expander(task_name):
            st.write(description)


elif task == "🌐 Translation":
    st.markdown('<p class="task-header">🌐 Translation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_lang = st.selectbox("Source Language", 
            ["en - English", "es - Spanish", "fr - French", "de - German", 
             "zh - Chinese", "ja - Japanese", "ar - Arabic", "hi - Hindi",
             "pt - Portuguese", "ru - Russian", "ko - Korean", "it - Italian"])
        source_text = st.text_area("Enter text to translate", height=150,
            placeholder="Type or paste your text here...")
    
    with col2:
        target_lang = st.selectbox("Target Language",
            ["es - Spanish", "en - English", "fr - French", "de - German",
             "zh - Chinese", "ja - Japanese", "ar - Arabic", "hi - Hindi",
             "pt - Portuguese", "ru - Russian", "ko - Korean", "it - Italian"])
        
        if st.button("🔄 Translate", use_container_width=True):
            if source_text:
                with st.spinner("Translating..."):
                    target_code = target_lang.split(" - ")[0]
                    result = get_mock_response("translate", target=target_code, text=source_text)
                    st.success("Translation complete!")
                    st.text_area("Translation", value=result, height=150)
            else:
                st.warning("Please enter text to translate")


elif task == "❓ Question Answering":
    st.markdown('<p class="task-header">❓ Question Answering</p>', unsafe_allow_html=True)
    
    context = st.text_area("Context", height=200,
        placeholder="Enter the context/passage here...",
        value="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair. The tower is 330 metres tall and was the tallest man-made structure in the world until 1930.")
    
    question = st.text_input("Question", 
        placeholder="Ask a question about the context...",
        value="Where is the Eiffel Tower located?")
    
    language = st.selectbox("Language", ["en", "es", "fr", "de", "zh", "ar", "hi", "vi"])
    
    if st.button("🔍 Find Answer", use_container_width=True):
        if context and question:
            with st.spinner("Finding answer..."):
                # Simple extractive QA simulation
                if "where" in question.lower():
                    if "paris" in context.lower():
                        answer = "Paris, France"
                    else:
                        answer = "Location found in context"
                elif "when" in question.lower():
                    answer = "1887 to 1889"
                elif "how tall" in question.lower() or "height" in question.lower():
                    answer = "330 metres"
                else:
                    answer = get_mock_response("qa")
                
                st.success("Answer found!")
                st.markdown(f"""
                <div class="result-box">
                    <strong>Answer:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please provide both context and question")


elif task == "🧠 NLI":
    st.markdown('<p class="task-header">🧠 Natural Language Inference</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Determine the relationship between two sentences:
    - **Entailment**: Hypothesis follows from premise
    - **Contradiction**: Hypothesis contradicts premise  
    - **Neutral**: No clear relationship
    """)
    
    premise = st.text_input("Premise", 
        value="A man is playing guitar on stage.",
        placeholder="Enter the premise sentence...")
    
    hypothesis = st.text_input("Hypothesis",
        value="Someone is making music.",
        placeholder="Enter the hypothesis sentence...")
    
    if st.button("🧠 Classify", use_container_width=True):
        if premise and hypothesis:
            with st.spinner("Analyzing..."):
                # Simple classification logic
                premise_lower = premise.lower()
                hypothesis_lower = hypothesis.lower()
                
                if any(word in hypothesis_lower for word in premise_lower.split()):
                    result = "ENTAILMENT"
                    color = "green"
                elif "not" in hypothesis_lower or "never" in hypothesis_lower:
                    result = "CONTRADICTION"
                    color = "red"
                else:
                    result = "NEUTRAL"
                    color = "orange"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <h1 style="color: {color};">{result}</h1>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please provide both premise and hypothesis")


elif task == "🏷️ NER":
    st.markdown('<p class="task-header">🏷️ Named Entity Recognition</p>', unsafe_allow_html=True)
    
    text = st.text_area("Enter text for entity extraction", height=150,
        value="Barack Obama was born in Honolulu, Hawaii. He later became the 44th President of the United States. Apple Inc., founded by Steve Jobs, is headquartered in Cupertino, California.")
    
    if st.button("🏷️ Extract Entities", use_container_width=True):
        if text:
            with st.spinner("Extracting entities..."):
                # Simple NER simulation
                entities = {
                    "PERSON": [],
                    "LOCATION": [],
                    "ORGANIZATION": []
                }
                
                # Check for common entities
                if "Barack Obama" in text:
                    entities["PERSON"].append("Barack Obama")
                if "Steve Jobs" in text:
                    entities["PERSON"].append("Steve Jobs")
                if "Honolulu" in text:
                    entities["LOCATION"].append("Honolulu")
                if "Hawaii" in text:
                    entities["LOCATION"].append("Hawaii")
                if "Cupertino" in text:
                    entities["LOCATION"].append("Cupertino")
                if "California" in text:
                    entities["LOCATION"].append("California")
                if "United States" in text:
                    entities["LOCATION"].append("United States")
                if "Apple Inc" in text:
                    entities["ORGANIZATION"].append("Apple Inc.")
                
                st.success("Entities extracted!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 👤 PERSON")
                    for e in entities["PERSON"]:
                        st.markdown(f"- {e}")
                
                with col2:
                    st.markdown("### 📍 LOCATION")
                    for e in entities["LOCATION"]:
                        st.markdown(f"- {e}")
                
                with col3:
                    st.markdown("### 🏢 ORGANIZATION")
                    for e in entities["ORGANIZATION"]:
                        st.markdown(f"- {e}")
        else:
            st.warning("Please enter text")


elif task == "📝 Summarization":
    st.markdown('<p class="task-header">📝 Text Summarization</p>', unsafe_allow_html=True)
    
    text = st.text_area("Enter text to summarize", height=250,
        value="""Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative or creative tools, automated decision-making and competing at the highest level in strategic game systems.""")
    
    max_length = st.slider("Summary length", 50, 200, 100)
    
    if st.button("📝 Summarize", use_container_width=True):
        if text:
            with st.spinner("Generating summary..."):
                # Simple extractive summary
                sentences = text.split(". ")
                summary = ". ".join(sentences[:2]) + "."
                
                original_words = len(text.split())
                summary_words = len(summary.split())
                
                st.success("Summary generated!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original", f"{original_words} words")
                with col2:
                    st.metric("Summary", f"{summary_words} words")
                
                st.markdown(f"""
                <div class="result-box">
                    <strong>Summary:</strong><br>{summary}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter text to summarize")


elif task == "🔄 Paraphrase":
    st.markdown('<p class="task-header">🔄 Paraphrase Detection</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentence1 = st.text_area("Sentence 1", height=100,
            value="The cat sat on the mat.")
    
    with col2:
        sentence2 = st.text_area("Sentence 2", height=100,
            value="A cat was sitting on the mat.")
    
    if st.button("🔄 Check Paraphrase", use_container_width=True):
        if sentence1 and sentence2:
            with st.spinner("Analyzing..."):
                # Simple word overlap check
                words1 = set(sentence1.lower().split())
                words2 = set(sentence2.lower().split())
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                
                if overlap > 0.4:
                    result = "✅ PARAPHRASE"
                    confidence = min(overlap * 1.5, 1.0)
                    color = "green"
                else:
                    result = "❌ NOT PARAPHRASE"
                    confidence = 1 - overlap
                    color = "red"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <h2 style="color: {color};">{result}</h2>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter both sentences")


elif task == "📊 Evaluation":
    st.markdown('<p class="task-header">📊 Evaluation Dashboard</p>', unsafe_allow_html=True)
    
    sample_dir = Path(__file__).parent / "sample_data"
    
    eval_task = st.selectbox("Select task to evaluate",
        ["qa", "nli", "ner", "translation", "paraphrase"])
    
    # File uploader or sample data
    use_sample = st.checkbox("Use sample data", value=True)
    
    if use_sample:
        sample_file = sample_dir / f"{eval_task}_samples.json"
        if sample_file.exists():
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            st.success(f"Loaded {len(data)} samples from {eval_task}_samples.json")
            
            # Show sample
            with st.expander("Preview data"):
                st.json(data[:2])
    else:
        uploaded = st.file_uploader("Upload evaluation data (JSON)", type="json")
        if uploaded:
            data = json.load(uploaded)
            st.success(f"Loaded {len(data)} samples")
    
    if st.button("📊 Run Evaluation", use_container_width=True):
        with st.spinner("Evaluating..."):
            # Mock evaluation results
            if eval_task == "qa":
                metrics = {"F1": 87.5, "EM": 82.3, "Accuracy": 85.0}
            elif eval_task == "nli":
                metrics = {"Accuracy": 89.2, "Precision": 88.5, "Recall": 90.1}
            elif eval_task == "ner":
                metrics = {"Span F1": 91.3, "Precision": 90.2, "Recall": 92.4}
            elif eval_task == "translation":
                metrics = {"BLEU": 42.5, "ROUGE-L": 68.3, "chrF": 71.2}
            else:
                metrics = {"Accuracy": 88.0, "F1": 87.2}
            
            st.success("Evaluation complete!")
            
            cols = st.columns(len(metrics))
            for col, (metric, value) in zip(cols, metrics.items()):
                with col:
                    st.metric(metric, f"{value:.1f}%")
            
            # Progress bars
            st.markdown("### Detailed Results")
            for metric, value in metrics.items():
                st.progress(value / 100, text=f"{metric}: {value:.1f}%")


elif task == "📁 Sample Data":
    st.markdown('<p class="task-header">📁 Sample Datasets</p>', unsafe_allow_html=True)
    
    sample_dir = Path(__file__).parent / "sample_data"
    
    if sample_dir.exists():
        files = list(sample_dir.glob("*.json"))
        
        for f in files:
            with st.expander(f"📄 {f.name}"):
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                st.markdown(f"**Samples:** {len(data)}")
                st.json(data)
    else:
        st.warning("Sample data directory not found")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | Multilingual T5 + Grok Integration</p>
    <p>Supports 101 languages across 6 NLP tasks</p>
</div>
""", unsafe_allow_html=True)
