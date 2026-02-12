"""
Multilingual T5 + Grok Chat Interface

An interactive chat interface for multilingual NLP tasks.

Run: streamlit run chat_app.py
"""

import streamlit as st
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="mT5 + Grok Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: #2d2d44;
        color: #e0e0e0;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #444;
    }
    
    .message-time {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.3rem;
    }
    
    .quick-action {
        background: #2d2d44;
        border: 1px solid #667eea;
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .quick-action:hover {
        background: #667eea;
        color: white;
    }
    
    .typing-indicator {
        display: flex;
        gap: 5px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    .sidebar-stats {
        background: #2d2d44;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .task-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .badge-translate { background: #4CAF50; color: white; }
    .badge-qa { background: #2196F3; color: white; }
    .badge-nli { background: #9C27B0; color: white; }
    .badge-ner { background: #FF9800; color: white; }
    .badge-summarize { background: #E91E63; color: white; }
    .badge-chat { background: #00BCD4; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "task_counts" not in st.session_state:
    st.session_state.task_counts = {
        "translate": 0, "qa": 0, "nli": 0, 
        "ner": 0, "summarize": 0, "chat": 0
    }


def detect_task(message: str) -> tuple:
    """Detect the NLP task from user message."""
    message_lower = message.lower()
    
    # Translation patterns
    if any(p in message_lower for p in ["translate", "translation", "convert to", "in spanish", "in french", "in german", "in chinese", "in japanese", "to english", "to spanish", "to french"]):
        return "translate", extract_translation_params(message)
    
    # QA patterns
    if any(p in message_lower for p in ["what is", "who is", "where is", "when did", "how many", "why did", "answer this", "find the answer"]):
        if "context:" in message_lower or "passage:" in message_lower:
            return "qa", extract_qa_params(message)
    
    # NLI patterns
    if any(p in message_lower for p in ["entailment", "contradiction", "premise:", "hypothesis:", "does this follow", "is this true"]):
        return "nli", extract_nli_params(message)
    
    # NER patterns
    if any(p in message_lower for p in ["entities", "extract names", "find people", "find locations", "named entity", "ner"]):
        return "ner", {"text": message}
    
    # Summarization patterns
    if any(p in message_lower for p in ["summarize", "summary", "shorten", "brief", "tldr", "main points"]):
        return "summarize", {"text": message}
    
    # Default to chat
    return "chat", {"message": message}


def extract_translation_params(message: str) -> dict:
    """Extract translation parameters from message."""
    # Detect target language
    lang_map = {
        "spanish": "es", "french": "fr", "german": "de",
        "chinese": "zh", "japanese": "ja", "korean": "ko",
        "arabic": "ar", "hindi": "hi", "portuguese": "pt",
        "russian": "ru", "italian": "it", "english": "en"
    }
    
    target = "es"  # default
    for lang, code in lang_map.items():
        if lang in message.lower():
            target = code
            break
    
    # Extract text to translate (simple heuristic)
    text = message
    for phrase in ["translate", "to spanish", "to french", "to german", "in spanish", "in french", "in german"]:
        text = text.lower().replace(phrase, "")
    
    # Look for quoted text
    quoted = re.findall(r'"([^"]*)"', message)
    if quoted:
        text = quoted[0]
    
    return {"text": text.strip(), "target": target}


def extract_qa_params(message: str) -> dict:
    """Extract QA parameters from message."""
    context = ""
    question = ""
    
    if "context:" in message.lower():
        parts = message.lower().split("context:")
        if len(parts) > 1:
            context_part = parts[1]
            if "question:" in context_part:
                context = context_part.split("question:")[0].strip()
                question = context_part.split("question:")[1].strip()
            else:
                context = context_part.strip()
    
    if not question:
        question = message
    
    return {"context": context, "question": question}


def extract_nli_params(message: str) -> dict:
    """Extract NLI parameters from message."""
    premise = ""
    hypothesis = ""
    
    if "premise:" in message.lower():
        parts = message.lower().split("premise:")
        if len(parts) > 1:
            premise_part = parts[1]
            if "hypothesis:" in premise_part:
                premise = premise_part.split("hypothesis:")[0].strip()
                hypothesis = premise_part.split("hypothesis:")[1].strip()
    
    return {"premise": premise, "hypothesis": hypothesis}


def process_translation(params: dict) -> str:
    """Process translation request."""
    translations = {
        ("hello", "es"): "Hola",
        ("hello", "fr"): "Bonjour",
        ("hello", "de"): "Hallo",
        ("how are you", "es"): "¿Cómo estás?",
        ("good morning", "es"): "Buenos días",
        ("thank you", "es"): "Gracias",
        ("thank you", "fr"): "Merci",
    }
    
    text = params.get("text", "").lower().strip()
    target = params.get("target", "es")
    
    # Check for exact matches
    result = translations.get((text, target))
    if result:
        return f"🌐 **Translation ({target.upper()}):**\n\n> {result}"
    
    # Generate mock translation
    lang_names = {"es": "Spanish", "fr": "French", "de": "German", 
                  "zh": "Chinese", "ja": "Japanese", "ar": "Arabic"}
    lang_name = lang_names.get(target, target.upper())
    
    return f"🌐 **Translation to {lang_name}:**\n\n> [{text} → translated to {lang_name}]\n\n*Note: For accurate translations, configure your Grok API key.*"


def process_qa(params: dict) -> str:
    """Process QA request."""
    context = params.get("context", "")
    question = params.get("question", "")
    
    # Simple extractive QA simulation
    if not context:
        return "❓ Please provide a context/passage for me to find the answer. Format:\n\n`Context: [your text] Question: [your question]`"
    
    # Look for common answer patterns
    answer = "Based on the provided context, I couldn't find a specific answer."
    
    if "where" in question.lower():
        # Look for location words
        locations = re.findall(r'in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', context)
        if locations:
            answer = locations[0]
    elif "who" in question.lower():
        # Look for names
        names = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', context)
        if names:
            answer = names[0]
    elif "when" in question.lower():
        # Look for dates/years
        dates = re.findall(r'\b(19\d{2}|20\d{2})\b', context)
        if dates:
            answer = dates[0]
    
    return f"❓ **Question:** {question}\n\n📖 **Context:** {context[:100]}...\n\n✅ **Answer:** {answer}"


def process_nli(params: dict) -> str:
    """Process NLI request."""
    premise = params.get("premise", "")
    hypothesis = params.get("hypothesis", "")
    
    if not premise or not hypothesis:
        return "🧠 Please provide both premise and hypothesis. Format:\n\n`Premise: [sentence 1] Hypothesis: [sentence 2]`"
    
    # Simple classification
    premise_words = set(premise.lower().split())
    hypothesis_words = set(hypothesis.lower().split())
    
    overlap = len(premise_words & hypothesis_words) / max(len(hypothesis_words), 1)
    
    if overlap > 0.5:
        label = "ENTAILMENT ✅"
        explanation = "The hypothesis follows from the premise."
    elif any(neg in hypothesis.lower() for neg in ["not", "never", "no"]):
        label = "CONTRADICTION ❌"
        explanation = "The hypothesis contradicts the premise."
    else:
        label = "NEUTRAL ⚖️"
        explanation = "The hypothesis neither follows from nor contradicts the premise."
    
    return f"🧠 **Natural Language Inference**\n\n**Premise:** {premise}\n**Hypothesis:** {hypothesis}\n\n**Result:** {label}\n\n_{explanation}_"


def process_ner(params: dict) -> str:
    """Process NER request."""
    text = params.get("text", "")
    
    entities = {"PERSON": [], "LOCATION": [], "ORGANIZATION": []}
    
    # Simple pattern matching for entities
    # People (capitalized names)
    people = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
    entities["PERSON"] = list(set(people))
    
    # Locations (common patterns)
    locations = re.findall(r'(?:in|at|from) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
    entities["LOCATION"] = list(set(locations))
    
    # Organizations (Inc, Corp, etc.)
    orgs = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:Inc|Corp|Ltd|Company|Organization)\.?)', text)
    entities["ORGANIZATION"] = list(set(orgs))
    
    result = "🏷️ **Named Entity Recognition**\n\n"
    
    if entities["PERSON"]:
        result += f"👤 **People:** {', '.join(entities['PERSON'])}\n"
    if entities["LOCATION"]:
        result += f"📍 **Locations:** {', '.join(entities['LOCATION'])}\n"
    if entities["ORGANIZATION"]:
        result += f"🏢 **Organizations:** {', '.join(entities['ORGANIZATION'])}\n"
    
    if not any(entities.values()):
        result += "_No entities detected. Try including proper nouns (capitalized names)._"
    
    return result


def process_summarization(params: dict) -> str:
    """Process summarization request."""
    text = params.get("text", "")
    
    # Remove the command words
    for word in ["summarize", "summary", "shorten", "tldr", "main points of"]:
        text = text.lower().replace(word, "")
    text = text.strip()
    
    if len(text) < 50:
        return "📝 Please provide a longer text to summarize (at least a paragraph)."
    
    # Simple extractive summary - take first 2 sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 2:
        summary = ". ".join(sentences[:2]) + "."
    else:
        summary = text
    
    original_words = len(text.split())
    summary_words = len(summary.split())
    
    return f"📝 **Summary**\n\n> {summary}\n\n_Compressed from {original_words} to {summary_words} words ({(1-summary_words/original_words)*100:.0f}% reduction)_"


def process_chat(params: dict) -> str:
    """Process general chat."""
    message = params.get("message", "").lower()
    
    # Greetings
    if any(g in message for g in ["hello", "hi", "hey", "greetings"]):
        return "👋 Hello! I'm your multilingual NLP assistant. I can help you with:\n\n• 🌐 **Translation** - Translate text to 100+ languages\n• ❓ **Question Answering** - Find answers in text\n• 🧠 **NLI** - Analyze text relationships\n• 🏷️ **NER** - Extract named entities\n• 📝 **Summarization** - Shorten long texts\n\nJust type your request!"
    
    if any(h in message for h in ["help", "what can you do", "commands"]):
        return """🤖 **Available Commands:**

**Translation:**
> "Translate 'hello' to Spanish"
> "How do you say 'thank you' in French?"

**Question Answering:**
> "Context: [text] Question: [question]"

**NLI (Natural Language Inference):**
> "Premise: [sentence] Hypothesis: [sentence]"

**Named Entity Recognition:**
> "Find entities in: Barack Obama visited Paris"

**Summarization:**
> "Summarize: [long text]"

**Tips:**
- Use quotes for specific text
- Specify target language for translations
- Provide context for QA tasks"""
    
    if "thank" in message:
        return "You're welcome! 😊 Let me know if you need anything else."
    
    if any(w in message for w in ["bye", "goodbye", "exit"]):
        return "👋 Goodbye! Have a great day!"
    
    # Default response
    return "🤔 I'm not sure how to help with that. Try asking me to:\n\n• Translate text\n• Answer a question with context\n• Analyze premise/hypothesis\n• Find named entities\n• Summarize text\n\nOr type **help** for examples!"


def get_response(message: str) -> tuple:
    """Get AI response based on detected task."""
    task, params = detect_task(message)
    
    if task == "translate":
        response = process_translation(params)
    elif task == "qa":
        response = process_qa(params)
    elif task == "nli":
        response = process_nli(params)
    elif task == "ner":
        response = process_ner(params)
    elif task == "summarize":
        response = process_summarization(params)
    else:
        response = process_chat(params)
    
    # Update task count
    st.session_state.task_counts[task] = st.session_state.task_counts.get(task, 0) + 1
    
    return task, response


# Sidebar
with st.sidebar:
    st.markdown("## 💬 mT5 Chat")
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Session Stats")
    total = sum(st.session_state.task_counts.values())
    st.markdown(f"**Total messages:** {total}")
    
    if total > 0:
        st.markdown("**Tasks used:**")
        for task, count in st.session_state.task_counts.items():
            if count > 0:
                st.markdown(f"• {task}: {count}")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📥 Export Chat", use_container_width=True):
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "Download JSON",
            chat_export,
            "chat_history.json",
            "application/json"
        )
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings"):
        api_key = st.text_input("Grok API Key", type="password")
        model = st.selectbox("Model", ["grok-2", "grok-2-mini"])
        st.checkbox("Show task badges", value=True, key="show_badges")


# Main chat interface
st.markdown('<p class="main-header">💬 mT5 + Grok Chat</p>', unsafe_allow_html=True)

# Quick action buttons
st.markdown("#### ⚡ Quick Examples")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🌐 Translate", use_container_width=True):
        st.session_state.quick_input = "Translate 'Hello, how are you?' to Spanish"

with col2:
    if st.button("❓ QA", use_container_width=True):
        st.session_state.quick_input = "Context: The Eiffel Tower is in Paris, France. Question: Where is the Eiffel Tower?"

with col3:
    if st.button("🧠 NLI", use_container_width=True):
        st.session_state.quick_input = "Premise: A dog is running. Hypothesis: An animal is moving."

with col4:
    if st.button("🏷️ NER", use_container_width=True):
        st.session_state.quick_input = "Find entities in: Steve Jobs founded Apple Inc. in Cupertino."

with col5:
    if st.button("📝 Summary", use_container_width=True):
        st.session_state.quick_input = "Summarize: Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI applications are becoming increasingly prevalent. Machine learning algorithms can now diagnose diseases, predict market trends, and automate complex tasks that previously required human expertise."

st.markdown("---")

# Chat messages container
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                <div class="user-message">
                    {msg["content"]}
                    <div class="message-time">{msg.get("time", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            badge = ""
            if st.session_state.get("show_badges", True) and "task" in msg:
                badge_class = f"badge-{msg['task']}"
                badge = f'<span class="task-badge {badge_class}">{msg["task"]}</span>'
            
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
                <div class="assistant-message">
                    {badge}
                    <div style="margin-top: 0.5rem;">{msg["content"]}</div>
                    <div class="message-time">{msg.get("time", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown("---")

# Check for quick input
default_input = st.session_state.pop("quick_input", "")

user_input = st.chat_input("Type your message... (try: translate, summarize, find entities, etc.)")

if user_input or default_input:
    message = user_input or default_input
    current_time = datetime.now().strftime("%H:%M")
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": message,
        "time": current_time
    })
    
    # Get response
    task, response = get_response(message)
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "task": task,
        "time": current_time
    })
    
    st.rerun()

# Welcome message if no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #888;">
        <h2>👋 Welcome to mT5 + Grok Chat!</h2>
        <p>I'm your multilingual NLP assistant. Try one of the quick actions above or type a message below.</p>
        <p style="font-size: 0.9rem; margin-top: 2rem;">
            <strong>Examples:</strong><br>
            "Translate 'hello' to French"<br>
            "Summarize: [your text]"<br>
            "Find entities in: Apple was founded by Steve Jobs"
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; font-size: 0.8rem;">
    mT5 + Grok Chat Interface | Supports 101 languages | Built with Streamlit
</div>
""", unsafe_allow_html=True)
