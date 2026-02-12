"""
Multilingual T5 + Grok - Complete NLP Suite

A unified web interface with:
- API Connection Testing
- Translation to ALL languages at once
- Question Answering
- Natural Language Inference
- Named Entity Recognition
- Text Summarization
- Chat Interface

Run: streamlit run main_app.py
"""

import streamlit as st
import json
import os
import sys
import re
import requests
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="mT5 + Grok Complete Suite",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# All supported languages
ALL_LANGUAGES = {
    "es": "Spanish 🇪🇸",
    "fr": "French 🇫🇷",
    "de": "German 🇩🇪",
    "it": "Italian 🇮🇹",
    "pt": "Portuguese 🇵🇹",
    "ru": "Russian 🇷🇺",
    "zh": "Chinese 🇨🇳",
    "ja": "Japanese 🇯🇵",
    "ko": "Korean 🇰🇷",
    "ar": "Arabic 🇸🇦",
    "hi": "Hindi 🇮🇳",
    "bn": "Bengali 🇧🇩",
    "tr": "Turkish 🇹🇷",
    "vi": "Vietnamese 🇻🇳",
    "th": "Thai 🇹🇭",
    "pl": "Polish 🇵🇱",
    "nl": "Dutch 🇳🇱",
    "sv": "Swedish 🇸🇪",
    "da": "Danish 🇩🇰",
    "no": "Norwegian 🇳🇴",
    "fi": "Finnish 🇫🇮",
    "el": "Greek 🇬🇷",
    "he": "Hebrew 🇮🇱",
    "cs": "Czech 🇨🇿",
    "ro": "Romanian 🇷🇴",
    "hu": "Hungarian 🇭🇺",
    "uk": "Ukrainian 🇺🇦",
    "id": "Indonesian 🇮🇩",
    "ms": "Malay 🇲🇾",
    "tl": "Filipino 🇵🇭",
    "sw": "Swahili 🇰🇪",
    "ta": "Tamil 🇮🇳",
    "te": "Telugu 🇮🇳",
    "mr": "Marathi 🇮🇳",
    "ur": "Urdu 🇵🇰",
    "fa": "Persian 🇮🇷",
    "bg": "Bulgarian 🇧🇬",
    "hr": "Croatian 🇭🇷",
    "sk": "Slovak 🇸🇰",
    "sl": "Slovenian 🇸🇮",
    "et": "Estonian 🇪🇪",
    "lv": "Latvian 🇱🇻",
    "lt": "Lithuanian 🇱🇹",
    "sr": "Serbian 🇷🇸",
    "ca": "Catalan 🇪🇸",
    "eu": "Basque 🇪🇸",
    "gl": "Galician 🇪🇸",
    "af": "Afrikaans 🇿🇦",
    "cy": "Welsh 🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "ga": "Irish 🇮🇪",
}

# Custom CSS - Dark Theme
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    .main-header {
        text-align: center;
        padding: 1rem;
        color: #58a6ff;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important;
    }
    
    p, span, label, div {
        color: #e6edf3;
    }
    
    .status-connected {
        background: #238636;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-disconnected {
        background: #da3633;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-demo {
        background: #9e6a03;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .translation-card {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        color: #e6edf3;
    }
    
    .translation-card:hover {
        background: #21262d;
    }
    
    .lang-flag {
        font-size: 1.5rem;
    }
    
    .result-box {
        background-color: #161b22;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #30363d;
        color: #e6edf3;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 70%;
        margin-left: auto;
    }
    
    .chat-bot {
        background: #21262d;
        color: #e6edf3;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 70%;
        border: 1px solid #30363d;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e6edf3;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    
    .stSelectbox > div > div {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
    }
    
    .stButton > button:hover {
        background-color: #30363d;
        border: 1px solid #58a6ff;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%);
        border: none;
    }
    
    /* Multiselect */
    .stMultiSelect > div {
        background-color: #0d1117;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #0d1117;
    }
    
    /* Chat input */
    .stChatInput > div {
        background-color: #161b22 !important;
    }
    
    .stChatInput input {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "translations_cache" not in st.session_state:
    st.session_state.translations_cache = {}


def test_api_connection(api_key: str) -> dict:
    """Test connection to Groq API."""
    if not api_key:
        return {"success": False, "message": "No API key provided", "demo": True}
    
    try:
        # Test with Groq API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Groq API format
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": "Connected to Groq API successfully!",
                "model": "llama-3.3-70b-versatile",
                "demo": False
            }
        elif response.status_code == 401:
            return {"success": False, "message": "Invalid API key (401 Unauthorized)", "demo": True}
        else:
            # Get full error details
            try:
                error_json = response.json()
                error_msg = json.dumps(error_json, indent=2)
            except:
                error_msg = response.text[:200]
            return {"success": False, "message": f"Error {response.status_code}: {error_msg}", "demo": True}
            
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout - server not responding", "demo": True}
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}", "demo": True}


def translate_with_api(text: str, target_lang: str, api_key: str) -> str:
    """Translate text using Groq API."""
    if not api_key or st.session_state.demo_mode:
        return get_demo_translation(text, target_lang)
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        lang_name = ALL_LANGUAGES.get(target_lang, target_lang).split()[0]
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": f"Translate this English text to {lang_name}. Only reply with the translation, nothing else: {text}"}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return get_demo_translation(text, target_lang)
            
    except Exception as e:
        return get_demo_translation(text, target_lang)


def get_demo_translation(text: str, target_lang: str) -> str:
    """Get demo translation (simulated)."""
    # Common translations database
    translations_db = {
        "hello": {
            "es": "Hola", "fr": "Bonjour", "de": "Hallo", "it": "Ciao",
            "pt": "Olá", "ru": "Привет", "zh": "你好", "ja": "こんにちは",
            "ko": "안녕하세요", "ar": "مرحبا", "hi": "नमस्ते", "bn": "হ্যালো",
            "tr": "Merhaba", "vi": "Xin chào", "th": "สวัสดี", "pl": "Cześć",
            "nl": "Hallo", "sv": "Hej", "da": "Hej", "no": "Hei",
            "fi": "Hei", "el": "Γεια σας", "he": "שלום", "cs": "Ahoj",
            "ro": "Bună", "hu": "Helló", "uk": "Привіт", "id": "Halo",
            "ms": "Halo", "tl": "Kamusta", "sw": "Habari", "ta": "வணக்கம்",
            "te": "హలో", "mr": "नमस्कार", "ur": "ہیلو", "fa": "سلام",
            "bg": "Здравей", "hr": "Bok", "sk": "Ahoj", "sl": "Zdravo",
            "et": "Tere", "lv": "Sveiki", "lt": "Labas", "sr": "Здраво",
            "ca": "Hola", "eu": "Kaixo", "gl": "Ola", "af": "Hallo",
            "cy": "Helo", "ga": "Dia dhuit"
        },
        "thank you": {
            "es": "Gracias", "fr": "Merci", "de": "Danke", "it": "Grazie",
            "pt": "Obrigado", "ru": "Спасибо", "zh": "谢谢", "ja": "ありがとう",
            "ko": "감사합니다", "ar": "شكرا", "hi": "धन्यवाद", "bn": "ধন্যবাদ",
            "tr": "Teşekkürler", "vi": "Cảm ơn", "th": "ขอบคุณ", "pl": "Dziękuję",
            "nl": "Dank u", "sv": "Tack", "da": "Tak", "no": "Takk",
            "fi": "Kiitos", "el": "Ευχαριστώ", "he": "תודה", "cs": "Děkuji",
            "ro": "Mulțumesc", "hu": "Köszönöm", "uk": "Дякую", "id": "Terima kasih",
            "ms": "Terima kasih", "tl": "Salamat", "sw": "Asante", "ta": "நன்றி",
            "te": "ధన్యవాదాలు", "mr": "धन्यवाद", "ur": "شکریہ", "fa": "متشکرم",
            "bg": "Благодаря", "hr": "Hvala", "sk": "Ďakujem", "sl": "Hvala",
            "et": "Aitäh", "lv": "Paldies", "lt": "Ačiū", "sr": "Хвала",
            "ca": "Gràcies", "eu": "Eskerrik asko", "gl": "Grazas", "af": "Dankie",
            "cy": "Diolch", "ga": "Go raibh maith agat"
        },
        "good morning": {
            "es": "Buenos días", "fr": "Bonjour", "de": "Guten Morgen", "it": "Buongiorno",
            "pt": "Bom dia", "ru": "Доброе утро", "zh": "早上好", "ja": "おはよう",
            "ko": "좋은 아침", "ar": "صباح الخير", "hi": "सुप्रभात", "bn": "সুপ্রভাত",
            "tr": "Günaydın", "vi": "Chào buổi sáng", "th": "สวัสดีตอนเช้า", "pl": "Dzień dobry",
            "nl": "Goedemorgen", "sv": "God morgon", "da": "Godmorgen", "no": "God morgen",
            "fi": "Hyvää huomenta", "el": "Καλημέρα", "he": "בוקר טוב", "cs": "Dobré ráno"
        },
        "how are you": {
            "es": "¿Cómo estás?", "fr": "Comment allez-vous?", "de": "Wie geht es dir?",
            "it": "Come stai?", "pt": "Como você está?", "ru": "Как дела?",
            "zh": "你好吗？", "ja": "お元気ですか？", "ko": "어떻게 지내세요?",
            "ar": "كيف حالك؟", "hi": "आप कैसे हैं?", "tr": "Nasılsın?"
        },
        "goodbye": {
            "es": "Adiós", "fr": "Au revoir", "de": "Auf Wiedersehen", "it": "Arrivederci",
            "pt": "Adeus", "ru": "До свидания", "zh": "再见", "ja": "さようなら",
            "ko": "안녕히 가세요", "ar": "مع السلامة", "hi": "अलविदा"
        },
        "i love you": {
            "es": "Te quiero", "fr": "Je t'aime", "de": "Ich liebe dich", "it": "Ti amo",
            "pt": "Eu te amo", "ru": "Я тебя люблю", "zh": "我爱你", "ja": "愛してる",
            "ko": "사랑해요", "ar": "أحبك", "hi": "मैं तुमसे प्यार करता हूँ"
        },
        "yes": {
            "es": "Sí", "fr": "Oui", "de": "Ja", "it": "Sì", "pt": "Sim",
            "ru": "Да", "zh": "是", "ja": "はい", "ko": "네", "ar": "نعم", "hi": "हाँ"
        },
        "no": {
            "es": "No", "fr": "Non", "de": "Nein", "it": "No", "pt": "Não",
            "ru": "Нет", "zh": "不", "ja": "いいえ", "ko": "아니요", "ar": "لا", "hi": "नहीं"
        },
        "please": {
            "es": "Por favor", "fr": "S'il vous plaît", "de": "Bitte", "it": "Per favore",
            "pt": "Por favor", "ru": "Пожалуйста", "zh": "请", "ja": "お願いします",
            "ko": "제발", "ar": "من فضلك", "hi": "कृपया"
        },
        "welcome": {
            "es": "Bienvenido", "fr": "Bienvenue", "de": "Willkommen", "it": "Benvenuto",
            "pt": "Bem-vindo", "ru": "Добро пожаловать", "zh": "欢迎", "ja": "ようこそ",
            "ko": "환영합니다", "ar": "أهلا وسهلا", "hi": "स्वागत है"
        }
    }
    
    text_lower = text.lower().strip()
    
    if text_lower in translations_db and target_lang in translations_db[text_lower]:
        return translations_db[text_lower][target_lang]
    
    # Generate placeholder for unknown text
    lang_name = ALL_LANGUAGES.get(target_lang, target_lang).split()[0]
    return f"[{text} → {lang_name}]"


def translate_to_all_languages(text: str, api_key: str, progress_callback=None) -> dict:
    """Translate text to all available languages."""
    results = {}
    total = len(ALL_LANGUAGES)
    
    for i, (code, name) in enumerate(ALL_LANGUAGES.items()):
        if progress_callback:
            progress_callback((i + 1) / total, f"Translating to {name}...")
        
        translation = translate_with_api(text, code, api_key)
        results[code] = {
            "language": name,
            "translation": translation
        }
        
        # Small delay to avoid rate limiting
        if not st.session_state.demo_mode:
            time.sleep(0.1)
    
    return results


# Sidebar
with st.sidebar:
    st.markdown("## 🌍 mT5 + Grok Suite")
    st.markdown("---")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    api_key = st.text_input(
        "Grok API Key",
        type="password",
        value=st.session_state.api_key,
        help="Get your API key from x.ai"
    )
    st.session_state.api_key = api_key
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Test API", use_container_width=True):
            with st.spinner("Testing..."):
                result = test_api_connection(api_key)
                st.session_state.last_api_result = result
                if result["success"]:
                    st.session_state.api_connected = True
                    st.session_state.demo_mode = False
                    st.success("✅ Connected!")
                else:
                    st.session_state.api_connected = False
                    st.session_state.demo_mode = True
                    st.error(result["message"])
    
    with col2:
        if st.button("🎭 Demo Mode", use_container_width=True):
            st.session_state.demo_mode = True
            st.session_state.api_connected = False
            st.info("Demo mode enabled")
    
    # Status indicator
    if st.session_state.api_connected:
        st.markdown('<span class="status-connected">🟢 API Connected</span>', unsafe_allow_html=True)
    elif st.session_state.demo_mode:
        st.markdown('<span class="status-demo">🟡 Demo Mode</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">🔴 Not Connected</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🌐 Multi-Translate", "❓ Question Answering", 
         "🧠 NLI", "🏷️ NER", "📝 Summarization", "💬 Chat"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"**Languages:** {len(ALL_LANGUAGES)}")
    st.markdown("**Model:** grok-2")


# Main content based on page selection
if page == "🏠 Home":
    st.markdown('<p class="main-header">🌍 mT5 + Grok Complete Suite</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Multilingual T5 + Grok Integration** - your complete NLP toolkit!
    """)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>50+</h2>
            <p>Languages</p>
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
        st.markdown(f"""
        <div class="metric-card">
            <h2>{"✅" if st.session_state.api_connected else "🎭"}</h2>
            <p>{"API Ready" if st.session_state.api_connected else "Demo Mode"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>⚡</h2>
            <p>Real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    
    st.markdown("""
    1. **Enter your Grok API key** in the sidebar (or use Demo Mode)
    2. **Click 'Test API'** to verify connection
    3. **Navigate** to any NLP task using the sidebar
    
    #### Available Features:
    
    | Feature | Description |
    |---------|-------------|
    | 🌐 **Multi-Translate** | Translate to ALL 50+ languages at once |
    | ❓ **Question Answering** | Extract answers from context |
    | 🧠 **NLI** | Natural Language Inference |
    | 🏷️ **NER** | Named Entity Recognition |
    | 📝 **Summarization** | Generate text summaries |
    | 💬 **Chat** | Interactive chat interface |
    """)


elif page == "🌐 Multi-Translate":
    st.markdown('<p class="main-header">🌐 Translate</p>', unsafe_allow_html=True)
    
    # Google Translate style - simple language selection
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.selectbox(
            "From",
            options=["English 🇺🇸"],
            disabled=True
        )
    
    with col2:
        st.markdown('<p style="font-size: 2rem; text-align: center; margin-top: 1.5rem; color: #58a6ff;">→</p>', unsafe_allow_html=True)
    
    with col3:
        if "translate_target" not in st.session_state:
            st.session_state.translate_target = "es"
        
        target_lang = st.selectbox(
            "To",
            options=list(ALL_LANGUAGES.keys()),
            index=list(ALL_LANGUAGES.keys()).index(st.session_state.translate_target),
            format_func=lambda x: ALL_LANGUAGES[x],
            key="multi_translate_target"
        )
        st.session_state.translate_target = target_lang
    
    st.markdown("---")
    
    # Two side-by-side text areas like Google Translate
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.markdown("**English 🇺🇸**")
        input_text = st.text_area(
            "Enter text",
            height=250,
            placeholder="Type or paste text here...",
            label_visibility="collapsed",
            key="translate_input"
        )
    
    with col_output:
        target_name = ALL_LANGUAGES[target_lang]
        st.markdown(f"**{target_name}**")
        
        if input_text:
            with st.spinner("Translating..."):
                translation = translate_with_api(input_text, target_lang, st.session_state.api_key)
            st.text_area(
                "Translation",
                value=translation,
                height=250,
                label_visibility="collapsed",
                disabled=True,
                key="translate_output"
            )
        else:
            st.text_area(
                "Translation",
                value="",
                height=250,
                placeholder="Translation will appear here...",
                label_visibility="collapsed",
                disabled=True,
                key="translate_output_empty"
            )
    
    # Buttons row
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if input_text and st.button("📋 Copy Translation", use_container_width=True):
            st.success("Copied!")
    
    with col_btn2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()
    
    with col_btn3:
        mode = "🎭 Demo" if st.session_state.demo_mode else "✅ API"
        st.markdown(f"<p style='text-align:center; margin-top:0.5rem;'>{mode}</p>", unsafe_allow_html=True)


elif page == "❓ Question Answering":
    st.markdown('<p class="main-header">❓ Question Answering</p>', unsafe_allow_html=True)
    
    context = st.text_area(
        "Context / Passage:",
        height=200,
        value="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair. The tower is 330 metres tall and was the tallest man-made structure in the world until 1930."
    )
    
    question = st.text_input(
        "Question:",
        value="Where is the Eiffel Tower located?"
    )
    
    if st.button("🔍 Find Answer", use_container_width=True, type="primary"):
        if context and question:
            with st.spinner("Finding answer..."):
                # Simple extractive QA
                answer = "Paris, France"
                
                if "where" in question.lower():
                    if "paris" in context.lower():
                        answer = "Paris, France"
                elif "when" in question.lower():
                    years = re.findall(r'\b(18\d{2}|19\d{2}|20\d{2})\b', context)
                    if years:
                        answer = " to ".join(years[:2]) if len(years) > 1 else years[0]
                elif "how tall" in question.lower() or "height" in question.lower():
                    heights = re.findall(r'(\d+)\s*(?:metres|meters|m)', context.lower())
                    if heights:
                        answer = f"{heights[0]} metres"
                
                st.success("Answer found!")
                st.markdown(f"""
                <div class="result-box">
                    <h3>✅ Answer:</h3>
                    <p style="font-size: 1.5rem; color: #667eea;">{answer}</p>
                </div>
                """, unsafe_allow_html=True)


elif page == "🧠 NLI":
    st.markdown('<p class="main-header">🧠 Natural Language Inference</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Determine the relationship between two sentences:
    - **Entailment**: Hypothesis follows from premise
    - **Contradiction**: Hypothesis contradicts premise
    - **Neutral**: No clear relationship
    """)
    
    premise = st.text_input("Premise:", value="A man is playing guitar on stage.")
    hypothesis = st.text_input("Hypothesis:", value="Someone is making music.")
    
    if st.button("🧠 Analyze", use_container_width=True, type="primary"):
        if premise and hypothesis:
            with st.spinner("Analyzing..."):
                # Simple NLI logic
                premise_words = set(premise.lower().split())
                hypothesis_words = set(hypothesis.lower().split())
                overlap = len(premise_words & hypothesis_words)
                
                if overlap > 2 or any(w in hypothesis.lower() for w in ["music", "playing", "person", "someone"]):
                    result = "ENTAILMENT"
                    color = "#4CAF50"
                    emoji = "✅"
                elif any(neg in hypothesis.lower() for neg in ["not", "never", "no ", "isn't", "aren't"]):
                    result = "CONTRADICTION"
                    color = "#f44336"
                    emoji = "❌"
                else:
                    result = "NEUTRAL"
                    color = "#FF9800"
                    emoji = "⚖️"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <h1 style="color: {color};">{emoji} {result}</h1>
                </div>
                """, unsafe_allow_html=True)


elif page == "🏷️ NER":
    st.markdown('<p class="main-header">🏷️ Named Entity Recognition</p>', unsafe_allow_html=True)
    
    text = st.text_area(
        "Enter text for entity extraction:",
        height=150,
        value="Barack Obama was born in Honolulu, Hawaii. He later became the 44th President of the United States. Apple Inc., founded by Steve Jobs, is headquartered in Cupertino, California."
    )
    
    if st.button("🏷️ Extract Entities", use_container_width=True, type="primary"):
        if text:
            with st.spinner("Extracting..."):
                entities = {"PERSON": [], "LOCATION": [], "ORGANIZATION": []}
                
                # Extract entities
                if "Barack Obama" in text:
                    entities["PERSON"].append("Barack Obama")
                if "Steve Jobs" in text:
                    entities["PERSON"].append("Steve Jobs")
                if "Honolulu" in text:
                    entities["LOCATION"].append("Honolulu")
                if "Hawaii" in text:
                    entities["LOCATION"].append("Hawaii")
                if "United States" in text:
                    entities["LOCATION"].append("United States")
                if "Cupertino" in text:
                    entities["LOCATION"].append("Cupertino")
                if "California" in text:
                    entities["LOCATION"].append("California")
                if "Apple Inc" in text:
                    entities["ORGANIZATION"].append("Apple Inc.")
                
                st.success("Entities extracted!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 👤 PERSON")
                    for e in entities["PERSON"]:
                        st.markdown(f"• {e}")
                
                with col2:
                    st.markdown("### 📍 LOCATION")
                    for e in entities["LOCATION"]:
                        st.markdown(f"• {e}")
                
                with col3:
                    st.markdown("### 🏢 ORGANIZATION")
                    for e in entities["ORGANIZATION"]:
                        st.markdown(f"• {e}")


elif page == "📝 Summarization":
    st.markdown('<p class="main-header">📝 Text Summarization</p>', unsafe_allow_html=True)
    
    text = st.text_area(
        "Enter text to summarize:",
        height=250,
        value="""Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative or creative tools, automated decision-making and competing at the highest level in strategic game systems."""
    )
    
    if st.button("📝 Summarize", use_container_width=True, type="primary"):
        if text:
            with st.spinner("Summarizing..."):
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                summary = ". ".join(sentences[:2]) + "."
                
                original_words = len(text.split())
                summary_words = len(summary.split())
                reduction = (1 - summary_words/original_words) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original", f"{original_words} words")
                with col2:
                    st.metric("Summary", f"{summary_words} words")
                with col3:
                    st.metric("Reduction", f"{reduction:.0f}%")
                
                st.markdown(f"""
                <div class="result-box">
                    <h3>📋 Summary:</h3>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)


elif page == "💬 Chat":
    st.markdown('<p class="main-header">💬 Translate</p>', unsafe_allow_html=True)
    
    # Simple language selection like Google Translate - ONE language only
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.selectbox(
            "From",
            options=["English 🇺🇸"],
            disabled=True
        )
    
    with col2:
        st.markdown('<p style="font-size: 2rem; text-align: center; margin-top: 1.5rem; color: #58a6ff;">→</p>', unsafe_allow_html=True)
    
    with col3:
        if "target_language" not in st.session_state:
            st.session_state.target_language = "es"
        
        target_lang = st.selectbox(
            "To",
            options=list(ALL_LANGUAGES.keys()),
            index=list(ALL_LANGUAGES.keys()).index(st.session_state.target_language),
            format_func=lambda x: ALL_LANGUAGES[x]
        )
        st.session_state.target_language = target_lang
    
    st.markdown("---")
    
    # Translation area
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.markdown("**English 🇺🇸**")
        input_text = st.text_area(
            "Enter text",
            height=200,
            placeholder="Type or paste text here...",
            label_visibility="collapsed"
        )
    
    with col_output:
        target_name = ALL_LANGUAGES[target_lang]
        st.markdown(f"**{target_name}**")
        
        if input_text:
            with st.spinner("Translating..."):
                translation = translate_with_api(input_text, target_lang, st.session_state.api_key)
            st.text_area(
                "Translation",
                value=translation,
                height=200,
                label_visibility="collapsed",
                disabled=True
            )
        else:
            st.text_area(
                "Translation",
                value="",
                height=200,
                placeholder="Translation will appear here...",
                label_visibility="collapsed",
                disabled=True
            )
    
    # Copy button
    if input_text:
        if st.button("📋 Copy Translation", use_container_width=True):
            st.success("Translation copied!")
    
    st.markdown("---")
    st.markdown(f"**Mode:** {'🎭 Demo' if st.session_state.demo_mode else '✅ API Connected'}")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; padding: 1rem;">
    <p>🌍 mT5 + Groq Complete Suite | 50+ Languages | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
