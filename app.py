import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="AI Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .question-box:hover {
        background-color: #d1e7dd;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_qa_dataset():
    """Load sample questions from the dataset"""
    try:
        questions = []
        with open("qa_dataset_improved_short.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line.strip())
                questions.append(qa['question'])
        return questions[:5]  # Return first 5 questions
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return [
            "What is science?",
            "What are the main branches of science?",
            "What is physics?",
            "What is chemistry?",
            "What is the scientific method?"
        ]

def validate_model_directory(model_path):
    """Check if model directory has required files"""
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    if os.path.exists(model_path):
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        return True
    return False

def get_model_paths():
    """Get model paths from configuration or use defaults"""
    try:
        # Try to load from model_config.py if it exists
        if os.path.exists("model_config.py"):
            import model_config
            return {
                "Base Model": model_config.BASE_MODEL_REPO,
                "Fine-tuned Model": model_config.FINE_TUNED_MODEL_REPO
            }
    except Exception:
        pass
    
    # Use the uploaded models on HuggingFace Hub
    return {
        "Base Model": "alexputhiyadom/science-qa-base",
        "Fine-tuned Model": "alexputhiyadom/science-qa-finetuned"
    }

class QAModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        try:
            # Get HuggingFace token from environment or secrets
            hf_token = st.secrets.get("HUGGINGFACE_TOKEN", None)
            
            # Determine the actual model name/path
            if self.model_path in ["base_model", "fine_tuned_model"]:
                # Check if local model exists
                if validate_model_directory(self.model_path):
                    model_name = self.model_path
                    use_auth_token = None
                else:
                    # Use HuggingFace Hub models
                    model_paths = get_model_paths()
                    if self.model_path == "base_model":
                        model_name = model_paths["Base Model"]
                    else:
                        model_name = model_paths["Fine-tuned Model"]
                    use_auth_token = hf_token
            else:
                # Direct HuggingFace Hub path
                model_name = self.model_path
                use_auth_token = hf_token
            
            with st.spinner(f"Loading model {model_name}..."):
                # Load tokenizer with authentication if needed
                if use_auth_token:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        use_auth_token=use_auth_token
                    )
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name, 
                        use_auth_token=use_auth_token
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model.to(self.device)
                self.model.eval()
            
            st.success(f"✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            if "401" in str(e):
                st.error("Authentication failed. Please check your HuggingFace token.")
            elif "404" in str(e):
                st.error("Model not found. Please check the model path.")
            else:
                st.error("Please check your internet connection and try again.")
            self.model = None
            self.tokenizer = None
    
    def generate_answer(self, question: str, context: str = "") -> str:
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please check the model path."
        
        try:
            if context:
                input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            else:
                input_text = f"Question: {question}\nAnswer:"
            
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation parameters optimized for short answers
            generation_params = {
                "max_length": 128,  # Shorter for concise answers
                "num_beams": 3,
                "early_stopping": True,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.strip()
            
            # Clean up the answer
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            if len(answer) < 5 or answer.lower() in ['', 'none', 'unknown', 'error', 'n/a']:
                return "I'm sorry, I couldn't generate a proper answer for this question. Please try rephrasing your question."
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 AI Question Answering System</h1>', unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.title("⚙️ Settings")
    
    # Get model options
    model_options = get_model_paths()
    
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        list(model_options.keys()),
        index=0  # Default to base model
    )
    
    model_path = model_options[selected_model]
    
    # Load model
    if 'qa_model' not in st.session_state or st.session_state.get('current_model') != model_path:
        st.session_state.qa_model = QAModel(model_path)
        st.session_state.current_model = model_path
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Model:** {selected_model}")
    st.sidebar.markdown(f"**Model Path:** {model_path}")
    st.sidebar.markdown(f"**Device:** {st.session_state.qa_model.device}")
    
    # Check if using HuggingFace token
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", None)
    if hf_token:
        st.sidebar.success("✅ Using HuggingFace authentication")
    else:
        st.sidebar.warning("⚠️ No HuggingFace token configured")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Ask a Question")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is science?",
            key="question_input"
        )
        
        # Context input (optional)
        context = st.text_area(
            "Context (optional):",
            placeholder="Provide additional context if needed...",
            height=100,
            key="context_input"
        )
        
        # Generate button
        if st.button("🚀 Generate Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Generating answer..."):
                    answer = st.session_state.qa_model.generate_answer(question, context)
                
                st.markdown("### 💡 Answer:")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.markdown("### 📋 Sample Questions")
        
        # Load sample questions
        sample_questions = load_qa_dataset()
        
        for i, sample_q in enumerate(sample_questions, 1):
            if st.button(f"Q{i}: {sample_q}", key=f"sample_{i}", use_container_width=True):
                st.session_state.question_input = sample_q
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI system can answer questions about science topics based on the provided content.
        
        **Features:**
        - 🤖 Two model options (base and fine-tuned)
        - 📝 Short, concise answers
        - 🔄 Context-aware responses
        - ⚡ Fast generation
        - 🌐 Cloud-ready deployment
        - 🔐 Secure HuggingFace authentication
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit • Powered by Transformers • Fine-tuned for Science Q&A"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 