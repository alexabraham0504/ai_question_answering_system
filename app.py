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
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #28a745;
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
    }
    .question-box {
        background-color: #e8f4fd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid #d1e7dd;
    }
    .question-box:hover {
        background-color: #d1e7dd;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .answer-header {
        color: #28a745;
        font-weight: bold;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
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
        self.device = self._get_optimal_device()
        self.load_model()
    
    def _get_optimal_device(self):
        """Get the optimal device for model inference"""
        if torch.cuda.is_available():
            # Check if CUDA is properly configured
            try:
                # Get GPU info
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
                
                st.sidebar.success(f"🚀 GPU Available: {gpu_name}")
                st.sidebar.info(f"📊 GPU Memory: {gpu_memory:.1f} GB")
                
                # Set CUDA optimization flags
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                return torch.device("cuda")
            except Exception as e:
                st.sidebar.warning(f"⚠️ GPU detection error: {str(e)}")
                return torch.device("cpu")
        else:
            st.sidebar.warning("⚠️ No GPU detected. Using CPU.")
            return torch.device("cpu")
    
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
            
            with st.spinner(f"Loading model {model_name} on {self.device}..."):
                try:
                    # Load tokenizer with authentication if needed
                    if use_auth_token:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name, 
                            use_auth_token=use_auth_token
                        )
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name, 
                            use_auth_token=use_auth_token,
                            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                        )
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                        )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Move model to device with optimization
                    self.model.to(self.device)
                    self.model.eval()
                    
                    # Enable GPU optimizations if available
                    if self.device.type == "cuda":
                        # Enable memory efficient attention if available
                        if hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
                            try:
                                self.model.enable_xformers_memory_efficient_attention()
                            except:
                                pass
                        
                        # Clear GPU cache
                        torch.cuda.empty_cache()
                    
                    device_status = "GPU" if self.device.type == "cuda" else "CPU"
                    st.success(f"✅ Model loaded successfully on {device_status}!")
                    
                except ImportError as import_error:
                    if "sentencepiece" in str(import_error):
                        st.error("❌ SentencePiece library error. This is a known issue with Python 3.13.")
                        st.error("💡 Solution: Please set Python version to 3.11 in Streamlit Cloud settings.")
                        st.info("📋 Go to your app settings → Advanced settings → Python version → Select 3.11")
                    else:
                        raise import_error
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            if "401" in str(e):
                st.error("Authentication failed. Please check your HuggingFace token.")
            elif "404" in str(e):
                st.error("Model not found. Please check the model path.")
            elif "sentencepiece" in str(e):
                st.error("💡 Please set Python version to 3.11 in Streamlit Cloud settings.")
            elif "CUDA" in str(e) or "cuda" in str(e):
                st.error("❌ GPU error detected. Falling back to CPU.")
                self.device = torch.device("cpu")
                st.info("🔄 Please restart the app to try GPU again or continue with CPU.")
            else:
                st.error("Please check your internet connection and try again.")
            self.model = None
            self.tokenizer = None
    
    def generate_answer(self, question: str, context: str = "") -> str:
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please check the model path."
        
        try:
            # Improved prompt engineering for better answers
            if context:
                input_text = f"Context: {context}\n\nQuestion: {question}\n\nProvide a clear and concise answer:"
            else:
                input_text = f"Question: {question}\n\nProvide a clear and concise answer:"
            
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512,  # Increased for better context handling
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Improved generation parameters for better quality answers
            generation_params = {
                "max_length": 200,  # Increased for more detailed answers
                "min_length": 20,   # Ensure minimum answer length
                "num_beams": 4,     # Increased for better quality
                "early_stopping": True,
                "do_sample": True,  # Enable sampling for more natural responses
                "temperature": 0.7, # Add some creativity
                "top_p": 0.9,      # Nucleus sampling
                "pad_token_id": self.tokenizer.pad_token_id,
                "repetition_penalty": 1.2  # Prevent repetitive text
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.strip()
            
            # Enhanced answer cleaning and formatting
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            elif "answer:" in answer:
                answer = answer.split("answer:")[-1].strip()
            
            # Remove any remaining prompt text
            if "Question:" in answer:
                answer = answer.split("Question:")[0].strip()
            if "Context:" in answer:
                answer = answer.split("Context:")[0].strip()
            
            # Clean up common artifacts
            answer = answer.replace("Provide a clear and concise answer:", "").strip()
            answer = answer.replace("Provide a clear answer:", "").strip()
            
            # Validate answer quality
            if len(answer) < 10 or answer.lower() in ['', 'none', 'unknown', 'error', 'n/a', 'i don\'t know', 'i do not know']:
                return "I'm sorry, I couldn't generate a proper answer for this question. Please try rephrasing your question or providing more context."
            
            # Ensure the answer starts with a capital letter and ends properly
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            
            # Add period if missing
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def check_gpu_setup():
    """Check GPU setup and provide diagnostics"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 GPU Diagnostics")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        st.sidebar.success(f"✅ CUDA Available - {gpu_count} GPU(s) detected")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            st.sidebar.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check PyTorch CUDA version
        pytorch_cuda = torch.version.cuda
        st.sidebar.info(f"PyTorch CUDA: {pytorch_cuda}")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            st.sidebar.success(f"✅ GPU memory test: {allocated:.3f} GB allocated")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            st.sidebar.error(f"❌ GPU memory test failed: {str(e)}")
    else:
        st.sidebar.error("❌ CUDA not available")
        st.sidebar.markdown("**Possible reasons:**")
        st.sidebar.markdown("- No NVIDIA GPU")
        st.sidebar.markdown("- CUDA drivers not installed")
        st.sidebar.markdown("- PyTorch not compiled with CUDA")
        st.sidebar.markdown("- Environment issues")

def main():
    # Initialize session state
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    if 'context_input' not in st.session_state:
        st.session_state.context_input = ""
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = ""
    if 'show_gpu_diagnostics' not in st.session_state:
        st.session_state.show_gpu_diagnostics = False
    
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
    
    # Enhanced device information
    device = st.session_state.qa_model.device
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        st.sidebar.markdown(f"**Device:** 🚀 GPU ({gpu_name})")
        st.sidebar.markdown(f"**GPU Memory:** {gpu_memory:.1f} GB")
        
        # Show current GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            st.sidebar.markdown(f"**Used:** {allocated:.2f} GB")
            st.sidebar.markdown(f"**Cached:** {cached:.2f} GB")
    else:
        st.sidebar.markdown(f"**Device:** 💻 CPU")
        st.sidebar.markdown("**Note:** GPU not available")
    
    # Check if using HuggingFace token
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", None)
    if hf_token:
        st.sidebar.success("✅ Using HuggingFace authentication")
    else:
        st.sidebar.warning("⚠️ No HuggingFace token configured")
    
    # GPU Diagnostics button
    if st.sidebar.button("🔧 GPU Diagnostics", use_container_width=True):
        st.session_state.show_gpu_diagnostics = not st.session_state.show_gpu_diagnostics
    
    # Show GPU diagnostics if requested
    if st.session_state.show_gpu_diagnostics:
        check_gpu_setup()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Ask a Question")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.question_input,
            placeholder="e.g., What is science?",
            key="question_input_new"
        )
        
        # Context input (optional)
        context = st.text_area(
            "Context (optional):",
            value=st.session_state.context_input,
            placeholder="Provide additional context if needed...",
            height=100,
            key="context_input_new"
        )
        
        # Generate button
        if st.button("🚀 Generate Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Generating answer..."):
                    answer = st.session_state.qa_model.generate_answer(question, context)
                    st.session_state.current_answer = answer
                    st.session_state.question_input = question
                    st.session_state.context_input = context
                st.rerun()
            else:
                st.warning("Please enter a question.")
        
        # Display answer
        if st.session_state.current_answer:
            st.markdown('<h3 class="answer-header">💡 Answer</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{st.session_state.current_answer}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Sample Questions")
        
        # Load sample questions
        sample_questions = load_qa_dataset()
        
        for i, sample_q in enumerate(sample_questions, 1):
            if st.button(f"Q{i}: {sample_q}", key=f"sample_{i}", use_container_width=True):
                st.session_state.question_input = sample_q
                st.session_state.context_input = ""
                st.session_state.current_answer = ""
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
        - 🚀 GPU acceleration (when available)
        - 🌐 Cloud-ready deployment
        - 🔐 Secure HuggingFace authentication
        """)
        
        # Show GPU optimization info if GPU is available
        if torch.cuda.is_available():
            st.markdown("**🚀 GPU Optimizations Active:**")
            st.markdown("""
            - ✅ Mixed precision (FP16)
            - ✅ CUDA optimizations
            - ✅ Memory efficient attention
            - ✅ Automatic GPU memory management
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