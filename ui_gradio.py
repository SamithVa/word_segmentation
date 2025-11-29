import gradio as gr
import time
import os
import torch
import pickle

# Import models
from models import HMMTokenizer, RNNTokenizer, LSTMTokenizer, TransformerTokenizer
from models.classical import FMM, BMM, BiMM

class ModelWrapper:
    def __init__(self):
        print("Initializing Chinese Word Segmentation System...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training files
        training_files = [
            'icwb2-data/training/pku_training.utf8',
            'icwb2-data/training/msr_training.utf8',
            'icwb2-data/training/cityu_training.utf8',
            'icwb2-data/training/as_training.utf8'
        ]
        self.existing_files = [f for f in training_files if os.path.exists(f)]
        
        # Initialize basic models
        self._init_basic_models()
        
        # Deep learning models (lazy loading)
        self.transformer_tokenizer = None
        self.rnn_tokenizer = None
        self.bilstm_tokenizer = None
    
    def _init_basic_models(self):
        """Initialize dictionary and statistical models"""
        try:
            self.mm_tokenizer = MMTokenizer(dictionary_path=self.existing_files)
        except Exception as e:
            print(f"Error loading dictionary models: {e}")
            self.mm_tokenizer = None
        
        try:
            self.hmm_tokenizer = HMMTokenizer(smoothing=1e-8)
            model_path = 'ml_based/model_weight/hmm_model.pkl'
            
            if os.path.exists(model_path):
                self.hmm_tokenizer.load_model(model_path)
            else:
                self.hmm_tokenizer.train(self.existing_files)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.hmm_tokenizer.save_model(model_path)
        except Exception as e:
            print(f"Error loading HMM model: {e}")
            self.hmm_tokenizer = None
    
    def _load_model(self, model_type, model_path, vocab_path, model_class, model_params):
        """Generic model loader for deep learning models"""
        tokenizer_attr = f"{model_type.lower()}_tokenizer"
        if getattr(self, tokenizer_attr) is not None:
            return True
            
        try:
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)

            char2idx = vocab_data['char2idx']
            idx2tag = vocab_data['idx2tag']
            tag2idx = vocab_data['tag2idx']

            # Initialize model
            model = model_class(**model_params, vocab_size=len(char2idx),
                              num_classes=len(tag2idx)).to(self.device)

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Create tokenizer
            setattr(self, tokenizer_attr, BaseTokenizer(model, char2idx, idx2tag, self.device))
            return True
        except Exception as e:
            setattr(self, tokenizer_attr, None)
            return False
    
    def _load_transformer_model(self):
        return self._load_model(
            "Transformer",
            'transformer_based/d128/transformer_seg_best.pth',
            'transformer_based/d128/vocab.pkl',
            TransformerTokenizer,
            {"d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1}
        )
    
    def _load_rnn_model(self):
        return self._load_model(
            "RNN",
            'rnn/model_weight/rnn_seg_best.pth',
            'rnn/model_weight/vocab.pkl',
            RNNTokenizer,
            {"d_model": 128, "hidden_dim": 256, "num_layers": 2}
        )
    
    def _load_bilstm_model(self):
        return self._load_model(
            "BiLSTM",
            'lstm/model_weight/bilstm_seg_best.pth',
            'lstm/model_weight/vocab.pkl',
            LSTMTokenizer,
            {"embedding_dim": 128, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3}
        )
    
    # Segmentation methods
    def segment_fmm(self, text): return self.mm_tokenizer.forward_maximum_matching(text)
    def segment_bmm(self, text): return self.mm_tokenizer.backward_maximum_matching(text)
    def segment_bimm(self, text): return self.mm_tokenizer.bidirectional_maximum_matching(text)
    def segment_hmm(self, text): return self.hmm_tokenizer.tokenize(text)
    
    def segment_transformer(self, text):
        if not self._load_transformer_model(): return ["Error", "loading", "Transformer", "model"]
        return self.transformer_tokenizer.tokenize(text)
    
    def segment_rnn(self, text):
        if not self._load_rnn_model(): return ["Error", "loading", "RNN", "model"]
        return self.rnn_tokenizer.tokenize(text)
    
    def segment_bilstm(self, text):
        if not self._load_bilstm_model(): return ["Error", "loading", "BiLSTM", "model"]
        return self.bilstm_tokenizer.tokenize(text)

    def segment(self, text, algorithm):
        """Main segmentation function"""
        if not text or not text.strip(): return "", "0 ms"
        
        start_time = time.time()
        
        # Algorithm dispatch
        algorithm_map = {
            "FMM": self.segment_fmm,
            "BMM": self.segment_bmm,
            "BiMM": self.segment_bimm,
            "HMM": self.segment_hmm,
            "RNN": self.segment_rnn,
            "BiLSTM": self.segment_bilstm,
            "Transformer": self.segment_transformer,
        }
        
        result = algorithm_map.get(algorithm, lambda: ["Unknown", "algorithm"])(text)
        
        output_str = " / ".join(result)
        cost_time = (time.time() - start_time) * 1000
        
        return output_str, f"{cost_time:.2f} ms"

# Initialize model
model = ModelWrapper()

# UI Functions
def process_text(text, algo_choice):
    if not text: return "", "0 ms"
    
    segmented, time_taken = model.segment(text, algo_choice)
    return segmented, time_taken

# Build UI
with gr.Blocks(title="Chinese Word Segmentation System") as demo:
    gr.Markdown("# 🔤 Chinese Word Segmentation System")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Sentence", placeholder="输入中文句子", lines=3)
            algo_selector = gr.Dropdown([
                "FMM", "BMM", "BiMM", "HMM", "RNN", "BiLSTM", "Transformer"
            ], label="Select Algorithm", value="FMM")
            btn = gr.Button("Segment", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Segmentation Result", lines=3, interactive=False)
            time_display = gr.Label(label="Inference Time")
    
    btn.click(fn=process_text, inputs=[input_text, algo_selector], outputs=[output_text, time_display])
    
    gr.Examples([
        ["今天天气不错"], ["中国人民站起来了"], ["机器学习是人工智能的重要分支"]
    ], inputs=[input_text])

# Launch
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)