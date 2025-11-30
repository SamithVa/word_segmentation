import gradio as gr
import time
import os
import torch
import pickle

# Import THULAC
try:
    import thulac
    THULAC_AVAILABLE = True
except ImportError:
    THULAC_AVAILABLE = False
    print("THULAC not installed. Install with: pip install thulac")

# Import models
from models.hmm_seg import HMMSeg
from models.classical import FMM, BMM, BiMM
from models.rnn_seg import RNNSeg
from models.lstm_seg import LSTMSeg
from models.transformer_seg import TransformerSeg

# Import utility and config
from utils import BaseTokenizer
from config import (
    DATASETS, SAVED_MODELS_DIR, TRAINING_CONFIG,
    RNN_CONFIG, LSTM_CONFIG, TRANSFORMER_CONFIG
)

class ModelWrapper:
    def __init__(self):
        print("Initializing Chinese Word Segmentation System...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training files from config
        training_files = [dataset['train'] for dataset in DATASETS.values()]
        self.existing_files = [f for f in training_files if os.path.exists(f)]

        # Initialize classical models (no training needed - just load dictionary)
        self._init_classical_models()

        # Initialize HMM model
        self._init_hmm_model()

        # Deep learning models (lazy loading)
        self.transformer_tokenizer = None
        self.rnn_tokenizer = None
        self.lstm_tokenizer = None

        # Initialize THULAC model
        self._init_thulac_model()

    def _init_classical_models(self):
        """Initialize classical models (FMM, BMM, BiMM)"""
        try:
            self.fmm_model = FMM()
            self.fmm_model.load_dict(self.existing_files)
            print(f"FMM model initialized with {len(self.fmm_model.vocab)} words")
        except Exception as e:
            print(f"Error initializing FMM model: {e}")
            self.fmm_model = None

        try:
            self.bmm_model = BMM()
            self.bmm_model.load_dict(self.existing_files)
            print(f"BMM model initialized with {len(self.bmm_model.vocab)} words")
        except Exception as e:
            print(f"Error initializing BMM model: {e}")
            self.bmm_model = None

        try:
            self.bimm_model = BiMM(self.existing_files)
            print(f"BiMM model initialized")
        except Exception as e:
            print(f"Error initializing BiMM model: {e}")
            self.bimm_model = None

    def _init_hmm_model(self):
        """Initialize HMM model"""
        try:
            self.hmm_model = HMMSeg()
            model_path = 'outputs/saved_models/hmm/hmm_model.pkl'

            if os.path.exists(model_path):
                self.hmm_model.load_model(model_path)
                print("HMM model loaded")
            else:
                print("HMM model not found, training new one...")
                self.hmm_model.train(self.existing_files)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.hmm_model.save_model(model_path)
                print("HMM model trained and saved")
        except Exception as e:
            print(f"Error initializing HMM model: {e}")
            self.hmm_model = None

    def _init_thulac_model(self):
        """Initialize THULAC model"""
        if not THULAC_AVAILABLE:
            self.thulac_model = None
            return
        try:
            # seg_only=True for segmentation only (no POS tagging)
            self.thulac_model = thulac.thulac(seg_only=True)
            print("THULAC model initialized")
        except Exception as e:
            print(f"Error initializing THULAC model: {e}")
            self.thulac_model = None

    def _load_model(self, model_type, model_path, vocab_path, model_class, model_params):
        """Generic model loader for deep learning models"""
        tokenizer_attr = f"{model_type.lower()}_tokenizer"
        if getattr(self, tokenizer_attr, None) is not None:
            return True

        try:
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)

            char2idx = vocab_data['char2idx']
            idx2tag = vocab_data['idx2tag']
            tag2idx = vocab_data['tag2idx']

            # Initialize model
            model = model_class(
                vocab_size=len(char2idx),
                **model_params,
                num_classes=len(tag2idx)
            ).to(self.device)

            # Load weights - handle both wrapped and direct state_dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            # Create tokenizer
            setattr(self, tokenizer_attr, BaseTokenizer(model, char2idx, idx2tag, self.device))
            print(f"{model_type} model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            setattr(self, tokenizer_attr, None)
            return False

    def _load_transformer_model(self):
        return self._load_model(
            "Transformer",
            f'{SAVED_MODELS_DIR}/transformer/transformer_seg_best.pth',
            f'{SAVED_MODELS_DIR}/transformer/vocab.pkl',
            TransformerSeg,
            {
                "d_model": TRANSFORMER_CONFIG['d_model'],
                "nhead": TRANSFORMER_CONFIG['nhead'],
                "num_layers": TRANSFORMER_CONFIG['num_layers'],
                "dropout": TRANSFORMER_CONFIG['dropout'],
                "max_len": TRAINING_CONFIG['max_len']
            }
        )

    def _load_rnn_model(self):
        return self._load_model(
            "RNN",
            f'{SAVED_MODELS_DIR}/rnn/rnn_seg_best.pth',
            f'{SAVED_MODELS_DIR}/rnn/vocab.pkl',
            RNNSeg,
            {
                "d_model": RNN_CONFIG['d_model'],
                "hidden_dim": RNN_CONFIG['hidden_dim'],
                "num_layers": RNN_CONFIG['num_layers']
            }
        )

    def _load_lstm_model(self):
        return self._load_model(
            "LSTM",
            f'{SAVED_MODELS_DIR}/lstm/lstm_seg_best.pth',
            f'{SAVED_MODELS_DIR}/lstm/vocab.pkl',
            LSTMSeg,
            {
                "embedding_dim": LSTM_CONFIG['embedding_dim'],
                "hidden_dim": LSTM_CONFIG['hidden_dim'],
                "num_layers": LSTM_CONFIG['num_layers'],
                "dropout": LSTM_CONFIG['dropout']
            }
        )

    # Segmentation methods
    def segment_fmm(self, text):
        if not self.fmm_model:
            return ["Error", "loading", "FMM", "model"]
        return self.fmm_model.tokenize(text)

    def segment_bmm(self, text):
        if not self.bmm_model:
            return ["Error", "loading", "BMM", "model"]
        return self.bmm_model.tokenize(text)

    def segment_bimm(self, text):
        if not self.bimm_model:
            return ["Error", "loading", "BiMM", "model"]
        return self.bimm_model.tokenize(text)

    def segment_hmm(self, text):
        if not self.hmm_model:
            return ["Error", "loading", "HMM", "model"]
        return self.hmm_model.tokenize(text)

    def segment_transformer(self, text):
        if not self._load_transformer_model():
            return ["Error", "loading", "Transformer", "model"]
        return self.transformer_tokenizer.tokenize(text)

    def segment_rnn(self, text):
        if not self._load_rnn_model():
            return ["Error", "loading", "RNN", "model"]
        return self.rnn_tokenizer.tokenize(text)

    def segment_lstm(self, text):
        if not self._load_lstm_model():
            return ["Error", "loading", "LSTM", "model"]
        return self.lstm_tokenizer.tokenize(text)

    def segment_thulac(self, text):
        if not self.thulac_model:
            return ["Error", "loading", "THULAC", "model"]
        # THULAC returns list of [word, pos] pairs, we only need words
        result = self.thulac_model.cut(text, text=False)
        return [word for word, _ in result]

    def segment(self, text, algorithm):
        """Main segmentation function"""
        if not text or not text.strip():
            return "", "0 ms"

        start_time = time.time()

        # Algorithm dispatch
        algorithm_map = {
            "FMM": self.segment_fmm,
            "BMM": self.segment_bmm,
            "BiMM": self.segment_bimm,
            "HMM": self.segment_hmm,
            "RNN": self.segment_rnn,
            "LSTM": self.segment_lstm,
            "Transformer": self.segment_transformer,
            "THULAC": self.segment_thulac,
        }

        result = algorithm_map.get(algorithm, lambda: ["Unknown", "algorithm"])(text)

        output_str = " / ".join(result)
        cost_time = (time.time() - start_time) * 1000

        return output_str, f"{cost_time:.2f} ms"

# Initialize model
model = ModelWrapper()

# UI Functions
def process_text(text, algo_choice):
    if not text:
        return "", "0 ms"

    segmented, time_taken = model.segment(text, algo_choice)
    return segmented, time_taken

def process_all_models(text):
    """Run all models and return their outputs"""
    if not text or not text.strip():
        return [""] * 8 + ["0 ms"]
    
    algorithms = ["FMM", "BMM", "BiMM", "HMM", "RNN", "LSTM", "Transformer", "THULAC"]
    results = []
    
    total_start = time.time()
    for algo in algorithms:
        segmented, _ = model.segment(text, algo)
        results.append(segmented)
    total_time = (time.time() - total_start) * 1000
    
    return results + [f"{total_time:.2f} ms"]

# Build UI
with gr.Blocks(title="Chinese Word Segmentation System") as demo:
    gr.Markdown("# 🔤 Chinese Word Segmentation System")

    with gr.Tabs():
        # Tab 1: Single Model
        with gr.TabItem("Single Model"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label="Input Sentence", placeholder="输入中文句子", lines=3)
                    algo_selector = gr.Dropdown([
                        "FMM", "BMM", "BiMM", "HMM", "RNN", "LSTM", "Transformer", "THULAC"
                    ], label="Select Algorithm", value="FMM")
                    btn = gr.Button("Segment", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(label="Segmentation Result", lines=3, interactive=False)
                    time_display = gr.Label(label="Inference Time")

            btn.click(fn=process_text, inputs=[input_text, algo_selector], outputs=[output_text, time_display])

            gr.Examples([
                ["研究生命起源"], ["结合成分子"], ["长春市长春节讲话"], ["乒乓球拍卖完了"]
            ], inputs=[input_text])

        # Tab 2: Compare All Models
        with gr.TabItem("Compare All Models"):
            with gr.Row():
                with gr.Column():
                    compare_input = gr.Textbox(label="Input Sentence", placeholder="输入中文句子", lines=3)
                    compare_btn = gr.Button("Compare All Models", variant="primary")
                    total_time_display = gr.Label(label="Total Inference Time")

            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Classical Methods**")
                    fmm_output = gr.Textbox(label="FMM (Forward Maximum Matching)", interactive=False)
                    bmm_output = gr.Textbox(label="BMM (Backward Maximum Matching)", interactive=False)
                    bimm_output = gr.Textbox(label="BiMM (Bidirectional Maximum Matching)", interactive=False)
                
                with gr.Column():
                    gr.Markdown("**Statistical & Deep Learning**")
                    hmm_output = gr.Textbox(label="HMM (Hidden Markov Model)", interactive=False)
                    rnn_output = gr.Textbox(label="RNN (Recurrent Neural Network)", interactive=False)
                    lstm_output = gr.Textbox(label="LSTM (Long Short-Term Memory)", interactive=False)
                    transformer_output = gr.Textbox(label="Transformer", interactive=False)
                    thulac_output = gr.Textbox(label="THULAC (THU Lexical Analyzer)", interactive=False)

            compare_btn.click(
                fn=process_all_models,
                inputs=[compare_input],
                outputs=[fmm_output, bmm_output, bimm_output, hmm_output, rnn_output, lstm_output, transformer_output, thulac_output, total_time_display]
            )

            gr.Examples([
                ["研究生命起源"], ["结合成分子"], ["长春市长春节讲话"], ["乒乓球拍卖完了"]
            ], inputs=[compare_input])

# Launch
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
