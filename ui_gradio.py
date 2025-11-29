import gradio as gr
import time
import os
import torch
import pickle

# Import models
from models.hmm_seg import HMMSeg
from models.classical import FMM, BMM, BiMM
from models.rnn_seg import RNNSeg
from models.lstm_seg import LSTMSeg
from models.transformer_seg import TransformerSeg

# Import utility
from utils import BaseTokenizer

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

        # Initialize classical models (no training needed - just load dictionary)
        self._init_classical_models()

        # Initialize HMM model
        self._init_hmm_model()

        # Deep learning models (lazy loading)
        self.transformer_tokenizer = None
        self.rnn_tokenizer = None
        self.lstm_tokenizer = None

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

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
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
            'outputs/saved_models/transformer/transformer_seg_best.pth',
            'outputs/saved_models/transformer/vocab.pkl',
            TransformerSeg,
            {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "max_len": 256
            }
        )

    def _load_rnn_model(self):
        return self._load_model(
            "RNN",
            'outputs/saved_models/rnn/rnn_seg_best.pth',
            'outputs/saved_models/rnn/vocab.pkl',
            RNNSeg,
            {
                "d_model": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.3
            }
        )

    def _load_lstm_model(self):
        return self._load_model(
            "LSTM",
            'outputs/saved_models/lstm/bilstm_seg_best.pth',
            'outputs/saved_models/lstm/vocab.pkl',
            LSTMSeg,
            {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.3
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

# Build UI
with gr.Blocks(title="Chinese Word Segmentation System") as demo:
    gr.Markdown("# 🔤 Chinese Word Segmentation System")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Sentence", placeholder="输入中文句子", lines=3)
            algo_selector = gr.Dropdown([
                "FMM", "BMM", "BiMM", "HMM", "RNN", "LSTM", "Transformer"
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