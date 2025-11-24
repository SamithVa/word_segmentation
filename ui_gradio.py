import gradio as gr
import time
import os
import torch
import pickle

# ==========================================
# 1. Your Algorithms (Wrapper)
# ==========================================
from fmm_bmm.forward_backward import MMTokenizer
from hmm.hmm import HMMTokenizer
from transformer_based.tranf import TransformerSegModel, TransformerTokenizer

class ModelWrapper:
    def __init__(self):
        print("Initializing models...")
        
        # 1. Initialize FMM/BMM/BiMM
        print("Loading FMM dictionary...")
        training_files = [
            'icwb2-data/training/pku_training.utf8',
            'icwb2-data/training/msr_training.utf8',
            'icwb2-data/training/cityu_training.utf8',
            'icwb2-data/training/as_training.utf8'
        ]
        existing_files = [f for f in training_files if os.path.exists(f)]
        self.mm_tokenizer = MMTokenizer(dictionary_path=existing_files)
        print(f"✓ FMM loaded with {len(self.mm_tokenizer.vocab)} words")
        
        # 2. Initialize HMM
        print("Loading HMM model...")
        self.hmm_tokenizer = HMMTokenizer(smoothing=1e-8)
        if os.path.exists('hmm/hmm_model.pkl'):
            self.hmm_tokenizer.load_model('hmm/hmm_model.pkl')
            print("✓ HMM model loaded from file")
        else:
            print("Training HMM model (this may take a minute)...")
            self.hmm_tokenizer.train(existing_files)
            self.hmm_tokenizer.save_model('hmm/hmm_model.pkl')
            print("✓ HMM model trained and saved")
        
        # 3. Initialize Transformer
        print("Loading Transformer model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load vocabulary
            with open('transformer_based/vocab.pkl', 'rb') as f:
                vocab_data = pickle.load(f)
            
            char2idx = vocab_data['char2idx']
            idx2tag = vocab_data['idx2tag']
            tag2idx = vocab_data['tag2idx']
            
            # Initialize model
            transformer_model = TransformerSegModel(
                vocab_size=len(char2idx),
                d_model=128,
                nhead=4,
                num_layers=2,
                num_classes=len(tag2idx),
                dropout=0.1
            ).to(self.device)
            
            # Load weights
            model_path = 'transformer_based/transformer_seg_best.pth'
            if not os.path.exists(model_path):
                model_path = 'transformer_based/transformer_seg_final.pth'
            
            checkpoint = torch.load(model_path, map_location=self.device)
            transformer_model.load_state_dict(checkpoint['model_state_dict'])
            transformer_model.eval()
            
            self.transformer_tokenizer = TransformerTokenizer(
                transformer_model, char2idx, idx2tag, self.device
            )
            print(f"✓ Transformer model loaded (device: {self.device})")
        except Exception as e:
            print(f"Warning: Could not load Transformer model: {e}")
            self.transformer_tokenizer = None
        
        print("All models initialized successfully!")

    def segment_fmm(self, text):
        """Forward Maximum Matching"""
        return self.mm_tokenizer.forward_maximum_matching(text)
    
    def segment_bmm(self, text):
        """Backward Maximum Matching"""
        return self.mm_tokenizer.backward_maximum_matching(text)
    
    def segment_bimm(self, text):
        """Bidirectional Maximum Matching"""
        return self.mm_tokenizer.bidirectional_maximum_matching(text)
    
    def segment_hmm(self, text):
        """HMM-based segmentation"""
        return self.hmm_tokenizer.tokenize(text)
    
    def segment_transformer(self, text):
        """Transformer-based segmentation"""
        if self.transformer_tokenizer is None:
            return ["Transformer", "model", "not", "loaded"]
        return self.transformer_tokenizer.tokenize(text)

    def segment(self, text, algorithm):
        """
        Main logic function called by Gradio
        """
        if not text or not text.strip():
            return "", "0 ms"
        
        start_time = time.time()
        
        # Dispatch to the correct algorithm
        if algorithm == "FMM (Forward Maximum Matching)":
            result = self.segment_fmm(text)
        elif algorithm == "BMM (Backward Maximum Matching)":
            result = self.segment_bmm(text)
        elif algorithm == "BiMM (Bidirectional Maximum Matching)":
            result = self.segment_bimm(text)
        elif algorithm == "HMM (Hidden Markov Model)":
            result = self.segment_hmm(text)
        elif algorithm == "Transformer (Deep Learning)":
            result = self.segment_transformer(text)
        else:
            result = ["Unknown", "algorithm"]
        
        # Formatting output
        output_str = " / ".join(result)
        cost_time = (time.time() - start_time) * 1000
        
        return output_str, f"{cost_time:.2f} ms"
    
    def segment_all(self, text):
        """
        Segment text with all three main algorithms and compare
        """
        if not text or not text.strip():
            return "", "", "", "0 ms", "0 ms", "0 ms"
        
        # FMM/BMM/BiMM
        start_time = time.time()
        fmm_result = self.segment_fmm(text)
        fmm_time = (time.time() - start_time) * 1000
        fmm_output = " / ".join(fmm_result)
        
        # HMM
        start_time = time.time()
        hmm_result = self.segment_hmm(text)
        hmm_time = (time.time() - start_time) * 1000
        hmm_output = " / ".join(hmm_result)
        
        # Transformer
        start_time = time.time()
        transformer_result = self.segment_transformer(text)
        transformer_time = (time.time() - start_time) * 1000
        transformer_output = " / ".join(transformer_result)
        
        return (
            fmm_output, 
            hmm_output, 
            transformer_output,
            f"{fmm_time:.2f} ms",
            f"{hmm_time:.2f} ms",
            f"{transformer_time:.2f} ms"
        )

# ==========================================
# 2. Gradio Interface
# ==========================================
print("Starting Gradio interface...")
model = ModelWrapper()

def process_text(text, algo_choice):
    if not text:
        return "", "0 ms"
    return model.segment(text, algo_choice)

def process_all(text):
    return model.segment_all(text)

# Define the UI
with gr.Blocks(title="Chinese Word Segmentation System") as demo:
    gr.Markdown("# 🔤 Chinese Word Segmentation System")
    gr.Markdown("Compare different word segmentation algorithms: **FMM**, **HMM**, and **Transformer**")
    
    with gr.Tab("🎯 Single Algorithm"):
        gr.Markdown("### Test individual algorithms")
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Sentence", 
                    placeholder="输入中文句子，例如：今天天气不错", 
                    lines=3
                )
                algo_selector = gr.Dropdown(
                    [
                        "FMM (Forward Maximum Matching)", 
                        "BMM (Backward Maximum Matching)",
                        "BiMM (Bidirectional Maximum Matching)",
                        "HMM (Hidden Markov Model)", 
                        "Transformer (Deep Learning)"
                    ], 
                    label="Select Algorithm", 
                    value="FMM (Forward Maximum Matching)"
                )
                btn = gr.Button("🚀 Segment", variant="primary", size="lg")
                
            with gr.Column():
                output_text = gr.Textbox(label="Segmentation Result", lines=3, interactive=False)
                time_display = gr.Label(label="⏱️ Inference Time")
        
        btn.click(fn=process_text, inputs=[input_text, algo_selector], outputs=[output_text, time_display])
        
        gr.Markdown("### Example Inputs")
        gr.Examples(
            examples=[
                ["今天天气不错"],
                ["中国人民站起来了"],
                ["机器学习是人工智能的重要分支"],
                ["自然语言处理技术发展迅速"],
                ["北京大学的研究生"],
            ],
            inputs=input_text
        )
    
    with gr.Tab("⚖️ Compare All Algorithms"):
        gr.Markdown("### Compare FMM, HMM, and Transformer side-by-side")
        
        input_text_compare = gr.Textbox(
            label="Input Sentence", 
            placeholder="输入中文句子进行对比，例如：今天天气不错", 
            lines=3
        )
        
        btn_compare = gr.Button("🔍 Compare All", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📊 FMM (Rule-based)")
                output_fmm = gr.Textbox(label="FMM Result", lines=3, interactive=False)
                time_fmm = gr.Label(label="Time")
                
            with gr.Column():
                gr.Markdown("#### 📈 HMM (Statistical)")
                output_hmm = gr.Textbox(label="HMM Result", lines=3, interactive=False)
                time_hmm = gr.Label(label="Time")
                
            with gr.Column():
                gr.Markdown("#### 🤖 Transformer (Deep Learning)")
                output_transformer = gr.Textbox(label="Transformer Result", lines=3, interactive=False)
                time_transformer = gr.Label(label="Time")
        
        btn_compare.click(
            fn=process_all, 
            inputs=[input_text_compare], 
            outputs=[output_fmm, output_hmm, output_transformer, time_fmm, time_hmm, time_transformer]
        )
        
        gr.Markdown("### Example Inputs")
        gr.Examples(
            examples=[
                ["今天天气不错"],
                ["中国人民站起来了"],
                ["机器学习是人工智能的重要分支"],
                ["自然语言处理技术发展迅速"],
                ["北京大学的研究生"],
                ["他说的确实在理"],
                ["这块地面积还真不小"],
            ],
            inputs=input_text_compare
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This System
        
        This Chinese Word Segmentation system implements three different approaches:
        
        ### 1. 📊 Maximum Matching (FMM/BMM/BiMM)
        - **Type**: Rule-based / Dictionary-based
        - **Method**: Greedy matching with vocabulary
        - **Pros**: Fast, simple, no training needed
        - **Cons**: Limited by dictionary, cannot handle unknown words well
        
        ### 2. 📈 Hidden Markov Model (HMM)
        - **Type**: Statistical machine learning
        - **Method**: BMES tagging with Viterbi algorithm
        - **Pros**: Handles unknown words, probabilistic framework
        - **Cons**: Limited by Markov assumption, moderate accuracy
        
        ### 3. 🤖 Transformer
        - **Type**: Deep learning (Neural Network)
        - **Method**: Self-attention mechanism with BMES tagging
        - **Pros**: Best accuracy, captures long-range dependencies
        - **Cons**: Requires training data and GPU, slower inference
        
        ---
        
        **Dataset**: SIGHAN Bakeoff 2005 (PKU, MSR, CITYU, AS)
        
        **Author**: NLP Course Project
        """)

# Launch
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)