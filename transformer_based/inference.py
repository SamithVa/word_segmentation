import torch
import pickle
from tranf import TransformerSegModel, TransformerTokenizer

def load_model(model_path='transformer_seg_best.pth', vocab_path='vocab.pkl', device=None):
    """Load trained model and vocabulary."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    char2idx = vocab_data['char2idx']
    idx2tag = vocab_data['idx2tag']
    tag2idx = vocab_data['tag2idx']
    
    # Initialize model
    model = TransformerSegModel(
        vocab_size=len(char2idx),
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=len(tag2idx),
        dropout=0.1
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Device: {device}")
    
    return model, char2idx, idx2tag, device

def main():
    # Load model
    model, char2idx, idx2tag, device = load_model()
    
    # Create tokenizer
    tokenizer = TransformerTokenizer(model, char2idx, idx2tag, device)
    
    print("\n" + "="*60)
    print("Transformer Chinese Word Segmentation - Interactive Mode")
    print("="*60)
    print("Enter Chinese text to segment (or 'quit' to exit)")
    print()
    
    while True:
        try:
            text = input("Input: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            # Tokenize
            words = tokenizer.tokenize(text)
            print(f"Output: {' / '.join(words)}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
