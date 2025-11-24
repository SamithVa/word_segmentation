import numpy as np
import pickle
import os
from collections import defaultdict

class HMMTokenizer:
    def __init__(self, smoothing=1e-10):
        self.states = ['B', 'M', 'E', 'S']
        self.state_map = {s: i for i, s in enumerate(self.states)}
        self.smoothing = smoothing
        
        # Initialize matrices
        self.pi = np.zeros(4)       # Start probability
        self.A = np.zeros((4, 4))   # Transition probability (Previous -> Current)
        self.B = {}                 # Emission probability (State -> Char)
        self.trained = False

    def train(self, filepath):
        """
        Reads spaced training data (e.g., '今天 天气 不错') and updates counts.
        Supports single file or list of files.
        """
        # Support multiple files
        if isinstance(filepath, str):
            filepaths = [filepath]
        else:
            filepaths = filepath
        
        # 1. Count Frequencies
        for fp in filepaths:
            print(f"Processing {fp}...")
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split()
                    if not words: continue
                    
                    # Convert words to BMES tags
                    tags = []
                    chars = []
                    for word in words:
                        chars.extend(list(word))
                        if len(word) == 1:
                            tags.append('S')
                        else:
                            tags.append('B')
                            tags.extend(['M'] * (len(word) - 2))
                            tags.append('E')
                    
                    # Update Counts
                    for i, tag in enumerate(tags):
                        tag_idx = self.state_map[tag]
                        char = chars[i]
                        
                        # Emission Count
                        if char not in self.B: self.B[char] = np.zeros(4)
                        self.B[char][tag_idx] += 1
                        
                        # Start Count
                        if i == 0:
                            self.pi[tag_idx] += 1
                        # Transition Count
                        else:
                            prev_tag_idx = self.state_map[tags[i-1]]
                            self.A[prev_tag_idx, tag_idx] += 1
        
        print(f"Processed {len(filepaths)} file(s)")

        # 2. Normalize to Log Probabilities with Smoothing
        # Pi (Start probabilities)
        total_pi = np.sum(self.pi)
        if total_pi > 0:
            self.pi = np.log((self.pi + self.smoothing) / (total_pi + 4 * self.smoothing))
        else:
            self.pi = np.full(4, np.log(0.25))  # Uniform distribution
        
        # A (Transition probabilities)
        for i in range(4):
            total = np.sum(self.A[i])
            if total > 0:
                # Add-one smoothing
                self.A[i] = np.log((self.A[i] + self.smoothing) / (total + 4 * self.smoothing))
            else:
                # If no transitions observed from this state, uniform distribution
                self.A[i] = np.full(4, np.log(0.25))
        
        # B (Emission probabilities)
        state_counts = np.zeros(4)
        for char in self.B:
            state_counts += self.B[char]
            
        for char in self.B:
            # Add-one smoothing
            self.B[char] = np.log((self.B[char] + self.smoothing) / (state_counts + len(self.B) * self.smoothing))
        
        self.trained = True
        print(f"HMM Training Complete. Vocabulary size: {len(self.B)}")

    def viterbi(self, text):
            T = len(text)
            if T == 0: return []
            
            # dp[i][j] = max log probability of ending at step i with state j
            dp = np.full((T, 4), -float('inf'))
            # path[i][j] = which state at step i-1 led to state j at step i
            path = np.zeros((T, 4), dtype=int)

            # 1. Initialization (First character)
            first_char = text[0]
            # Handle unseen characters: assign a uniform low probability
            emission = self.B.get(first_char, np.full(4, -20.0)) 
            
            dp[0] = self.pi + emission

            # 2. Dynamic Programming
            for i in range(1, T):
                char = text[i]
                emission = self.B.get(char, np.full(4, -20.0)) # penalty for unseen chars
                
                for j in range(4): # Current state (0-3)
                    # Find best transition from previous states k -> current state j
                    # prob = dp[i-1][k] + A[k][j] + emission[j]
                    
                    # Vectorized operation for speed:
                    trans_probs = dp[i-1] + self.A[:, j]
                    best_prev_state = np.argmax(trans_probs)
                    
                    dp[i, j] = trans_probs[best_prev_state] + emission[j]
                    path[i, j] = best_prev_state

            # 3. Backtracking to find the best path
            best_path = [0] * T
            best_path[-1] = np.argmax(dp[T-1])
            
            for i in range(T-2, -1, -1):
                best_path[i] = path[i+1, best_path[i+1]]
                
            # 4. Convert path (0123) to Tags (BMES) and then to Words
            tags = [self.states[i] for i in best_path]
            return self.tags_to_words(text, tags)

    def tags_to_words(self, text, tags):
        words = []
        word = ""
        for char, tag in zip(text, tags):
            if tag == 'B':
                if word: words.append(word)
                word = char
            elif tag == 'M':
                word += char
            elif tag == 'E':
                word += char
                words.append(word)
                word = ""
            elif tag == 'S':
                if word: words.append(word)
                words.append(char)
                word = ""
        if word: words.append(word)
        return words

    def tokenize(self, text):
        if not self.trained:
            raise ValueError("Model must be trained before tokenization!")
        return self.viterbi(text)
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if not self.trained:
            raise ValueError("Model must be trained before saving!")
        
        model_data = {
            'states': self.states,
            'state_map': self.state_map,
            'pi': self.pi,
            'A': self.A,
            'B': self.B,
            'smoothing': self.smoothing,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found!")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.states = model_data['states']
        self.state_map = model_data['state_map']
        self.pi = model_data['pi']
        self.A = model_data['A']
        self.B = model_data['B']
        self.smoothing = model_data['smoothing']
        self.trained = model_data['trained']
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, gold_filepath, test_filepath, output_filepath=None):
        """Evaluate the model on test data and calculate precision, recall, F1."""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation!")
        
        # Read test data (unsegmented)
        with open(test_filepath, 'r', encoding='utf-8') as f:
            test_lines = [line.strip() for line in f if line.strip()]
        
        # Read gold standard (segmented)
        with open(gold_filepath, 'r', encoding='utf-8') as f:
            gold_lines = [line.strip() for line in f if line.strip()]
        
        if len(test_lines) != len(gold_lines):
            print(f"Warning: test lines ({len(test_lines)}) != gold lines ({len(gold_lines)})")
            min_len = min(len(test_lines), len(gold_lines))
            test_lines = test_lines[:min_len]
            gold_lines = gold_lines[:min_len]
        
        total_pred_words = 0
        total_gold_words = 0
        total_correct_words = 0
        
        predictions = []
        
        for test_line, gold_line in zip(test_lines, gold_lines):
            # Tokenize test sentence
            pred_words = self.tokenize(test_line)
            predictions.append(' '.join(pred_words))
            
            # Gold standard words
            gold_words = gold_line.split()
            
            # Calculate metrics
            total_pred_words += len(pred_words)
            total_gold_words += len(gold_words)
            
            # Count correct words (must match exactly)
            pred_set = []
            char_idx = 0
            for word in pred_words:
                pred_set.append((char_idx, char_idx + len(word), word))
                char_idx += len(word)
            
            gold_set = []
            char_idx = 0
            for word in gold_words:
                gold_set.append((char_idx, char_idx + len(word), word))
                char_idx += len(word)
            
            # Find intersection
            for pred_tuple in pred_set:
                if pred_tuple in gold_set:
                    total_correct_words += 1
        
        # Calculate metrics
        precision = total_correct_words / total_pred_words if total_pred_words > 0 else 0
        recall = total_correct_words / total_gold_words if total_gold_words > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n=== Evaluation Results ===")
        print(f"Total predicted words: {total_pred_words}")
        print(f"Total gold words: {total_gold_words}")
        print(f"Total correct words: {total_correct_words}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save predictions if output file specified
        if output_filepath:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(predictions))
            print(f"Predictions saved to {output_filepath}")
        
        return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    # Initialize HMM tokenizer
    hmm = HMMTokenizer(smoothing=1e-8)
    
    # Training with all available datasets
    print("Training HMM model with all datasets...")
    training_files = [
        '../icwb2-data/training/pku_training.utf8',
        '../icwb2-data/training/msr_training.utf8',
        '../icwb2-data/training/cityu_training.utf8',
        '../icwb2-data/training/as_training.utf8'
    ]
    # Filter only existing files
    import glob
    existing_files = [f for f in training_files if os.path.exists(f)]
    print(f"Found {len(existing_files)} training file(s)")
    
    hmm.train(existing_files)
    
    # Save model
    hmm.save_model('hmm_model.pkl')
    
    # Test on example sentences
    print("\n=== Testing on example sentences ===")
    test_sentences = [
        "今天天气不错",
        "中国人民站起来了",
        "机器学习是人工智能的重要分支",
        "自然语言处理技术发展迅速"
    ]
    
    for sentence in test_sentences:
        tokens = hmm.tokenize(sentence)
        print(f"{sentence} -> {' / '.join(tokens)}")
    
    # Evaluate on all test datasets
    print("\n=== Evaluating on all test datasets ===")
    test_datasets = [
        ('PKU', '../icwb2-data/gold/pku_test_gold.utf8', '../icwb2-data/testing/pku_test.utf8'),
        ('MSR', '../icwb2-data/gold/msr_test_gold.utf8', '../icwb2-data/testing/msr_test.utf8'),
        ('CITYU', '../icwb2-data/gold/cityu_test_gold.utf8', '../icwb2-data/testing/cityu_test.utf8'),
        ('AS', '../icwb2-data/gold/as_testing_gold.utf8', '../icwb2-data/testing/as_test.utf8')
    ]
    
    all_results = {}
    for name, gold_path, test_path in test_datasets:
        if os.path.exists(gold_path) and os.path.exists(test_path):
            print(f"\n--- Evaluating {name} ---")
            try:
                results = hmm.evaluate(
                    gold_filepath=gold_path,
                    test_filepath=test_path,
                    output_filepath=f'{name.lower()}_output.txt'
                )
                all_results[name] = results
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        else:
            print(f"Skipping {name}: files not found")
    
    # Print summary
    if all_results:
        print("\n=== Summary of All Datasets ===")
        print(f"{'Dataset':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 50)
        for name, results in all_results.items():
            print(f"{name:<10} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1']:<12.4f}")
        
        # Average scores
        avg_p = sum(r['precision'] for r in all_results.values()) / len(all_results)
        avg_r = sum(r['recall'] for r in all_results.values()) / len(all_results)
        avg_f1 = sum(r['f1'] for r in all_results.values()) / len(all_results)
        print("-" * 50)
        print(f"{'Average':<10} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f1:<12.4f}")
    
    # Demonstrate loading model
    print("\n=== Testing model loading ===")
    hmm_loaded = HMMTokenizer()
    hmm_loaded.load_model('hmm_model.pkl')
    test_text = "测试加载的模型"
    print(f"{test_text} -> {' / '.join(hmm_loaded.tokenize(test_text))}")