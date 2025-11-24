class MMTokenizer:
    def __init__(self, dictionary_path=None):
        self.vocab = set()
        self.max_word_len = 0
        
        # Load dictionary if provided
        if dictionary_path:
            if isinstance(dictionary_path, str):
                self.load_dictionary(dictionary_path)
            else:
                # Load multiple dictionaries
                self.load_multiple_dictionaries(dictionary_path)
        else:
            # Minimal example dictionary for testing
            self.vocab = {'北京', '大学', '生活', '学生', '研究生', '生命'}
            self.max_word_len = 3

    def load_dictionary(self, path):
        """
        Load words from the training data provided in SIGHAN dataset
        """
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming the training data is space-separated like: "我 爱 北京"
                words = line.strip().split()
                for w in words:
                    self.vocab.add(w)
                    if len(w) > self.max_word_len:
                        self.max_word_len = len(w)
        print(f"Loaded {path}: {len(self.vocab)} words")
    
    def load_multiple_dictionaries(self, paths):
        """
        Load words from multiple training files
        """
        for path in paths:
            try:
                self.load_dictionary(path)
            except FileNotFoundError:
                print(f"Warning: {path} not found, skipping...")
        print(f"Total vocabulary: {len(self.vocab)} words, Max length: {self.max_word_len}")

    def forward_maximum_matching(self, text):
        """
        Forward Maximum Matching (FMM) Algorithm
        """
        result = []
        index = 0
        text_len = len(text)

        while index < text_len:
            matched = False
            # Try to match the longest possible word first
            # Limit the window size to the remaining text length or max_word_len
            current_window_size = min(self.max_word_len, text_len - index)
            
            while current_window_size > 0:
                # Slice the text: text[start : start + window]
                sub_string = text[index : index + current_window_size]
                
                if sub_string in self.vocab:
                    result.append(sub_string)
                    index += current_window_size # Move pointer forward
                    matched = True
                    break
                else:
                    # If no match, shrink window from the right
                    current_window_size -= 1
            
            # If window size becomes 0 (no match found), treat single char as a word
            if not matched:
                result.append(text[index])
                index += 1
                
        return result

    def backward_maximum_matching(self, text):
        """
        Backward Maximum Matching (BMM) Algorithm
        """
        result = []
        index = len(text)

        while index > 0:
            matched = False
            # Try to match the longest possible word first
            current_window_size = min(self.max_word_len, index)
            
            while current_window_size > 0:
                start_index = index - current_window_size
                sub_string = text[start_index : index]
                
                if sub_string in self.vocab:
                    result.insert(0, sub_string)  # Insert at the beginning
                    index -= current_window_size # Move pointer backward
                    matched = True
                    break
                else:
                    # If no match, shrink window from the left
                    current_window_size -= 1
            
            # If window size becomes 0 (no match found), treat single char as a word
            if not matched:
                result.insert(0, text[index - 1])
                index -= 1
                
        return result
    
    def bidirectional_maximum_matching(self, text):
        """
        Bidirectional Maximum Matching (BiMM) Algorithm
        Compare FMM and BMM results, choose the better one based on:
        1. Fewer segments (longer words preferred)
        2. Fewer single-character words
        3. If still tied, prefer FMM
        """
        fmm_result = self.forward_maximum_matching(text)
        bmm_result = self.backward_maximum_matching(text)
        
        # If results are the same, return either
        if fmm_result == bmm_result:
            return fmm_result
        
        # Count single-character words
        fmm_singles = sum(1 for word in fmm_result if len(word) == 1)
        bmm_singles = sum(1 for word in bmm_result if len(word) == 1)
        
        # Rule 1: Fewer segments
        if len(fmm_result) < len(bmm_result):
            return fmm_result
        elif len(bmm_result) < len(fmm_result):
            return bmm_result
        
        # Rule 2: Fewer single-character words
        if fmm_singles < bmm_singles:
            return fmm_result
        elif bmm_singles < fmm_singles:
            return bmm_result
        
        # Rule 3: Default to FMM
        return fmm_result
    
    def evaluate(self, gold_filepath, test_filepath, method='fmm'):
        """
        Evaluate the tokenizer on test data
        method: 'fmm', 'bmm', or 'bimm'
        """
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
        
        # Choose segmentation method
        if method == 'fmm':
            segment_func = self.forward_maximum_matching
        elif method == 'bmm':
            segment_func = self.backward_maximum_matching
        elif method == 'bimm':
            segment_func = self.bidirectional_maximum_matching
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for test_line, gold_line in zip(test_lines, gold_lines):
            # Tokenize test sentence
            pred_words = segment_func(test_line)
            
            # Gold standard words
            gold_words = gold_line.split()
            
            # Calculate metrics
            total_pred_words += len(pred_words)
            total_gold_words += len(gold_words)
            
            # Count correct words (must match exactly at same position)
            pred_positions = []
            char_idx = 0
            for word in pred_words:
                pred_positions.append((char_idx, char_idx + len(word), word))
                char_idx += len(word)
            
            gold_positions = []
            char_idx = 0
            for word in gold_words:
                gold_positions.append((char_idx, char_idx + len(word), word))
                char_idx += len(word)
            
            # Find intersection
            for pred_tuple in pred_positions:
                if pred_tuple in gold_positions:
                    total_correct_words += 1
        
        # Calculate metrics
        precision = total_correct_words / total_pred_words if total_pred_words > 0 else 0
        recall = total_correct_words / total_gold_words if total_gold_words > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_pred': total_pred_words,
            'total_gold': total_gold_words,
            'total_correct': total_correct_words
        }

# --- Usage Example ---
if __name__ == "__main__":
    import os
    
    # Load all training datasets
    print("="*60)
    print("Loading all training datasets...")
    print("="*60)
    
    training_files = [
        '../icwb2-data/training/pku_training.utf8',
        '../icwb2-data/training/msr_training.utf8',
        '../icwb2-data/training/cityu_training.utf8',
        '../icwb2-data/training/as_training.utf8'
    ]
    
    # Filter only existing files
    existing_files = [f for f in training_files if os.path.exists(f)]
    print(f"Found {len(existing_files)} training file(s)\n")
    
    tokenizer = MMTokenizer(dictionary_path=existing_files)
    
    # Test on example sentences
    print("\n" + "="*60)
    print("Testing on example sentences")
    print("="*60)
    
    test_sentences = [
        "北京大学的研究生",
        "今天天气不错",
        "中国人民站起来了",
        "机器学习是人工智能的重要分支",
        "自然语言处理技术发展迅速"
    ]
    
    for sentence in test_sentences:
        fmm_result = tokenizer.forward_maximum_matching(sentence)
        bmm_result = tokenizer.backward_maximum_matching(sentence)
        bimm_result = tokenizer.bidirectional_maximum_matching(sentence)
        
        print(f"\nInput: {sentence}")
        print(f"FMM:   {' / '.join(fmm_result)}")
        print(f"BMM:   {' / '.join(bmm_result)}")
        print(f"BiMM:  {' / '.join(bimm_result)}")
    
    # Evaluate on all test datasets
    print("\n" + "="*60)
    print("Evaluating on test datasets")
    print("="*60)
    
    test_datasets = [
        ('PKU', '../icwb2-data/gold/pku_test_gold.utf8', '../icwb2-data/testing/pku_test.utf8'),
        ('MSR', '../icwb2-data/gold/msr_test_gold.utf8', '../icwb2-data/testing/msr_test.utf8'),
        ('CITYU', '../icwb2-data/gold/cityu_test_gold.utf8', '../icwb2-data/testing/cityu_test.utf8'),
        ('AS', '../icwb2-data/gold/as_testing_gold.utf8', '../icwb2-data/testing/as_test.utf8')
    ]
    
    methods = ['fmm', 'bmm', 'bimm']
    all_results = {method: {} for method in methods}
    
    for method in methods:
        print(f"\n--- Evaluating {method.upper()} ---")
        for name, gold_path, test_path in test_datasets:
            if os.path.exists(gold_path) and os.path.exists(test_path):
                try:
                    results = tokenizer.evaluate(gold_path, test_path, method=method)
                    all_results[method][name] = results
                    print(f"{name}: P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1']:.4f}")
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
            else:
                print(f"Skipping {name}: files not found")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("Summary - Comparison of All Methods")
    print("="*60)
    
    for method in methods:
        if all_results[method]:
            print(f"\n{method.upper()} Results:")
            print(f"{'Dataset':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
            print("-" * 50)
            
            for name, results in all_results[method].items():
                print(f"{name:<10} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1']:<12.4f}")
            
            # Average
            avg_p = sum(r['precision'] for r in all_results[method].values()) / len(all_results[method])
            avg_r = sum(r['recall'] for r in all_results[method].values()) / len(all_results[method])
            avg_f1 = sum(r['f1'] for r in all_results[method].values()) / len(all_results[method])
            print("-" * 50)
            print(f"{'Average':<10} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f1:<12.4f}")
    
    # Final comparison
    print("\n" + "="*60)
    print("Method Comparison (Average F1 Score)")
    print("="*60)
    for method in methods:
        if all_results[method]:
            avg_f1 = sum(r['f1'] for r in all_results[method].values()) / len(all_results[method])
            print(f"{method.upper():<10} F1={avg_f1:.4f}")