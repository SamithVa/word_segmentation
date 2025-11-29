"""CRF model for Chinese Word Segmentation"""

import os
import pickle
from tqdm import tqdm

try:
    import sklearn_crfsuite
except ImportError:
    raise ImportError("Please install sklearn-crfsuite: pip install sklearn-crfsuite")


class CRFSeg:
    """Conditional Random Field segmenter using BMES tagging"""

    def __init__(self):
        self.crf = None
        self.trained = False

    def _word_to_tags(self, word):
        """Convert word to BMES tags"""
        return ['S'] if len(word) == 1 else ['B'] + ['M'] * (len(word) - 2) + ['E']

    def _text_to_features(self, text):
        """Extract features for CRF"""
        features = []
        for i, char in enumerate(text):
            feature = {
                'char': char,
                'bias': 1.0,
                'is_digit': char.isdigit(),
                'is_punct': char in '，。！？、；：""''（）【】《》'
            }

            # Add context features
            if i > 0:
                feature['prev_char'] = text[i-1]
                feature['bigram_-1'] = text[i-1] + char
            if i > 1:
                feature['prev2_char'] = text[i-2]
                feature['trigram_-2'] = text[i-2] + text[i-1] + char

            if i < len(text) - 1:
                feature['next_char'] = text[i+1]
                feature['bigram_+1'] = char + text[i+1]
            if i < len(text) - 2:
                feature['next2_char'] = text[i+2]
                feature['trigram_+2'] = char + text[i+1] + text[i+2]

            features.append(feature)
        return features

    def _prepare_data(self, filepath):
        """Prepare training data from file"""
        X, y = [], []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                words = line.strip().split()
                if not words:
                    continue

                chars = []
                tags = []
                for word in words:
                    chars.extend(list(word))
                    tags.extend(self._word_to_tags(word))

                X.append(self._text_to_features(chars))
                y.append(tags)

        return X, y

    def train(self, filepath, c1=0.1, c2=0.1, max_iter=100):
        """Train CRF model"""
        print(f"Training CRF on {filepath}...")

        X, y = self._prepare_data(filepath)

        # Initialize and train CRF
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iter,
            all_possible_transitions=True
        )

        self.crf.fit(X, y)
        self.trained = True
        print("Training complete!")

    def _tags_to_words(self, chars, tags):
        """Convert BMES tags back to words"""
        words = []
        word = ""

        for char, tag in zip(chars, tags):
            if tag == 'B':
                if word:
                    words.append(word)
                word = char
            elif tag == 'M':
                word += char
            elif tag == 'E':
                word += char
                words.append(word)
                word = ""
            elif tag == 'S':
                if word:
                    words.append(word)
                words.append(char)
                word = ""

        if word:
            words.append(word)
        return words

    def tokenize(self, text):
        """Segment text into words"""
        if not self.trained:
            raise ValueError("Model must be trained first!")

        if not text:
            return []

        chars = list(text)
        features = self._text_to_features(chars)
        tags = self.crf.predict([features])[0]

        return self._tags_to_words(chars, tags)

    def save(self, filepath):
        """Save trained model"""
        if not self.trained:
            raise ValueError("Model must be trained before saving!")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.crf, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found!")

        with open(filepath, 'rb') as f:
            self.crf = pickle.load(f)
        self.trained = True
        print(f"Model loaded from {filepath}")