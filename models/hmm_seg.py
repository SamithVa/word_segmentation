import numpy as np
import pickle
import os
import re

class HMMSeg:
    def __init__(self, smoothing=1e-3, use_preprocess=True):
        self.states = ['B', 'I', 'E', 'S']
        self.state_map = {s: i for i, s in enumerate(self.states)}
        self.smoothing = smoothing
        self.use_preprocess = use_preprocess
        self.NUM_TOKEN = '<NUM>'
        self.ENG_TOKEN = '<ENG>'

        self.pi = np.zeros(4)
        self.A = np.zeros((4, 4))
        self.B = {}
        self.trained = False

    def _preprocess(self, text):
        """Handle numbers and English in text."""
        if not self.use_preprocess:
            return list(text), list(text)

        tokens, originals = [], []
        pattern = r'[0-9]+(?:\.[0-9]+)?|[a-zA-Z]+'
        last_end = 0

        for match in re.finditer(pattern, text):
            start, end = match.span()
            tokens.extend(text[last_end:start])
            originals.extend(text[last_end:start])

            token = self.NUM_TOKEN if match.group()[0].isdigit() else self.ENG_TOKEN
            tokens.append(token)
            originals.append(match.group())
            last_end = end

        tokens.extend(text[last_end:])
        originals.extend(text[last_end:])
        return tokens, originals

    def _get_bies_tags(self, word_tokens):
        """Generate BIES tags for word tokens."""
        return ['S'] if len(word_tokens) == 1 else ['B'] + ['I'] * (len(word_tokens) - 2) + ['E']

    def _apply_constraints(self):
        """Apply BIES transition constraints."""
        self.pi[self.state_map['I']] = -float('inf')
        self.pi[self.state_map['E']] = -float('inf')

        # Invalid transitions
        invalid = [('B', 'B'), ('B', 'S'), ('I', 'B'), ('I', 'S'),
                  ('E', 'I'), ('E', 'E'), ('S', 'I'), ('S', 'E')]
        for from_state, to_state in invalid:
            self.A[self.state_map[from_state], self.state_map[to_state]] = -float('inf')

    def train(self, filepaths):
        """Train HMM model."""
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        print("Training HMM...")
        for filepath in filepaths:
            print(f"Processing {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split()
                    if not words:
                        continue

                    tokens, tags = [], []
                    for word in words:
                        word_tokens = self._preprocess(word)[0]
                        tokens.extend(word_tokens)
                        tags.extend(self._get_bies_tags(word_tokens))

                    # Update counts
                    for i, (token, tag) in enumerate(zip(tokens, tags)):
                        tag_idx = self.state_map[tag]

                        if token not in self.B:
                            self.B[token] = np.zeros(4)
                        self.B[token][tag_idx] += 1

                        if i == 0:
                            self.pi[tag_idx] += 1
                        else:
                            prev_idx = self.state_map[tags[i-1]]
                            self.A[prev_idx, tag_idx] += 1

        self._normalize()
        self.trained = True
        print(f"Training complete. Vocabulary size: {len(self.B)}")

    def _normalize(self):
        """Convert counts to log probabilities."""
        # Normalize start probabilities
        total = np.sum(self.pi)
        if total > 0:
            self.pi = np.log((self.pi + self.smoothing) / (total + 4 * self.smoothing))
        else:
            self.pi = np.full(4, np.log(0.25))

        # Normalize transition probabilities
        for i in range(4):
            total = np.sum(self.A[i])
            if total > 0:
                self.A[i] = np.log((self.A[i] + self.smoothing) / (total + 4 * self.smoothing))
            else:
                self.A[i] = np.full(4, np.log(0.25))

        # Apply constraints
        self._apply_constraints()

        # Normalize emission probabilities
        state_totals = np.zeros(4)
        for char in self.B:
            state_totals += self.B[char]

        self.vocab_size = len(self.B)
        for char in self.B:
            self.B[char] = np.log((self.B[char] + self.smoothing) /
                                  (state_totals + self.vocab_size * self.smoothing))

        # Default for unseen characters
        self.unseen_emission = np.array([-10.0, -50.0, -50.0, -2.0])

    def _viterbi(self, tokens):
        """Find most likely state sequence."""
        T = len(tokens)
        if T == 0:
            return []

        dp = np.full((T, 4), -float('inf'))
        path = np.zeros((T, 4), dtype=int)

        # Initialize
        dp[0] = self.pi + self.B.get(tokens[0], self.unseen_emission)

        # DP
        for i in range(1, T):
            emission = self.B.get(tokens[i], self.unseen_emission)
            scores = dp[i-1][:, None] + self.A
            dp[i] = np.max(scores, axis=0) + emission
            path[i] = np.argmax(scores, axis=0)

        # Terminate (only E or S can end)
        final = dp[T-1].copy()
        final[self.state_map['B']] = -float('inf')
        final[self.state_map['I']] = -float('inf')

        # Backtrack
        best = [np.argmax(final)]
        for i in range(T-2, -1, -1):
            best.append(path[i+1, best[-1]])

        return [self.states[i] for i in reversed(best)]

    def _tags_to_words(self, originals, tags):
        """Convert tagged sequence to words."""
        words = []
        word = ""

        for original, tag in zip(originals, tags):
            if tag == 'B':
                if word:
                    words.append(word)
                word = original
            elif tag in ['I', 'E']:
                word += original
                if tag == 'E':
                    words.append(word)
                    word = ""
            else:  # S
                if word:
                    words.append(word)
                words.append(original)
                word = ""

        if word:
            words.append(word)
        return words

    def tokenize(self, text):
        """Segment Chinese text."""
        if not self.trained:
            raise ValueError("Model must be trained first!")

        tokens, originals = self._preprocess(text)
        if not tokens:
            return []

        tags = self._viterbi(tokens)
        return self._tags_to_words(originals, tags)

    def save_model(self, filepath):
        """Save model to file."""
        if not self.trained:
            raise ValueError("Model must be trained before saving!")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'states': self.states, 'state_map': self.state_map,
            'pi': self.pi, 'A': self.A, 'B': self.B,
            'smoothing': self.smoothing, 'trained': self.trained,
            'vocab_size': self.vocab_size, 'unseen_emission': self.unseen_emission,
            'use_preprocess': self.use_preprocess,
            'NUM_TOKEN': self.NUM_TOKEN, 'ENG_TOKEN': self.ENG_TOKEN
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found!")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        for key in ['states', 'state_map', 'pi', 'A', 'B', 'smoothing', 'trained']:
            setattr(self, key, data[key])

        self.vocab_size = data.get('vocab_size', len(self.B))
        self.unseen_emission = data.get('unseen_emission', np.full(4, -20.0))
        self.use_preprocess = data.get('use_preprocess', True)
        self.NUM_TOKEN = data.get('NUM_TOKEN', '<NUM>')
        self.ENG_TOKEN = data.get('ENG_TOKEN', '<ENG>')

        print(f"Model loaded from {filepath}")