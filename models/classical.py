"""
Simple wrapper for classical Chinese word segmentation methods.
"""

import os


class FMM:
    """Forward Maximum Matching"""

    def __init__(self, dict_paths=None):
        self.vocab = set()
        self.max_len = 0
        if dict_paths:
            self.load_dict(dict_paths)
        else:
            # Default small dictionary for testing
            self.vocab = {'清华', '大学', '学生', '研究生', '国际'}
            self.max_len = 3

    def load_dict(self, paths):
        """Load dictionary from paths"""
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    for word in line.strip().split():
                        if word:
                            self.vocab.add(word)
                            self.max_len = max(self.max_len, len(word))

    def tokenize(self, text):
        """Segment text using forward maximum matching"""
        if not text:
            return []

        words = []
        i = 0
        n = len(text)

        while i < n:
            # Find longest word from current position
            j = min(i + self.max_len, n)
            while j > i:
                if text[i:j] in self.vocab:
                    words.append(text[i:j])
                    i = j
                    break
                j -= 1
            else:
                # Single character
                words.append(text[i])
                i += 1

        return words


class BMM:
    """Backward Maximum Matching"""

    def __init__(self, dict_paths=None):
        self.vocab = set()
        self.max_len = 0
        if dict_paths:
            self.load_dict(dict_paths)
        else:
            self.vocab = {'清华', '大学', '学生', '研究生', '国际'}
            self.max_len = 3

    def load_dict(self, paths):
        """Load dictionary from paths"""
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    for word in line.strip().split():
                        if word:
                            self.vocab.add(word)
                            self.max_len = max(self.max_len, len(word))

    def tokenize(self, text):
        """Segment text using backward maximum matching"""
        if not text:
            return []

        words = []
        i = len(text)

        while i > 0:
            # Find longest word ending at current position
            j = max(i - self.max_len, 0)
            while j < i:
                if text[j:i] in self.vocab:
                    words.insert(0, text[j:i])
                    i = j
                    break
                j += 1
            else:
                # Single character
                words.insert(0, text[i-1])
                i -= 1

        return words


class BiMM:
    """Bidirectional Maximum Matching"""

    def __init__(self, dict_paths=None):
        self.fmm = FMM(dict_paths)
        self.bmm = BMM(dict_paths)

    def tokenize(self, text):
        """Segment text using bidirectional maximum matching"""
        fmm_result = self.fmm.tokenize(text)
        bmm_result = self.bmm.tokenize(text)

        # Simple rule: choose the one with fewer words
        if len(fmm_result) < len(bmm_result):
            return fmm_result
        else:
            return bmm_result