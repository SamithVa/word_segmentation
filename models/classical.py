"""
Classical Chinese word segmentation methods.

Implements Forward Maximum Matching (FMM), Backward Maximum Matching (BMM),
and Bidirectional Maximum Matching (BiMM) algorithms.
"""

import os
from typing import List, Union, Set

from .base import BaseSegmenter


class BaseMaxMatch(BaseSegmenter):
    """Base class for maximum matching algorithms."""
    
    # Default dictionary for testing
    DEFAULT_VOCAB = {'清华', '大学', '学生', '研究生', '国际', '中国', '人民'}
    DEFAULT_MAX_LEN = 4

    def __init__(self, dict_paths: Union[str, List[str], None] = None):
        """
        Initialize the maximum matching model.
        
        Args:
            dict_paths: Path(s) to dictionary files. If None, uses default vocab.
        """
        super().__init__()
        self.vocab: Set[str] = set()
        self.max_len: int = 0
        
        if dict_paths:
            self.load_dict(dict_paths)
        else:
            self.vocab = self.DEFAULT_VOCAB.copy()
            self.max_len = self.DEFAULT_MAX_LEN

    def load_dict(self, paths: Union[str, List[str]]) -> None:
        """
        Load dictionary from file paths.
        
        Args:
            paths: Single path or list of paths to dictionary files.
        """
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

    def segment(self, text: str) -> List[str]:
        """Segment text into words. Must be implemented by subclasses."""
        raise NotImplementedError


class FMM(BaseMaxMatch):
    """Forward Maximum Matching (正向最大匹配)"""

    def segment(self, text: str) -> List[str]:
        """
        Segment text using forward maximum matching.
        
        Scans from left to right, always matching the longest possible word.
        
        Args:
            text: Input Chinese text
            
        Returns:
            List of segmented words
        """
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
                # Single character fallback
                words.append(text[i])
                i += 1

        return words


class BMM(BaseMaxMatch):
    """Backward Maximum Matching (逆向最大匹配)"""

    def segment(self, text: str) -> List[str]:
        """
        Segment text using backward maximum matching.
        
        Scans from right to left, always matching the longest possible word.
        
        Args:
            text: Input Chinese text
            
        Returns:
            List of segmented words
        """
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
                # Single character fallback
                words.insert(0, text[i-1])
                i -= 1

        return words


class BiMM(BaseSegmenter):
    """
    Bidirectional Maximum Matching (双向最大匹配)
    
    Combines FMM and BMM results using heuristic rules to select
    the better segmentation.
    """

    def __init__(self, dict_paths: Union[str, List[str], None] = None):
        """
        Initialize BiMM with shared dictionary.
        
        Args:
            dict_paths: Path(s) to dictionary files.
        """
        super().__init__()
        self.fmm = FMM(dict_paths)
        self.bmm = BMM(dict_paths)
        # Share vocabulary between FMM and BMM
        self.bmm.vocab = self.fmm.vocab
        self.bmm.max_len = self.fmm.max_len

    @property
    def vocab(self) -> Set[str]:
        """Access vocabulary through FMM."""
        return self.fmm.vocab

    @staticmethod
    def _count_single_chars(words: List[str]) -> int:
        """Count single-character words in segmentation result."""
        return sum(1 for w in words if len(w) == 1)

    @staticmethod
    def _variance_word_length(words: List[str]) -> float:
        """Calculate variance of word lengths (lower is better)."""
        if not words:
            return 0.0
        lengths = [len(w) for w in words]
        mean = sum(lengths) / len(lengths)
        return sum((l - mean) ** 2 for l in lengths) / len(lengths)

    def segment(self, text: str) -> List[str]:
        """
        Segment text using bidirectional maximum matching.
        
        Selection rules (in order of priority):
        1. Fewer total words
        2. Fewer single-character words
        3. Lower variance in word lengths
        4. Default to FMM result
        
        Args:
            text: Input Chinese text
            
        Returns:
            List of segmented words
        """
        fmm_result = self.fmm.segment(text)
        bmm_result = self.bmm.segment(text)
        
        # If results are identical, return either
        if fmm_result == bmm_result:
            return fmm_result
        
        # Rule 1: Fewer total words
        if len(fmm_result) != len(bmm_result):
            return fmm_result if len(fmm_result) < len(bmm_result) else bmm_result
        
        # Rule 2: Fewer single-character words
        fmm_singles = self._count_single_chars(fmm_result)
        bmm_singles = self._count_single_chars(bmm_result)
        if fmm_singles != bmm_singles:
            return fmm_result if fmm_singles < bmm_singles else bmm_result
        
        # Rule 3: Lower variance in word lengths
        fmm_var = self._variance_word_length(fmm_result)
        bmm_var = self._variance_word_length(bmm_result)
        if abs(fmm_var - bmm_var) > 0.01:
            return fmm_result if fmm_var < bmm_var else bmm_result
        
        # Default: return FMM result
        return fmm_result
