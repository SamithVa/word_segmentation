"""Simple base class for tokenizers (optional)"""

class BaseTokenizer:
    """Optional base class for all tokenizers"""

    def __init__(self):
        self.trained = False
        self.model_name = self.__class__.__name__

    def train(self, data):
        """Train the model"""
        raise NotImplementedError

    def tokenize(self, text):
        """Segment text into words"""
        raise NotImplementedError

    def save(self, path):
        """Save model to file"""
        print(f"Saving {self.model_name} to {path}")

    def load(self, path):
        """Load model from file"""
        print(f"Loading {self.model_name} from {path}")