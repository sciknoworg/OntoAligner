import torch
import struct
from pathlib import Path


class WordEmbedding:
    """
    Generic word embedding loader supporting:
    - GloVe
    - Word2Vec (binary & text)
    - FastText (text)

    Example Usage:
    >>> embedding = WordEmbedding(emb_path = "/path/to/embedding.bin",
    >>>                           fmt='word2vec',
    >>>                           device = "cpu")
    >>> print(embedding.sim("bank", "finance"))
    """
    def __init__(self, path, fmt=None, device="cpu", lowercase=False):
        """
        Parameters
        ----------
        path : str
            Path to embedding file
        fmt : str | None
            "glove", "word2vec", "fasttext", "bin"
            If None, inferred from extension
        device : str
            cpu or cuda
        lowercase : bool
            Normalize words
        """
        self.path = Path(path)
        self.device = device
        self.lowercase = lowercase
        self.word2idx = {}
        self.embeddings = None
        self.dim = None
        self.fmt = fmt or self._infer_format()
        self._load()

    def get(self, word):
        if self.lowercase:
            word = word.lower()

        idx = self.word2idx.get(word)
        if idx is None:
            return torch.zeros(self.dim, device=self.device)
        return self.embeddings[idx]

    def sim(self, w1, w2):
        return torch.cosine_similarity(self.get(w1).unsqueeze(0),
                                       self.get(w2).unsqueeze(0)).item()

    def _infer_format(self):
        if self.path.suffix == ".bin":
            return "word2vec-bin"
        return "text"

    def _load(self):
        if self.fmt == "glove":
            self._load_text(skip_header=False)
        elif self.fmt == "word2vec":
            self._load_text(skip_header=True)
        elif self.fmt == "fasttext":
            self._load_text(skip_header=True)
        elif self.fmt == "word2vec-bin":
            self._load_word2vec_bin()
        else:
            raise ValueError(f"Unsupported format: {self.fmt}")

    def _load_text(self, skip_header):
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            if skip_header:
                first = f.readline().split()
                self.dim = len(first) - 1
            vectors = []
            for line in f:
                parts = line.rstrip().split()
                word, values = parts[0], parts[1:]
                if self.lowercase:
                    word = word.lower()
                if self.dim is None:
                    self.dim = len(values)
                vec = torch.tensor([float(x) for x in values],
                                   dtype=torch.float32,
                                   device=self.device)
                self.word2idx[word] = len(vectors)
                vectors.append(vec)
        self.embeddings = torch.stack(vectors)

    def _load_word2vec_bin(self):
        with open(self.path, "rb") as f:
            vocab_size, self.dim = map(int, f.readline().split())
            self.embeddings = torch.empty(vocab_size, self.dim, device=self.device)
            for i in range(vocab_size):
                word = []
                while True:
                    c = f.read(1)
                    if c == b" ":
                        break
                    word.append(c)
                word = b"".join(word).decode("utf-8", errors="ignore")
                vec = struct.unpack(f"<{self.dim}f", f.read(self.dim * 4))
                if self.lowercase:
                    word = word.lower()
                self.word2idx[word] = i
                self.embeddings[i] = torch.tensor(vec, device=self.device)
