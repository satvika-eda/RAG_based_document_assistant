import numpy as np
import pickle

# Custom Embeddings model to load the saved mdoel
class CustomEmbeddings:
    def __init__(self, embedding_path="custom_embeddings.pkl"):
        # Load pre-trained embeddings
        with open(embedding_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def _embed_text(self, text):
        """
        Generate an embedding for a given text by averaging embeddings of words in the text.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Text embedding as a list.
        """
        tokens = text.lower().split()
        word_embeddings = [self.embeddings[token] for token in tokens if token in self.embeddings]
        if not word_embeddings:  # Handle out-of-vocabulary case
            return np.zeros(len(next(iter(self.embeddings.values())))).tolist()
        return np.mean(word_embeddings, axis=0).tolist()

    def embed_documents(self, texts):
        """
        Embed a list of documents using custom embeddings.

        Args:
            texts (List[str]): List of documents to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text):
        """
        Embed a single query using custom embeddings.

        Args:
            text (str): Query text.

        Returns:
            List[float]: Query embedding.
        """
        return self._embed_text(text)
