import numpy as np

class Embeddings:
    def __init__(self):
        # Create a custom mapping of words to vectors
        # For simplicity, assign random vectors to each word
        self.token_to_vector = {}
        self.vector_dim = 100  # Dimension of embedding vectors

    def _tokenize(self, text):
        """
        Simple tokenizer to split text into words.
        """
        return text.lower().split()

    def _vectorize_token(self, token):
        """
        Assign a random vector to a token if it doesn't already exist.
        """
        if token not in self.token_to_vector:
            self.token_to_vector[token] = np.random.rand(self.vector_dim)
        return self.token_to_vector[token]

    def embed(self, text):
        """
        Generate an embedding for a piece of text by averaging token vectors.

        Args:
            text (str): The input text to embed.

        Returns:
            np.array: The embedding vector for the input text.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.vector_dim)  # Return a zero vector for empty text
        token_vectors = [self._vectorize_token(token) for token in tokens]
        return np.mean(token_vectors, axis=0)

class CustomEmbeddings:
    def __init__(self):
        self.embedding_model = Embeddings()

    def embed_documents(self, texts):
        """
        Embed a list of documents using custom logic.
        Convert the embeddings to Python lists.
        """
        return [self.embedding_model.embed(text).tolist() for text in texts]

    def embed_query(self, text):
        """
        Embed a single query using custom logic.
        Convert the embedding to a Python list.
        """
        return self.embedding_model.embed(text).tolist()

