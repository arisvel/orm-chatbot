from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, text, normal):
        return self.model.encode(text, show_progress_bar=False, normalize_embeddings=normal)


def generate_embedding(text, normal=True, model_name='xlm-r-bert-base-nli-stsb-mean-tokens'):
    service = EmbeddingService(model_name)
    return service.encode(text, normal)


