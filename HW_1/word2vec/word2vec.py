from torch import nn
import torch


class Word2vecSGNegSamplingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.activation_function = nn.LogSigmoid()

    def forward(self, target, context, negative_samples):
        target_embedding = self.embeddings(target)
        context_embedding = self.context_embeddings(context)
        negative_embeddings = self.context_embeddings(negative_samples)

        positive_score = self.activation_function(torch.sum(target_embedding * context_embedding, dim=1))
        negative_score = self.activation_function(
            -torch.bmm(negative_embeddings, target_embedding.unsqueeze(2)).squeeze(2)
        ).sum(1)

        loss = -(positive_score + negative_score).mean()
        return loss
