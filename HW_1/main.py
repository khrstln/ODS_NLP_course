import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from gensim.models.word2vec import Word2Vec

from word2vec.negative_sampling import get_negative_samples
from common.clean import clean
from HW_1.data.prepare import prepare_data_skip_gram
from data.skipgram_dataset import SkipGramDataset
from word2vec.word2vec import Word2vecSGNegSamplingModel


def train_sg_ns_model(
    data, word_to_ix, vocab_size, embedding_dim=50, epochs=10, batch_size=1, lr=0.05, num_negative_samples=5
):
    dataset = SkipGramDataset(data, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Word2vecSGNegSamplingModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for target, context in dataloader:
            target = target.long()
            context = context.long()
            negative_samples = torch.LongTensor(
                [get_negative_samples(t.item(), num_negative_samples, vocab_size) for t in target]
            )

            optimizer.zero_grad()
            loss = model(target, context, negative_samples)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    return model


def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    window_size = 10
    embedding_dim = 30
    epochs = 25
    batch_size = 32
    num_negative_samples = 5

    ngramm_data, word_to_ix, vocab_size = prepare_data_skip_gram(data, window_size)
    model = train_sg_ns_model(
        ngramm_data,
        word_to_ix,
        vocab_size,
        embedding_dim,
        epochs,
        batch_size,
        num_negative_samples=num_negative_samples,
    )

    embeddings = model.embeddings.weight.data.numpy()
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    w2v_dict = {ix_to_word[ix]: embeddings[ix] for ix in range(vocab_size)}
    return w2v_dict


if __name__ == "__main__":
    test_text = "Captures Semantic Relationships: The skip-gram model effectively captures semantic relationships \
            between words. It learns word embeddings that encode similar meanings and associations, allowing for tasks \
            like word analogies and similarity calculations. Handles Rare Words: The skip-gram model performs well \
            even with rare words or words with limited occurrences in the training data. It can generate meaningful \
            representations for such words by leveraging the context in which they appear. Contextual Flexibility: \
            The skip-gram model allows for flexible context definitions by using a window around each target word. \
            This flexibility captures local and global word associations, resulting in richer semantic \
            representations. Scalability: The skip-gram model can be trained efficiently on large-scale \
            datasets due to its simplicity and parallelization potential. It can process vast amounts of text data \
            to generate high-quality word embeddings."
    w2v_dict = train(test_text)

    cleaned_test_text = clean(test_text)
    cleaned_test_text = [cleaned_test_text.split()]

    print(cleaned_test_text)

    model = Word2Vec(cleaned_test_text, vector_size=30, window=2, sg=1)

    print(w2v_dict)
