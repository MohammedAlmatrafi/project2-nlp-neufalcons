import numpy as np
import pandas as pd


def embed_text(text, word2vec_model):
    # For Google embeddings, access the vectors directly without .wv
    word_vectors = [word2vec_model[word] for word in text if word in word2vec_model]
    
    # If no words in the model, return a zero vector
    if len(word_vectors) == 0:
        return np.zeros(word2vec_model.vector_size)  # Note: .vector_size, not .wv.vector_size
    
    # Return the average vector
    return word_vectors

def embed_texts(texts, word2vec_model):
    return texts.apply(lambda x: embed_text(x, word2vec_model))


def embed_text_with_pool(text, word2vec_model, aggr_func=np.mean):
    # Get embeddings for words that exist in the model
    word_vectors = [word2vec_model[word] for word in text if word in word2vec_model]
    
    # If no words in the model, return a zero vector
    if len(word_vectors) == 0:
        return np.zeros(word2vec_model.vector_size)
    
    # Return the average vector
    return aggr_func(word_vectors, axis=0)


def pool(
    tokenized_train, 
    tokenized_test, 
    word2vec_model, 
    aggr_func=np.mean
):
    aggr_func = np.mean
    train_embeddings = np.array([embed_text_with_pool(text, word2vec_model, aggr_func) for text in tokenized_train])
    test_embeddings = np.array([embed_text_with_pool(text, word2vec_model, aggr_func) for text in tokenized_test])
    train_embeddings = pd.DataFrame(train_embeddings, index=tokenized_train.index)
    test_embeddings = pd.DataFrame(test_embeddings, index=tokenized_test.index)
    return train_embeddings, test_embeddings