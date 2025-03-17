import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def custom_preprocessor(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r"[^\w\s]", "", text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)


def create_features(
    texts, n_features=1600
):  # train set -> we learn counts and create features
    vectorizer = CountVectorizer(
        preprocessor=custom_preprocessor,
        stop_words="english",
        max_features=n_features,  # remove rare word, will keep 1000 most frequent words
    )
    tokens = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    x_df = pd.DataFrame(tokens.toarray(), columns=features)
    return x_df, vectorizer


def apply_features(texts, vectorizer):  # we only aplly learned features
    tokens = vectorizer.transform(texts)
    features = vectorizer.get_feature_names_out()
    x_df = pd.DataFrame(tokens.toarray(), columns=features)
    return x_df


def split_train_test(texts):
    texts.index = texts.text
    x_train, x_test, y_train, y_test = train_test_split(
        texts.text, texts.label, test_size=0.2, random_state=62
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    return x_train, x_test, y_train, y_test


def create_bow(x_train, x_test):
    x_df, vectorizer = create_features(x_train)
    x_df.index = x_train
    x_df_test = apply_features(x_test, vectorizer)
    return x_df, x_df_test, vectorizer


def train_model(x_df, y_train):
    model = MultinomialNB()
    model.fit(x_df, y_train)
    return model


def predict(model, x_df_test, y_test):
    output = model.predict(x_df_test)
    print("BOW and MultinomialNB accuracy: ", accuracy_score(y_test, output))
    return output


def tokenize_texts(texts, vectorizer):
    """
    Tokenize preprocessed texts for Word2Vec training using CountVectorizer's vectorizer
    """
    tkz_func = vectorizer.build_analyzer()

    # Apply the tokenizer to each text
    tokenized_texts = texts.apply(tkz_func)
    tokenized_texts.index = texts.index
    return tokenized_texts
