import pandas as pd
import nltk
import string

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(nltk.corpus.stopwords.words("english"))


class nlp_data:
    url = "https://raw.githubusercontent.com/ironhack-labs/project-nlp-challenge/main/dataset/data.csv"
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def __init__(self):
        self.df = pd.read_csv(self.__class__.url)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def __clean_text(self, text: str):
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        words = [
            self.__class__.lemmatizer.lemmatize(word)
            for word in words
            if word not in stop_words
        ]
        return " ".join(words)

    def preprocess(self):
        df = self.df.copy()
        df = df.dropna()
        df = df.drop_duplicates()
        df["clean_text"] = df["text"].apply(self.__clean_text)
        self.df = df
