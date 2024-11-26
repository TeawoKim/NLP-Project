from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk

nltk.download('punkt_tab')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text, remove_specials=False):
    tokens = word_tokenize(text.lower())

    if remove_specials:
        tokens = [word for word in tokens if word.isalnum()]


    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    result = " ".join(tokens)
    return result


def process_bow_and_tfidf(data, column_name="Processed_Comment"):


    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(data[column_name])

   
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)


    print("Bag of Words Shape:", X_counts.shape)
    print("TF-IDF Shape:", X_tfidf.shape)

    return X_counts, X_tfidf, vectorizer, tfidf_transformer