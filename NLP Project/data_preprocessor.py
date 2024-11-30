from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
import re

nltk.download('punkt_tab')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text, remove_specials=False):
    
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    result = " ".join(tokens)
    return result

def process_bow_and_tfidf(data, column_name="Processed_Comment", ngram_range=(1,2)):
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=5, max_df=0.8, max_features=5000)
    
    X_tfidf = tfidf_vectorizer.fit_transform(data[column_name].astype(str))
    
    print("TF-IDF Shape:", X_tfidf.shape)
    print("Example N-gram Features:", tfidf_vectorizer.get_feature_names_out()[:10])
    
    return X_tfidf, tfidf_vectorizer

