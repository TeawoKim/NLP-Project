import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# download all required corpus
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data():
    # Load data:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_directory = os.path.join(base_dir, "Youtube05-Shakira.csv")
    return pd.read_csv(file_directory)


def check_basic_data(df: pd.DataFrame):
    # print first 3 rows
    print("\nfirst 3 rows:\n", df.head(3))

    # print the shape of the dataframe
    print("\nshape of the dataframe:\n", df.shape)

    # print the column names
    # print the column types
    # print the missing values per column
    print(df.info())
    print("\nthe missing values per column:", df.isnull().sum(axis=0))

    # print the unique values for all columns
    for cl_name in df.columns:
        print(f"unique values for '{cl_name}' column:", len(df[cl_name].unique()))


def clear_comments(comments):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    en_stopwords = stopwords.words("english")
    remove_html_tags = re.compile(r'<[/]?[a-zA-Z0-9\s]+[/]?>')

    def clear_comments(comment: str):
        # lowercase all content
        l_comment = comment.lower()

        # remove html tags
        cleared_comment = re.sub(remove_html_tags, '', l_comment)

        # replaced escapled characters
        cleared_comment = cleared_comment.replace('&#39;', '\'')
        cleared_comment = cleared_comment.replace("&lt;", "<")
        cleared_comment = cleared_comment.replace("&gt;", ">")

        # remove unknown symbol at the end of each coment
        cleared_comment = cleared_comment.replace("\ufeff", "")

        # convert comments in list of tokens
        tokens = word_tokenize(cleared_comment)

        # filter stopwords, smiley, numbers and punctuation
        tokens = [t for t in tokens if t.isalpha() and t not in en_stopwords]

        # lemmatize all tokens
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]

        # stemming all tokens
        stemmed_tokens = [stemmer.stem(lemma) for lemma in lemmas]

        return ' '.join(stemmed_tokens)

    # iterate over each comment, clear it and convert it to list of tokens
    return comments.apply(clear_comments)


def pre_precess(df: pd.DataFrame):
    # drop columns that we don't need. We need to keep only CONTENT and CLASS
    df.drop(
        labels=["COMMENT_ID", "AUTHOR", "DATE"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    print("\nshape of the dataframe:\n", df.shape)

    # iterate over each comment, clear it and convert it to list of tokens
    df['preprocessed_comments'] = clear_comments(df['CONTENT'])

    print("\nfirst 5 rows:\n", df["preprocessed_comments"].head(5))

    return df


def vectorize_tokens(df: pd.DataFrame):
    documents = df["preprocessed_comments"]

    # vectorize text with bag of words
    bw_vectorizer = CountVectorizer()
    bw_vectorized = bw_vectorizer.fit_transform(documents)

    # print information about vectorized dataframe
    print("\nshape of the vectorized dataframe:\n", bw_vectorized.shape)
    bw_features = bw_vectorizer.get_feature_names_out()
    print("\nList of words:\n", bw_features)
    print(bw_vectorized.toarray())

    # print counts for words in first comment
    print([(bw_features[i], n) for i, n in enumerate(bw_vectorized.toarray()[0]) if n > 0])

    # vectorize text with Tf-Idf algorithm
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized = tfidf_vectorizer.fit_transform(documents)

    # print information about vectorized dataframe
    print("\nshape of the vectorized dataframe:\n", tfidf_vectorized.shape)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    print("\nList of words:\n", tfidf_features)
    print(tfidf_vectorized.toarray())

    # print tfidf log metrics for words in first comment
    print([(tfidf_features[i], n) for i, n in enumerate(tfidf_vectorized.toarray()[0]) if n > 0])

    return bw_vectorized, tfidf_vectorized, tfidf_vectorizer


def shuffle_and_split(vectirized, df: pd.DataFrame):
    joined: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(vectirized)
    joined["CLASS"] = df["CLASS"]
    shuffled = joined.sample(frac=1)

    trainig_amount = round(shuffled.shape[0] * 0.75)
    training = shuffled[:trainig_amount]
    test = shuffled[trainig_amount:]

    training_features = training.drop(
        labels=["CLASS"],
        axis=1,
    )
    training_target = training["CLASS"]

    test_features = test.drop(
        labels=["CLASS"],
        axis=1,
    )
    test_target = test["CLASS"]

    return training_features, training_target, test_features, test_target


def cross_validate_model(features, targets):
    scores = cross_val_score(MultinomialNB(), features, targets, scoring="accuracy", cv=5)
    print("Scores: ", scores)
    print("Mean: ", scores.mean())


def print_spam_and_non_spam_tokens(model: MultinomialNB, vectorizer):
    compare_df = pd.DataFrame({
        "spam_probs": model.feature_log_prob_[1],
        "non_span_probs": model.feature_log_prob_[0],
        "diff": model.feature_log_prob_[1] - model.feature_log_prob_[0],
        "features": vectorizer.get_feature_names_out()
    })

    sorted = compare_df.sort_values(by="diff", ascending=False)

    print("20 spam words: \n", sorted.head(20))
    print("20 non-spam words: \n", sorted.tail(20))


def main():
    comments_df = load_data()
    check_basic_data(comments_df)

    tokens_df = pre_precess(comments_df)
    _, tfidf_vectorized, tfidf_vectorizer = vectorize_tokens(tokens_df)
    training_feature, training_target, test_feature, test_target = shuffle_and_split(tfidf_vectorized, tokens_df)

    model = MultinomialNB()
    model.fit(training_feature, training_target)

    # cross validate model
    cross_validate_model(training_feature, training_target)

    predictions = model.predict(test_feature)

    print("Accuracy: ", accuracy_score(test_target, predictions))
    print("Classification report: ", classification_report(test_target, predictions))
    print("Confusion matrix: ", confusion_matrix(test_target, predictions))

    # print top 20 words for spam and for non spam comments
    print_spam_and_non_spam_tokens(model, tfidf_vectorizer)

    # Check non spam comments
    non_spam_comments = pd.Series([
        "I love Shakira, the sound is really good",
        "This clip is amaizing, video is really perfect. Waka Waka Waka)))))))",
        "the song is nice, but nothing special, have you heard Eminem",
        "I love Tailor Swift more"
    ])
    spam_comments = pd.Series([
        "Click a link to earn bitcoin, https://earn-bitcoin.com",
        "This is not a spam, do you earn some money, this is my facebook page"
    ])
    combined = pd.concat([non_spam_comments, spam_comments])
    predictions = model.predict(tfidf_vectorizer.transform(clear_comments(combined)))

    results = pd.DataFrame({
        "comment": combined,
        "predictions": ["NOT A SPAM" if p == 0 else "SPAM" for p in predictions]
    })

    print("comments predictions: \n", results)


if __name__ == "__main__":
    main()
