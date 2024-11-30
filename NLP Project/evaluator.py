from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessor import process_bow_and_tfidf
import pandas as pd

def stratified_split_data(data, x_tfidf, train_ratio=0.75, label_column="Label"):
    
    class_0 = data[data[label_column] == 0]
    class_1 = data[data[label_column] == 1]
    
    train_class_0 = class_0.sample(frac=train_ratio, random_state=42)
    test_class_0 = class_0.drop(train_class_0.index)
    
    train_class_1 = class_1.sample(frac=train_ratio, random_state=42)
    test_class_1 = class_1.drop(train_class_1.index)
    
    train_data = pd.concat([train_class_0, train_class_1]).sample(frac=1, random_state=42)
    test_data = pd.concat([test_class_0, test_class_1]).sample(frac=1, random_state=42)
    
    x_train = x_tfidf[train_data.index]
    x_test = x_tfidf[test_data.index]
    
    y_train = train_data[label_column]
    y_test = test_data[label_column]
    
    return x_train, x_test, y_train, y_test


def train_and_evaluate_model(x_train, x_test, y_train, y_test, tfidf_vectorizer):
    
    model = MultinomialNB()
    model.fit(x_train, y_train)
    
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    
    print("Cross Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())
    
    y_pred = model.predict(x_test)
    print("Test Data Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    feature_log_prob = model.feature_log_prob_
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    spam_features = pd.Series(feature_log_prob[1], index=feature_names).sort_values(ascending=False)
    non_spam_features = pd.Series(feature_log_prob[0], index=feature_names).sort_values(ascending=False)
    
    print("Top 10 Spam Features:\n", spam_features.head(10))
    print("Top 10 Non-Spam Features:\n", non_spam_features.head(10))



