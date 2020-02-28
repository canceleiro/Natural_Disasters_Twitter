import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords', 'averaged_perceptron_tagger'])


import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import f1_score
import pickle


def load_data(table_name, database_filepath):
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table(table_name, database_filepath) 
    X = df['message']
    Y = df.drop(['id', 'message','original', 'genre'], axis=1)
    return X, Y


def tokenize(text):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = PorterStemmer().stem(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(tree.DecisionTreeClassifier()))
    ])
    



def evaluate_model(model, X_test, Y_test):
    y_pred = pipeline.predict(X_test)
    
    f1_scores = []
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i])
        f1 = f1_score(Y_test.values[:,i], y_pred[:,i], average='weighted')
        f1_scores.append(f1)
        print(classification_report(y_test.values[:,i], y_pred[:,i]))
    print("The average f1 weighted score of all the columns is", np.mean(f1_scores))


def save_model(model, model_filepath):
    pickle.dump(pipeline, open('finalized_model.sav', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(table_name, database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()