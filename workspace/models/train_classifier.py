import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import pickle

import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(database_filepath):
    '''
    Load data from database given a file path
    Params:
      database_filepath(str): database file name included path
    Return:
      X(dataframe)                  : messages for X
      Y(dataframe)                  : categories in messages
      category_names(list of str)   : 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table('etl_pipeline', engine)  
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns.values
    return X, Y,category_names


def tokenize(text):
    '''
    Function: Process the text to return a standard version 
    Params:
      text(str): the message
    Return:
      lemmed(list of str): a lis of the tokenized words
    '''
    tokens = word_tokenize(text)
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = text.split()
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return lemmed


def build_model():
    '''
    Function: Process the text to return a standard version 
    Params:
    Return:
      cv(GridSearchCV): model
    '''
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {'text_pipeline__tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 


    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the given model. Report the f1 score, precision and recall of the model
    Params : 
        model(model)                : Classifier 
        X_test(list of str)         : Messages
        Y_test(dataframe)           : Categories
        category_names(list of str) : 
    Output: Report the f1 score, precision and recall for each output category of the dataset
    '''
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print("*************{}*************".format(col))
        print(classification_report(Y_test[col], Y_pred[:, i]))



def save_model(model, model_filepath):
    '''
    Save the given model as a pickle file
    Params:
      model(str): database file name included path
      model_filepath(str): filepath where to save the model
    Return:
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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