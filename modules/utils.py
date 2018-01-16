import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split


def read_sql():
    con = sqlite3.connect('../resources/database.sqlite')
    messages = pd.read_sql_query("""SELECT Score, Summary FROM Reviews WHERE Score != 3""", con)

    return messages

def partition(x):
    if x < 3:
        return 'negative'
    else:
        return 'positive'

def train_test_val_split(messages, test_size, seed):
    Score = messages['Score']
    Score = Score.map(partition)
    Summary = messages['Summary']
    X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=test_size, random_state=seed, shuffle=True)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, shuffle=True)
    return X_train, X_test, y_train, y_test, Score