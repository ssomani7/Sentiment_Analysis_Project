from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def Multinomial_Naive_Bayes(x_train, y_train):
    model = MultinomialNB().fit(x_train, y_train)
    return model

def Bernoulli_Naive_Bayes(x_train, y_train):
    model = BernoulliNB().fit(x_train, y_train)
    return model

def Logistic_Regression():
    logreg = LogisticRegression(C=1e5)
    return logreg

# def SVM(x_train, y_train):
#     model = SVC()
#     model.fit(x_train, y_train)
#     return model