import numpy as np
import pandas as pd
import modules.utils as ut
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from modules.preprocessing import Preprocessing
from modules.model_training import Multinomial_Naive_Bayes, Logistic_Regression, Bernoulli_Naive_Bayes
# from modules.model_training import SVM


class P3:
    def __init__(self):
        messages = ut.read_sql()
        self.idata = pd.read_csv('../resources/implement.csv')
        self.idata = self.idata['Summary']
        self.X_train, self.X_test, self.y_train, self.y_test, self.Score = ut.train_test_val_split(messages, 0.2, 42)
        self.X_train_tfidf, self.X_test_tfidf, self.idata_tfidf = Preprocessing().train_preprocess(self.X_train, self.X_test, self.idata)
        # print(self.X_test_tfidf)

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(set(self.Score)))
        plt.xticks(tick_marks, set(self.Score), rotation=45)
        plt.yticks(tick_marks, set(self.Score))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def model(self):
        prediction = dict()
        vfunc = np.vectorize(lambda x: 0 if x == 'negative' else 1)
        cmp = 0
        colors = ['b', 'g', 'y', 'm', 'k']

        multinomial_model = Multinomial_Naive_Bayes(self.X_train_tfidf, self.y_train)
        prediction['Multinomial'] = multinomial_model.predict(self.X_test_tfidf)

        bernoulli_model = Bernoulli_Naive_Bayes(self.X_train_tfidf, self.y_train)
        prediction['Bernoulli'] = bernoulli_model.predict(self.X_test_tfidf)

        self.logistic_regression_model = Logistic_Regression()
        self.logistic_regression_model.fit(self.X_train_tfidf, self.y_train)
        prediction['Logistic'] = self.logistic_regression_model.predict(self.X_test_tfidf)

        # SVM_model = SVM(self.X_train_tfidf, self.y_train)
        # prediction['SVM'] = SVM_model.predict(self.X_test_tfidf)

        for model, predicted in prediction.items():
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test.map(lambda x: 0 if x == 'negative' else 1), vfunc(predicted))
            roc_auc = auc(false_positive_rate, true_positive_rate)
            plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model, roc_auc))
            cmp += 1

        plt.title('Classifiers comparison with ROC')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        print(metrics.classification_report(self.y_test, prediction['Logistic'], target_names=['positive', 'negative']))

        cm = confusion_matrix(self.y_test, prediction['Logistic'])
        np.set_printoptions(precision=2)
        plt.figure()
        self.plot_confusion_matrix(cm)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        self.plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        plt.show()


    def implementation(self):
        out = self.logistic_regression_model.predict(self.idata_tfidf)
        print(len(out))


def main():
    ex = P3()
    ex.model()
    ex.implementation()

if __name__ == '__main__':
    main()
