####### learning curve (Python 代码)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
 
    plt.legend(loc="best")
    return plt

## 读取数据
dates=pd.read_csv('Matrix.txt',sep='\t') 
dates

y=dates['status']
X=dates.iloc[:,1:7]

# Naive Bayes
title = r"Learning Curves (Naive Bayes)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = GaussianNB()    #建模
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=cv, n_jobs=1)
plt.savefig('Learning Curves_NB.pdf',bbox_inches='tight')

# LogisticRegression
title = r"Learning Curves (Logistic Regression)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = LogisticRegression(penalty="l2", C=0.5, solver="liblinear") # 建模
plot_learning_curve(estimator, title, X, y, (0, 1.01), cv=cv, n_jobs=1)

#plt.show()
plt.savefig('Learning Curves_LR.pdf',bbox_inches='tight')


# SVM
title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)    # 建模
plot_learning_curve(estimator, title, X, y, (0, 1.01), cv=cv, n_jobs=1)
#plt.show()
plt.savefig('Learning Curves_SVM.pdf',bbox_inches='tight')
