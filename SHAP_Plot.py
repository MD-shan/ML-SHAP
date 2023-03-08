### SHAP 可视化解释机器学习模型
## https://zhuanlan.zhihu.com/p/441302127
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, PolynomialFeatures
from category_encoders import WOEEncoder, BinaryEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

## 读取数据
train=pd.read_csv('Matrix.txt',sep='\t') 
train.head()

## Creating categorical and continuous variable list
cat_var = ["status"]
con_var = ['NLR', 'PLR', 'MLR', 'MHR', 'GAR', 'RPR']

## Separating features and target
X = train.drop(["status"], axis=1)
Y = train["status"]

## Bulding list of models to be trained
model_rf = RandomForestClassifier(random_state=1, n_jobs=-1)
model_logr = LogisticRegression(random_state=1, n_jobs=-1, multi_class='multinomial')
model_lgbm = LGBMClassifier(random_state=1, n_jobs=-1)
model_xgb = XGBClassifier(random_state=1, n_jobs=-1)
model_gbr = GradientBoostingClassifier(random_state=1)
model_cat = CatBoostClassifier(random_state=1, verbose=0)

models = []
models.append(('LR',model_logr))
models.append(('RF',model_rf))
models.append(('GBR',model_gbr))
models.append(('XGB',model_xgb))
models.append(('LGB',model_lgbm))
models.append(('CAT',model_cat))

## Preparing Pipeline Steps
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
feature = SelectFromModel(model_rf, threshold=0.001)
ct = ColumnTransformer([#('onehot', onehot, cat_var),
                        ('scaler', scaler, con_var)], remainder='passthrough', n_jobs=-1)
                        
results = []
names = []
for name, model in models:
    #pipe = Pipeline([('ct', ct), ('fselect', feature), (name, model)]) # including feature selection step using RF
    pipe = Pipeline([('ct', ct), (name, model)])
    scores = cross_val_score(pipe, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)
    names.append(name)
    results.append(scores)
    print("model %s accuracy: %.4f variance: %.4f"%(name, np.mean(scores), np.std(scores)))
    
plt.figure(figsize=(12,5))
plt.boxplot(results)
plt.xticks(np.arange(1,len(names)+1),names)
plt.title("Accuracy for different machine learning algorithms")
plt.xlabel("Model Name")
plt.ylabel("Cross val Accuracies")
plt.show()

### 利用shap解释模型
## Model interpretation using Shap
from sklearn.linear_model import LogisticRegression
import shap
pd.set_option("display.max_columns",None)
shap.initjs()
import xgboost
import eli5

## Linear Explainer for Logistic Regression
ct.fit(X)
X_shap = ct.fit_transform(X)
#test_shap  = ct.transform(test)
explainer = shap.LinearExplainer(logr_pipe.named_steps['LR'], X_shap, feature_perturbation="interventional")
shap_values = explainer(X_shap)
#shap_values = explainer.shap_values(test_shap)

# visualize all the training set predictions
shap.plots.force(shap_values)

shap.summary_plot(shap_values, X_shap, feature_names=con_var, 
                  plot_type="bar",show=False)
plt.savefig('barplot_shap.pdf',bbox_inches='tight')

shap.summary_plot(shap_values, X_shap, feature_names=con_var,show=False)
plt.savefig('splitPlot_shap.pdf',bbox_inches='tight')

## Model agnostic example with KernelExplainer
## https://github.com/slundberg/shap
import sklearn
import shap
from sklearn.model_selection import train_test_split

# print the JS visualization code to the notebook
shap.initjs()

## 读取数据
train=pd.read_csv('Matrix.txt',sep='\t') 
train.head()

## Separating features and target
train_new = train.drop(["status"], axis=1)
label = train["status"]

# train a SVM classifier
#X_train,X_test,Y_train,Y_test = train_test_split(train_new,label, test_size=0.2, random_state=0)
#svm = sklearn.svm.SVC(kernel='rbf', probability=True)
#svm.fit(X_train, Y_train)

# train a LR classifier
X_train,X_test,Y_train,Y_test = train_test_split(train_new,label, test_size=0.2, random_state=0)
LR = sklearn.linear_model.LogisticRegression(random_state=1, n_jobs=-1, multi_class='multinomial')
LR.fit(X_train, Y_train)


# plot the SHAP values for the Setosa output of all instances
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
#plt.savefig('force_plot_shap.pdf',bbox_inches='tight')

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# fig = plt.gcf()
# shap._*plot本质上就是调用 plt.show()， 所有我们的思路就是如何在下面的文件进行保存
f=shap.plots.force(explainer.expected_value[0], shap_values[0], X_test,show=False)
shap.save_html("force_plot_shap.html", f)


# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[1],show=False)
plt.savefig('waterfall_shap_PLR.pdf',bbox_inches='tight')
