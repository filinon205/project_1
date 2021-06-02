import pandas as pd
import dill
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import f1_score
#working with text
from sklearn.feature_extraction.text import TfidfVectorizer
#normalizing data
from sklearn.preprocessing import StandardScaler
#pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_score,recall_score
#imputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
import sklearn.datasets
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

pd.options.display.max_columns = 150


df_ = pd.read_csv(r"C:\Users\nikita.saprykin\Desktop\Машинное обучение в бизнесе\new\spam.csv",  encoding='ISO-8859-1')
df_.head(15)

df_.describe().T

df_.isna().sum()

import missingno as msno
msno.bar(df_, color = '#6389df', figsize = (6,4))

df_ = df_[['v2','v1']]
df_.columns = ['sms','spam']
df=pd.DataFrame()
mapping = {'spam': 1,'ham': 0}
df['sms']=df_['sms']
df['spam']=df_['spam'].map(mapping)

df.head(5)


def remove_duplicate(data):
    print("До удаления дубликатов кол-во строк = ", df.shape[0])
    data.drop_duplicates(keep="first", inplace=True)
    print("После удаления дубликатов кол-во строк = ", df.shape[0])
    return "Проверка дубликатов"


remove_duplicate(df)

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(df)

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(df)

X_train, X_test, y_train, y_test = train_test_split(df,
                                                    df['spam'], test_size=0.33, random_state=42)
#save test
X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)
#save train
X_train.to_csv("X_train.csv", index=None)
y_train.to_csv("y_train.csv", index=None)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class TextImputer(BaseEstimator, TransformerMixin):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.key] = X[self.key].fillna(self.value)
        return X

    features = df['sms']
    target = df['spam']

#combine
sms = Pipeline([
                ('imputer', TextImputer('sms', '')),
                ('selector', ColumnSelector(key='sms')),
                ('tfidf', TfidfVectorizer(max_df=0.9, min_df=10))
            ])

feats = FeatureUnion([('sms', sms),
                      ])

pipeline = Pipeline([
    ('features',feats),
    ('classifier', LogisticRegression()),
])

pipeline.fit(X_train, y_train)
#Посмотрим, как выглядит наш pipeline
pipeline.steps

with open("pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)