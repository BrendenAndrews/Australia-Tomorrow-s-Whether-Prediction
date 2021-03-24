import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from category_encoders import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import pickle
import os

warnings.filterwarnings('ignore')
os.chdir("F:\Deploy Australia Weather Prediction")
print(os.getcwd())

df = pd.read_csv("F:\\Project CSV's\\weatherAUS.csv")

print(df.head())
print(len(df['Location'].unique()))


for i in df['WindGustDir'].unique():
   print(f'<option value="{i}">')
    
        
df.shape

df.info()

df.isna().sum()

thresholdmissing = (df.shape[0]*25)/100

df['MinTemp'].isna().sum()

for i in df.columns:
    if df[i].isna().sum()>thresholdmissing:
        df = df.drop([i],axis=1)

df.shape

df.head()

df = df.drop(['Date'],axis=1)

df.head()


df['Humidity'] = (df['Humidity9am']+df['Humidity3pm'])/2

df['Pressure'] = (df['Pressure9am']+df['Pressure3pm'])/2

df['Temp'] = (df['Temp9am']+ df['Temp3pm'])/2

df.columns

df = df.drop(['WindDir9am', 'WindDir3pm', 'WindSpeed9am','WindSpeed3pm', 'Humidity9am', 
         'Humidity3pm', 'Pressure9am','Pressure3pm', 'Temp9am', 'Temp3pm'],axis=1)

df.head()

df.isna().sum()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib qt
sns.countplot(df['Location'])
plt.xticks(rotation=90)
plt.plot()

df['Location'].unique()

df['Location'].replace('SydneyAirport','Sydney',inplace=True)
df['Location'].replace('MelbourneAirport','Melbourne',inplace=True)
df['Location'].replace('PerthAirport','Perth',inplace=True)

df['Location'].unique()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib qt
sns.countplot(df['Location'])
plt.xticks(rotation=90)
plt.plot()

df.isna().sum()

df['WindGustDir'].unique()

sns.countplot(df['WindGustDir'])
plt.xticks(rotation=90)
plt.plot()

df['WindGustDir'].value_counts()

df['RainToday'].unique()

sns.countplot(df['RainToday'])

sns.countplot(df['RainTomorrow'])

df['RainTomorrow'].value_counts()

df.shape[0]*0.30

df.head()

df = df[~(df['RainTomorrow'].isna())]
df = df[~(df['RainToday'].isna())]
df = df[~(df['WindGustDir'].isna())]

df.isna().sum()

df.shape

df['RainTomorrow'].value_counts()

df = df.reset_index()

df = df.drop(['index'],axis=1)

df.head()

raintom = {'No':0,
          'Yes':1}

df['RainTomorrow'] = df['RainTomorrow'].map(raintom)

df.head()

df['RainTomorrow'].unique()

dfloc = df.groupby(df['Location'])['RainTomorrow'].mean().to_dict()

df['Location']=df['Location'].map(dfloc)
print(dfloc)
df.head()

dfdir = df.groupby(df['WindGustDir'])['RainTomorrow'].mean().to_dict()
print(dfdir)

df['WindGustDir']=df['WindGustDir'].map(dfdir)

df.head()

dftoday = df.groupby(df['RainToday'])['RainTomorrow'].mean().to_dict()
print(dftoday)

df['RainToday']=df['RainToday'].map(dftoday)

df.head()

df = df.drop(['MinTemp','MaxTemp'],axis=1)

df.head()

df.isna().sum()

estimator = LinearRegression()
imp = IterativeImputer(estimator,max_iter=100)

dfnew = imp.fit_transform(df)

dfnew = pd.DataFrame(dfnew,columns = df.columns)

dfnew.isna().sum()

x = dfnew.drop(['RainTomorrow'],axis=1)
y = dfnew[['RainTomorrow']]

print(x['Location'].unique)

y.head()
print(x.columns)
y['RainTomorrow'] = df['RainTomorrow'].astype('category')

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=42)

train_x.shape,test_x.shape,train_y.shape,test_y.shape

sm = SMOTE(random_state=42)

train_x_sm,train_y_sm = sm.fit_resample(train_x,train_y)

train_y_sm.value_counts()

model = XGBClassifier(n_estimators=350,subsample =1.0,max_depth=6,learning_rate=0.12244897959183673,
                     gamma=8,colsample_bytree=0.8,booster='gbtree')

model.fit(train_x_sm,train_y_sm)

pred_train = model.predict(train_x)
pred_test = model.predict(test_x)

print(accuracy_score(pred_train,train_y))

print(accuracy_score(pred_test,test_y))


pickle.dump(model,open('AusWeather.pkl','wb'))
AusWeather = pickle.load(open('AusWeather.pkl','rb'))
