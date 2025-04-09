import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as cm
import pickle

df = pd.read_csv('C:/Users/Kidecha Kerul/Downloads/AirlinesRatingPrediction (2).csv')


df.head()

df.info()

df.isnull().sum()

mean_value = df['Arrival Delay in Minutes'].mean()
df['Arrival Delay in Minutes'].fillna(mean_value, inplace=True)

df.isnull().sum()

df.describe()
from sklearn.preprocessing import LabelEncoder
label={}
for col in df.select_dtypes (include=['object']).columns:

 label[col]= LabelEncoder()
 df[col]=label[col].fit_transform(df[col])

df.head()

df=df.iloc[:,2:]

df.head()

from sklearn.model_selection import train_test_split
X = df.loc[:, :'Arrival Delay in Minutes']
y = df['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()


X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Training set - Features:", X_train_scaled.shape, "Target:", y_train.shape)
print("Testing set - Features:", X_test_scaled.shape, "Target:", y_test.shape)

print("Training set - Features:", X_train.shape, "Target:", y_train.shape)
print("Testing set - Features:", X_test.shape, "Target:", y_test.shape)

features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target = ['satisfaction']
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC_AUC = {}".format(roc_auc))
    print(classification_report(y_test,y_pred,digits=5))
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')

    return model, roc_auc
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix

# Define parameters for XGBoost
params_xgb = {}

# Create XGBoost classifier
model_xgb = xgb.XGBClassifier(**params_xgb)

def run_model_xgb(model, X_train, y_train, X_test, y_test, verbose=True):
    if verbose:
        print("Training XGBoost model...")
    model.fit(X_train, y_train)
    print("...Complete")


    y_pred = model.predict(X_test)


    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("XGBoost Model Metrics:")
    print("ROC_AUC = {}".format(roc_auc))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(y_test, y_pred, digits=5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model, roc_auc, accuracy, cm
model_xgb, roc_auc_xgb, accuracy_xgb, cm_xgb = run_model_xgb(model_xgb, X_train, y_train, X_test, y_test)
pickle.dump(model_xgb, open("model.pkl", "wb"))
preprocessor = {"label_encoder": label, "scaler": scaler}
pickle.dump(preprocessor, open("preprocessor.pkl", "wb"))

