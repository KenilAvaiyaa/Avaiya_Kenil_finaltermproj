import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical


df = pd.read_csv("superhero dataset.csv")
df.head() # show how data look like 

df.describe() # show the basic statistics

df.info() # show the info of uploded data

print(df.isnull().sum()) # show if missing values
df = df.dropna()

print(df.duplicated().sum())
df= df.drop_duplicates() 

x = df.drop('is_good',axis=1)
y= df['is_good']

# check for unbalanced data
plt.figure(figsize=(8, 6))
sns.countplot(x='is_good', data=df,palette='Set2')
plt.title('Distribution of Target Variable (is_good)')
plt.xlabel('Is Good (0=Bad, 1=Good)')
plt.ylabel('Count')
plt.show()

print("Target value Distribution:")
print("\nVariable Distribution:")
print(y.value_counts())
print("\nVariable Percentages:")
print(y.value_counts(normalize=True) * 100)

# heatmap
fig, axis = plt.subplots(figsize=(10, 10))
correlation_x = x.corr()
sns.heatmap(correlation_x, annot=True, linewidths=.5, fmt='.2f', ax=axis, cmap="crest")
plt.show()


numerical_cols = ['height_cm', 'weight_kg', 'age', 'years_active',
                  'training_hours_per_week', 'civilian_casualties_past_year',
                  'power_level', 'public_approval_rating']

fig, axes = plt.subplots(4, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col], bins=40, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# pairplot
key_features = ['power_level', 'public_approval_rating', 'age',
                'civilian_casualties_past_year', 'is_good']
sns.pairplot(df[key_features], hue='is_good', diag_kind='hist')
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

std_scaler = StandardScaler()
normalized_x = std_scaler.fit_transform(x)

print(normalized_x[:1])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets to confirm
print("Training input data shape:", x_train.shape)
print("Testing input data shape:", x_test.shape)
print("Training lable shape:", y_train.shape)
print("Testing lable shape:", y_test.shape)


"""
### manual calculation of performance metrics
"""


def cal_performance_metrics(cm, y_test, y_pred, y_prob):
    TN, FP, FN, TP=cm.ravel()  

    # different metrics to calculate
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0 
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0 
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0 
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0 
    TSS = TPR - FPR 
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0 
    f1_score = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) != 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN)  
    Error_rate = 1 - Accuracy  
    BACC = (TPR + TNR) / 2
    HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) != 0 else 0  # Heidke Skill Score
    bs = brier_score_loss(y_test, y_prob)
    bs_ref = brier_score_loss(y_test, np.full_like(y_test, y_test.mean()))
    bss = (bs_ref - bs) / bs_ref if bs_ref != 0 else 0
    auc_value = roc_auc_score(y_test, y_prob) 

    return {
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "TPR": TPR, "TNR": TNR, "FPR": FPR, "FNR": FNR,"TSS":TSS,"HSS" : HSS,"BACC" : BACC,
        "Precision": Precision,"f1_score" : f1_score,  "Accuracy": Accuracy, "Error_rate": Error_rate,
        "BS" : bs, "BSS" : bss, "AUC" : auc_value
    }

"""
### Corss Validation
"""

# function for cross validations
def perform_10fold(normalized_x, y, model):
    cross_validation = KFold(n_splits=10, shuffle=True, random_state=42) #KFold() will split the data into 10 parts
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(cross_validation.split(normalized_x), 1):
        X_train, X_test = normalized_x[train_index], normalized_x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        metrics = cal_performance_metrics(cm,y_test,y_pred,y_prob)
        metrics['Fold'] = fold
        fold_metrics.append(metrics)

    return fold_metrics

"""
### KNN (k-nearest neighbors)
"""

print("\nKNN Algoritham\n")

knn = KNeighborsClassifier(n_neighbors=15)
fold_metrics = perform_10fold(normalized_x, y, knn)
df_fold_knn = pd.DataFrame(fold_metrics)
avg_fold_knn = df_fold_knn.mean().to_frame(name="Avarage(Mean)")
print(df_fold_knn.round(3).T)
print(avg_fold_knn.round(3))

"""### Randome Forest
"""
print("\nRandom Forest Alogoritham\n")

rf = RandomForestClassifier(min_samples_split= 10,n_estimators= 100, random_state=42)
fold_metrics=perform_10fold(normalized_x, y, rf)
df_fold_rf = pd.DataFrame(fold_metrics)
avg_fold_rf = df_fold_rf.mean().to_frame(name="Avarage(Mean)")
print(df_fold_rf.round(3).T)
print(avg_fold_rf.round(3))

"""### LSTM (Long-Short Term Memory)
"""

print("\nLSTM Algoritham\n")
x_train_lstm = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test_lstm = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

y_train_lstm = to_categorical(y_train)
y_test_lstm = to_categorical(y_test)

def lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cross_validation = KFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_index, test_index) in enumerate(cross_validation.split(x_train_lstm, y_train_lstm), 1):
    x_train, x_test = normalized_x[train_index], normalized_x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    print(f"Fold {fold}: x_train shape = {x_train.shape}")

    model = lstm_model(x_train.shape[1:])
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_prob = model.predict(x_test).reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    metrics = cal_performance_metrics(cm,y_test, y_pred,y_prob)

    metrics['Fold'] = fold
    fold_metrics.append(metrics)

df_fold_lstm = pd.DataFrame(fold_metrics)
avg_fold_lstm = df_fold_lstm.mean().to_frame(name="Average(Mean)")
print(df_fold_lstm.round(3).T)
print("")
print(avg_fold_lstm.round(3))

"""### Comparing all the Averages of Algorithms"""

print('\nComparing the average for all Algorithms')
print("")
averages_combined = pd.concat([avg_fold_knn, avg_fold_rf, avg_fold_lstm], axis=1)
averages_combined.columns = ["kNN average", "Random Forest average", "LSTM average"]

print(averages_combined)

"""
### generating ROC curve
"""
from sklearn.metrics import roc_curve, auc

x_train_roc = x_train.reshape(x_train.shape[0], x_train.shape[1])  # squeeze last dim
x_test_roc = x_test.reshape(x_test.shape[0], x_test.shape[1])


"""#### KNN
"""
knn_roc = KNeighborsClassifier(n_neighbors=15)
knn_roc.fit(x_train_roc, y_train)

y_pred = knn.predict(x_test_roc)
y_prob = knn.predict_proba(x_test_roc)[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=4, label= f'ROC curve area = {round(roc_auc,4)}')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN')
plt.legend(loc='lower right')
plt.show()


"""#### Randome Forest
"""
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
rf.fit(x_train_roc, y_train)

y_pred = rf.predict(x_test_roc)
y_prob = rf.predict_proba(x_test_roc)[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='green', lw=4, label=f'ROC curvecurve area = {round(roc_auc,4)}')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()


"""#### LSTM
"""

x_train_roc = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_roc = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(x_train_roc.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model.fit(x_train_roc, y_train, epochs=10, batch_size=32, verbose=0)

y_prob = lstm_model.predict(x_test_roc).flatten()
y_pred = (y_prob >= 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curvecurve area = {round(roc_auc,4)}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LSTM')
plt.legend(loc='lower right')
plt.show()