import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def separator():
    print()
    print("================================================================================")
    print()

os.makedirs("grafice", exist_ok=True)

df = pd.read_csv("./patient_cancer_prediction.csv")
X = df.drop('Has_Cancer', axis=1)
y = df['Has_Cancer']

print("Date lipsa din setul de date:")
lipsa = X.isnull().sum().to_frame('count')
lipsa['percent'] = 100 * lipsa['count'] / len(X)
print(lipsa)
separator()

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42)

print(f"Dimensiunea setului de antrenament: {X_train.shape}")
print(f"Dimensiunea setului de test: {X_test.shape}")
separator()

print("Statistici coloane numerice set antrenament:")
print(X_train.describe().T)
separator()

print("Statistici coloane numerice set test:")
print(X_test.describe().T)
separator()

print("Statistici variabile categorice set antrenament:")
print(X_train.select_dtypes(include='object').describe().T)
separator()

print("Statistici variabile categorice set test:")
print(X_test.select_dtypes(include='object').describe().T)
separator()

# Ploturi pentru variabile numerice
for col in X_train.select_dtypes(include='number').columns:
    plt.figure(figsize=(6, 3))
    sns.histplot(X_train[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribuția pentru {col}')
    plt.tight_layout()
    plt.savefig(f"grafice/{col}_distributie.png")
    plt.close()

# Ploturi pentru variabile categorice
for col in X_train.select_dtypes(include='object').columns:
    plt.figure(figsize=(6, 3))
    sns.countplot(data=X_train, x=col, order=X_train[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Distribuția categorică pentru {col}')
    plt.tight_layout()
    plt.savefig(f"grafice/{col}_categoric.png")
    plt.close()

# Matricea de corelații
plt.figure(figsize=(12, 8))
corr = X_train.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matricea de corelații")
plt.tight_layout()
plt.savefig("grafice/corelatii_heatmap.png")
plt.close()

# Codificare labeluri categorice
siruri = ['Gender', 'Ethnicity', 'Smoking_Status', 'Alcohol_Use',
         'Physical_Activity', 'Family_History_Cancer', 'Tumoral_Marker_2',
         'Tumoral_Marker_5', 'Tumoral_Marker_8', 'Tumoral_Marker_11']

le = LabelEncoder()
for i in siruri:
    combined = pd.concat([X_train[i], X_test[i]], axis=0)
    le.fit(combined)
    X_train[i] = le.transform(X_train[i])
    X_test[i] = le.transform(X_test[i])

model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratețea modelului de regresie logistică: {accuracy:.2f}")

# Matrice de confuzie
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matricea de confuzie')
plt.tight_layout()
plt.savefig("grafice/matrice_confuzie.png")
plt.close()

print("Raport clasificare:")
print(classification_report(y_test, y_pred))