import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
X = np.load("data/graph_features.npy")   # (156, 8)
y = np.array([0]*52 + [1]*52 + [2]*52)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------
# Train-Test Split FIRST
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=1,
    shuffle=True,
    stratify=y
)

# -----------------------
# Scale using TRAIN only
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train:", X_train.shape, "Test:", X_test.shape)

# -----------------------
# Random Forest
# -----------------------
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=150, random_state=1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_rf))
cm_rf = confusion_matrix(y_test, pred_rf)
print("Confusion Matrix:\n", cm_rf)
print(classification_report(y_test, pred_rf))

# -----------------------
# SVM
# -----------------------
print("\n--- SVM ---")
svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_svm))
cm_svm = confusion_matrix(y_test, pred_svm)
print("Confusion Matrix:\n", cm_svm)
print(classification_report(y_test, pred_svm))

# -----------------------
# Logistic Regression
# -----------------------
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_lr))
cm_lr = confusion_matrix(y_test, pred_lr)
print("Confusion Matrix:\n", cm_lr)
print(classification_report(y_test, pred_lr))

# -----------------------
# Plot Confusion Matrices
# -----------------------
models = {
    "Random Forest": cm_rf,
    "SVM": cm_svm,
    "Logistic Regression": cm_lr
}

plt.figure(figsize=(12,4))
i = 1

for name, cm in models.items():
    plt.subplot(1,3,i)
    plt.imshow(cm, cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1,2], ["State1","State2","State3"])
    plt.yticks([0,1,2], ["State1","State2","State3"])

    for r in range(3):
        for c in range(3):
            plt.text(c, r, cm[r,c], ha="center", va="center")

    i += 1

plt.suptitle("Confusion Matrices for Graph-Based Classification")
plt.tight_layout()
plt.show()
