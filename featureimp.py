import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# Load data
# -----------------------
X = np.load("data/graph_features.npy")   # (156, 8)
y = np.array([0]*52 + [1]*52 + [2]*52)

feature_names = [
    "Num_Nodes",
    "Num_Edges",
    "Density",
    "Mean_Degree",
    "Std_Degree",
    "Self_Loop_Ratio",
    "Num_SCC",
    "Transition_Entropy"
]

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

# -----------------------
# Train RF
# -----------------------
rf = RandomForestClassifier(n_estimators=150, random_state=1)
rf.fit(X_train, y_train)

# -----------------------
# Feature Importance
# -----------------------
importances = rf.feature_importances_

print("\nFeature Importance:")
for name, val in zip(feature_names, importances):
    print(f"{name:20s}: {val:.4f}")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8,5))
plt.barh(feature_names, importances)
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
