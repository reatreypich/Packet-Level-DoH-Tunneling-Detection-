import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

db = pd.read_csv("B.csv", dtype=np.float32)
dm = pd.read_csv("M.csv", dtype=np.float32)

db["Label"] = np.uint8(0)
dm["Label"] = np.uint8(1)

df = pd.concat([db, dm], ignore_index=True)
df = df.drop(columns=["skewness","seg18_entropy","seg15_entropy","seg14_entropy","min","std_dev","variance","kurtosis"], errors="ignore")
#df = df.drop(columns=["seg20_entropy","seg19_entropy","seg11_entropy","seg10_entropy","mean","seg18_entropy","max","kurtosis"], errors="ignore")
#df = df.drop(df.columns[[12, 17, 13, 4, 16, 5, 14, 8]], axis=1)
#df = df.drop(columns=["seg10_entropy","seg17_entropy","seg18_entropy","seg19_entropy","mean","std_dev","max","kurtosis"], errors="ignore")
X = df.drop("Label", axis=1)
y = df["Label"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
X_train = np.asarray(X_train, dtype=np.float32)
X_test  = np.asarray(X_test, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.uint8)
y_test  = np.asarray(y_test, dtype=np.uint8)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Train model
model = SGDClassifier(loss="hinge", max_iter=20, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# =========================
# 1. Save classification report
# =========================
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

report_df = pd.DataFrame(report_dict).transpose()
report_df.loc["accuracy", "precision"] = accuracy
report_df.to_csv("SDGC_results.csv", index=True)

# =========================
# 2. Save predictions
# =========================
pred_df = pd.DataFrame({
    "y_true": np.asarray(y_test, dtype=np.uint8),
    "y_pred": np.asarray(y_pred, dtype=np.uint8)
})

scores = model.decision_function(X_test)
pred_df["decision_score"] = scores

pred_df.to_csv("SDGC_predictions.csv", index=False)

print("Accuracy:", accuracy)
