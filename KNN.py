import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

db = pd.read_csv("B.csv", dtype=np.float32)
dm = pd.read_csv("M.csv", dtype=np.float32)

db["Label"] = np.uint8(0)
dm["Label"] = np.uint8(1)

df = pd.concat([db, dm], ignore_index=True)
#df = df.drop(columns=["seg20_entropy","seg19_entropy","seg11_entropy","seg10_entropy","mean","seg18_entropy","max","kurtosis"], errors="ignore")
#df = df.drop(df.columns[[12, 17, 13, 4, 16, 5, 14, 8]], axis=1)

#df = df.drop(columns=["seg10_entropy","seg17_entropy","seg18_entropy","seg19_entropy","mean","std_dev","max","kurtosis"], errors="ignore")
df = df.drop(columns=["skewness","seg18_entropy","seg15_entropy","seg14_entropy","min","std_dev","variance","kurtosis"], errors="ignore")


X = df.drop("Label", axis=1)
y = df["Label"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import csv
import gc

# ----------------------------
# Make data lighter
# ----------------------------
X_train = np.asarray(X_train, dtype=np.float32)
X_test  = np.asarray(X_test, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.uint8)
y_test  = np.asarray(y_test, dtype=np.uint8)

# ----------------------------
# Reduce training size (CRITICAL)
# ----------------------------
sample_size = min(200000, len(X_train))  # adjust if needed

idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_small = X_train[idx]
y_train_small = y_train[idx]

# Free memory
del X_train, y_train
gc.collect()

# ----------------------------
# Train KNN
# ----------------------------
model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=1   # safer for memory
)

model.fit(X_train_small, y_train_small)

# Free subset
del X_train_small, y_train_small
gc.collect()

# ----------------------------
# Predict in batches
# ----------------------------
batch_size = 5000

y_pred_all = []

with open("knn_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["y_true", "y_pred", "prob_class_0", "prob_class_1"])

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]

        y_pred_batch = model.predict(X_batch)
        probs_batch  = model.predict_proba(X_batch)

        y_pred_all.append(y_pred_batch)

        for yt, yp, pb in zip(y_batch, y_pred_batch, probs_batch):
            writer.writerow([int(yt), int(yp), float(pb[0]), float(pb[1])])

# Combine predictions
y_pred = np.concatenate(y_pred_all)

# ----------------------------
# Save report
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc["accuracy", "precision"] = accuracy

report_df.to_csv("knn_results.csv", index=True)

print("Accuracy:", accuracy)
print("Results saved to knn_results.csv")
print("Predictions saved to knn_predictions.csv")
