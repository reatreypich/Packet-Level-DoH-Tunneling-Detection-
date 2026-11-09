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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# -------- Model (light) --------
model = RandomForestClassifier(
    n_estimators=5,
    max_depth=5,
    random_state=42,
    n_jobs=1
)

# -------- Train --------
model.fit(X_train, y_train)

# -------- Predict (direct, no batch list) --------
y_pred = model.predict(X_test)

# -------- Accuracy --------
accuracy = accuracy_score(y_test, y_pred)

# -------- Save report (same format as before) --------
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.loc['accuracy', 'precision'] = accuracy

report_df.to_csv("random_forest_results.csv", index=True)

# -------- Save predictions (same format as before) --------
pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})

pred_df.to_csv("random_forest_predictions.csv", index=False)

print("Accuracy:", accuracy)
