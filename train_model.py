import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("netflix_titles.csv")

# Keep required columns
df = df[["type", "rating", "release_year", "duration"]].dropna()

# Convert target
df["type"] = df["type"].map({"Movie": 0, "TV Show": 1})

# Encode rating
le_rating = LabelEncoder()
df["rating_encoded"] = le_rating.fit_transform(df["rating"])

# Convert duration
def parse_duration(x):
    if "min" in x:
        return int(x.replace(" min", ""))
    elif "Season" in x:
        return int(x.split(" ")[0]) * 60
    return 0

df["duration_num"] = df["duration"].apply(parse_duration)

# Features & target
X = df[["release_year", "rating_encoded", "duration_num"]]
y = df["type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", accuracy)

# Save model & encoder
joblib.dump(model, "netflix_type_model.pkl")
joblib.dump(le_rating, "rating_encoder.pkl")
