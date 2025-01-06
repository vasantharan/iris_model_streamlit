from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
x = data.data
y = data.target

model = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=3)
model.fit(x, y)

joblib.dump(model, 'model.joblib')