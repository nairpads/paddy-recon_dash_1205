
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the labeled training data
df = pd.read_csv("recon_training_template.csv")

# Define features and label
X = df[['amount_diff', 'date_diff', 'dr_cr', 'ref_match', 'desc_match']]
y = df['match']

# Split data (for checking performance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "recon_rf_model.pkl")
print("✅ Trained model saved as recon_rf_model.pkl")
