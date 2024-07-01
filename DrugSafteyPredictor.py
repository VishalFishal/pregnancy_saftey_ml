import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from imblearn.over_sampling import SMOTE
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import joblib

# Load your dataset
df = pd.read_csv('Enter the path of where you saved the dataset here', low_memory=False)

category_mapping = {
    'A': 'safe',
    'B1': 'safe',
    'B2': 'safe',
    'B3': 'safe',
    'C': 'unsafe',
    'D': 'unsafe',
    'X': 'unsafe'
}
df['Category'] = df['Category'].replace(category_mapping)

# Initialize the descriptor calculator with all available descriptors
descriptor_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Function to standardize SMILES strings and calculate all descriptors
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Canonicalize and sanitize the molecule
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
            Chem.SanitizeMol(mol)
            return mol
    except:
        return None

def calculate_descriptors(smiles):
    mol = standardize_smiles(smiles)
    if mol:
        descriptors = calculator.CalcDescriptors(mol)
        descriptor_dict = dict(zip(descriptor_names, descriptors))
        return descriptor_dict
    return None

# Apply the descriptor calculation function to the SMILES column
df['Descriptors'] = df['SMILES'].apply(calculate_descriptors)

# Drop rows with NaN values in descriptors
df = df.dropna(subset=['Descriptors'])

# Expand the descriptors dictionary into separate columns
descriptors_df = pd.DataFrame(df['Descriptors'].tolist(), index=df.index)

# Concatenate the original DataFrame with the new descriptors DataFrame
df = pd.concat([df, descriptors_df], axis=1)

# Encode target variable without extra spaces
df['Category'] = df['Category'].map({'safe': 1, 'unsafe': 0})

df['Classification 1'] = df['Classification 1'].str.lower().str.strip()
categorical_features = ['Classification 1']
df = pd.get_dummies(df, columns=categorical_features)
feature_names = df.drop(['Name', 'SMILES', 'Descriptors', 'CAS Number', 'Category', 'Formula'], axis=1).columns
joblib.dump(feature_names, 'feature_names.pkl')

# Drop original 'SMILES', 'Standardized_SMILES', and 'Descriptors' columns
X = df.drop(['Name', 'SMILES', 'Descriptors', 'CAS Number', 'Category', 'Formula'], axis=1)
y = df['Category']

newdf = pd.concat([X, y], axis=1).dropna()
# Drop rows with any NaN values in the feature matrix

X = newdf.drop(['Category'], axis=1)
# Adjust the target variable to match the dropped rows
y = newdf['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for RandomForestClassifier
print('Doing hyperparameter tuning...')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)

# Best model from GridSearch
best_rf = grid_search.best_estimator_

# Fit the model to the training data
best_rf.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print("\nTest Set Performance:")
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
print("Kappa Score:", kappa)
print("MCC:", mcc)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
