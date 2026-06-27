import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('./dataset/public/train.csv')
test = pd.read_csv('./dataset/public/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")

# Feature Engineering
def create_features(df):
    df = df.copy()
    
    # Handle categorical columns
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0})
    if 'smoking' in df.columns:
        df['smoking_encoded'] = df['smoking'].map({'Never': 0, 'Former': 1, 'Current': 2})
    if 'alcohol_intake' in df.columns:
        df['alcohol_intake_encoded'] = df['alcohol_intake'].map({'None': 0, 'Moderate': 1, 'Heavy': 2})
    if 'family_history' in df.columns:
        df['family_history_encoded'] = df['family_history'].map({'No': 0, 'Yes': 1})
    if 'diabetes' in df.columns:
        df['diabetes_encoded'] = df['diabetes'].map({'No': 0, 'Yes': 1})
    if 'obesity' in df.columns:
        df['obesity_encoded'] = df['obesity'].map({'No': 0, 'Yes': 1})
    if 'exercise_induced_angina' in df.columns:
        df['exercise_induced_angina_encoded'] = df['exercise_induced_angina'].map({'No': 0, 'Yes': 1})
    if 'chest_pain_type' in df.columns:
        chest_pain_map = {
            'Asymptomatic': 0,
            'Non-anginal Pain': 1,
            'Atypical Angina': 2,
            'Typical Angina': 3
        }
        df['chest_pain_type_encoded'] = df['chest_pain_type'].map(chest_pain_map)
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    
    # Interaction features
    df['chol_bp'] = df['cholesterol'] * df['blood_pressure']
    df['hr_bp'] = df['heart_rate'] * df['blood_pressure']
    df['chol_hr'] = df['cholesterol'] * df['heart_rate']
    df['chol_bs'] = df['cholesterol'] * df['blood_sugar']
    df['bp_bs'] = df['blood_pressure'] * df['blood_sugar']
    
    # Risk scores
    df['cardiac_risk'] = (
        df['smoking_encoded'] * 2 + 
        df['diabetes_encoded'] * 2 + 
        df['family_history_encoded'] * 1.5 + 
        df['obesity_encoded'] * 1.5 +
        df['exercise_induced_angina_encoded'] * 2
    )
    
    df['lifestyle_risk'] = (
        df['smoking_encoded'] + 
        df['alcohol_intake_encoded'] + 
        (df['exercise_hours'] < 2).astype(int) * 2 +
        df['stress_level'] / 10
    )
    
    # Blood sugar categories
    df['high_blood_sugar'] = (df['blood_sugar'] > 126).astype(int)
    df['prediabetes'] = ((df['blood_sugar'] >= 100) & (df['blood_sugar'] <= 125)).astype(int)
    df['normal_blood_sugar'] = (df['blood_sugar'] < 100).astype(int)
    
    # Exercise categories
    df['exercise_adequate'] = (df['exercise_hours'] >= 3).astype(int)
    df['exercise_deficient'] = (df['exercise_hours'] < 1).astype(int)
    df['exercise_moderate'] = ((df['exercise_hours'] >= 1) & (df['exercise_hours'] < 3)).astype(int)
    
    # Cholesterol categories
    df['high_cholesterol'] = (df['cholesterol'] > 240).astype(int)
    df['borderline_cholesterol'] = ((df['cholesterol'] >= 200) & (df['cholesterol'] <= 239)).astype(int)
    df['normal_cholesterol'] = (df['cholesterol'] < 200).astype(int)
    
    # Blood pressure categories
    df['normal_bp'] = (df['blood_pressure'] < 120).astype(int)
    df['elevated_bp'] = ((df['blood_pressure'] >= 120) & (df['blood_pressure'] < 130)).astype(int)
    df['hypertension_stage1'] = ((df['blood_pressure'] >= 130) & (df['blood_pressure'] < 140)).astype(int)
    df['hypertension_stage2'] = (df['blood_pressure'] >= 140).astype(int)
    
    # Heart rate categories
    df['normal_hr'] = ((df['heart_rate'] >= 60) & (df['heart_rate'] <= 100)).astype(int)
    df['tachycardia'] = (df['heart_rate'] > 100).astype(int)
    df['bradycardia'] = (df['heart_rate'] < 60).astype(int)
    
    # Stress categories
    df['high_stress'] = (df['stress_level'] > 7).astype(int)
    df['moderate_stress'] = ((df['stress_level'] >= 4) & (df['stress_level'] <= 7)).astype(int)
    df['low_stress'] = (df['stress_level'] < 4).astype(int)
    
    # Combined risk factors
    df['multiple_risks'] = (
        df['smoking_encoded'] + 
        df['diabetes_encoded'] + 
        df['obesity_encoded'] + 
        df['family_history_encoded'] +
        df['high_cholesterol'] +
        df['hypertension_stage2']
    )
    
    # Metabolic syndrome indicators
    df['metabolic_syndrome'] = (
        df['obesity_encoded'] + 
        df['diabetes_encoded'] + 
        df['hypertension_stage2']
    )
    
    # Polynomial features
    df['cholesterol_squared'] = df['cholesterol'] ** 2
    df['blood_pressure_squared'] = df['blood_pressure'] ** 2
    df['heart_rate_squared'] = df['heart_rate'] ** 2
    df['blood_sugar_squared'] = df['blood_sugar'] ** 2
    df['stress_level_squared'] = df['stress_level'] ** 2
    
    # Ratios
    df['chol_bp_ratio'] = df['cholesterol'] / (df['blood_pressure'] + 1)
    df['hr_bp_ratio'] = df['heart_rate'] / (df['blood_pressure'] + 1)
    df['bs_bp_ratio'] = df['blood_sugar'] / (df['blood_pressure'] + 1)
    df['chol_hr_ratio'] = df['cholesterol'] / (df['heart_rate'] + 1)
    
    # Gender interactions
    df['gender_smoking'] = df['gender_encoded'] * df['smoking_encoded']
    df['gender_diabetes'] = df['gender_encoded'] * df['diabetes_encoded']
    df['gender_obesity'] = df['gender_encoded'] * df['obesity_encoded']
    df['gender_cholesterol'] = df['gender_encoded'] * df['cholesterol']
    df['gender_bp'] = df['gender_encoded'] * df['blood_pressure']
    
    return df

print("\nApplying feature engineering...")
train_fe = create_features(train)
test_fe = create_features(test)

# Prepare features - exclude original categorical columns and use encoded versions
exclude_cols = ['id', 'target', 'gender', 'smoking', 'alcohol_intake', 'family_history', 
                'diabetes', 'obesity', 'exercise_induced_angina', 'chest_pain_type']
feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
X = train_fe[feature_cols]
y = train_fe['target']
X_test = test_fe[feature_cols]

print(f"Number of features: {len(feature_cols)}")

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for predictions
oof_rf = np.zeros(len(X))
oof_gb = np.zeros(len(X))
oof_et = np.zeros(len(X))
oof_lr = np.zeros(len(X))
oof_mlp = np.zeros(len(X))

test_rf = np.zeros(len(X_test))
test_gb = np.zeros(len(X_test))
test_et = np.zeros(len(X_test))
test_lr = np.zeros(len(X_test))
test_mlp = np.zeros(len(X_test))

print("\nTraining models...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_folds}", end=" ")
    
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Random Forest
    model_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42 + fold,
        n_jobs=-1
    )
    model_rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = model_rf.predict_proba(X_val)[:, 1]
    test_rf += model_rf.predict_proba(X_test_scaled)[:, 1] / n_folds
    
    # Gradient Boosting
    model_gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42 + fold
    )
    model_gb.fit(X_tr, y_tr)
    oof_gb[val_idx] = model_gb.predict_proba(X_val)[:, 1]
    test_gb += model_gb.predict_proba(X_test_scaled)[:, 1] / n_folds
    
    # Extra Trees
    model_et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_et.fit(X_tr, y_tr)
    oof_et[val_idx] = model_et.predict_proba(X_val)[:, 1]
    test_et += model_et.predict_proba(X_test_scaled)[:, 1] / n_folds
    
    # Logistic Regression
    model_lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42 + fold
    )
    model_lr.fit(X_tr, y_tr)
    oof_lr[val_idx] = model_lr.predict_proba(X_val)[:, 1]
    test_lr += model_lr.predict_proba(X_test_scaled)[:, 1] / n_folds
    
    # Neural Network
    model_mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        alpha=0.01,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42 + fold
    )
    model_mlp.fit(X_tr, y_tr)
    oof_mlp[val_idx] = model_mlp.predict_proba(X_val)[:, 1]
    test_mlp += model_mlp.predict_proba(X_test_scaled)[:, 1] / n_folds
    
    print("✓")

# Ensemble predictions (weighted average)
oof_ensemble = (oof_rf * 0.25 + oof_gb * 0.25 + oof_et * 0.20 + oof_lr * 0.15 + oof_mlp * 0.15)
test_ensemble = (test_rf * 0.25 + test_gb * 0.25 + test_et * 0.20 + test_lr * 0.15 + test_mlp * 0.15)

# Find best threshold
thresholds = np.linspace(0, 1, 200)
best_mcc = -1
best_threshold = 0.5

print("\nOptimizing threshold...")
for threshold in thresholds:
    y_pred = (oof_ensemble >= threshold).astype(int)
    mcc = matthews_corrcoef(y, y_pred)
    if mcc > best_mcc:
        best_mcc = mcc
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best OOF MCC: {best_mcc:.6f}")

# Create submission
submission = pd.DataFrame({
    'id': test_fe['id'],
    'prediction': test_ensemble
})

# Save to working directory
submission.to_csv('./working/submission.csv', index=False)

print(f"\n✅ Submission saved!")
print(f"Shape: {submission.shape}")
print(f"Prediction range: [{submission['prediction'].min():.6f}, {submission['prediction'].max():.6f}]")
print(f"Mean prediction: {submission['prediction'].mean():.6f}")
