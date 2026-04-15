print("GANIKA R - 24BAD025")
import warnings  
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
df = pd.read_csv(r"C:\Users\Administrator\Downloads\archive (15)\StudentsPerformance.csv")
print("Initial Shape :", df.shape)

print("\nMissing Values:\n", df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

print("Shape After Cleaning :", df.shape)
le = LabelEncoder()
categorical_cols = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
df['final_exam_score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
) / 3
np.random.seed(42)
df['study_hours'] = np.random.randint(1, 6, size=len(df))
df['attendance'] = np.random.randint(60, 100, size=len(df))
df['sleep_hours'] = np.random.randint(5, 9, size=len(df))
X = df[
    [
        'gender',
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course',
        'study_hours',
        'attendance',
        'sleep_hours'
    ]
]
y = df['final_exam_score']
print("\nFeatures Used:\n", X.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(
    y.values.reshape(-1, 1)
)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n Linear Regression:")
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_.flatten()
})
print("\nFeature Importance:\n")
print(
    coefficients.sort_values(
        by='Coefficient',
        key=abs,
        ascending=False
    )
)

rfe = RFE(lr_model, n_features_to_select=3)
rfe.fit(X_train, y_train)
selected_features = list(X.columns[rfe.support_])
print("\nSelected Features (RFE):", selected_features)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("\nRidge R2 :", r2_score(y_test, ridge_pred))
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso R2 :", r2_score(y_test, lasso_pred))

plt.figure(figsize=(8,5))
plt.plot(
    y_test,
    label="Actual",
    linewidth=2
)
plt.plot(
    y_pred,
    label="Predicted",
    linewidth=2
)
plt.title("Actual vs Predicted Scores")
plt.xlabel("Test Samples")
plt.ylabel("Scaled Exam Score")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(
    x='Coefficient',
    y='Feature',
    hue='Feature',
    data=coefficients,
    palette="magma",
    legend=False
)
plt.title("Feature Influence")
plt.show()

residuals = y_test.flatten() - y_pred.flatten()
plt.figure(figsize=(8,5))
sns.histplot(
    residuals,
    kde=True
)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()