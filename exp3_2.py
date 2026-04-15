print("GANIKA R - 24BAD025")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\Administrator\Downloads\archive (16)\auto-mpg.csv")
print("Initial Shape :", df.shape)
df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = df['horsepower'].astype(float)
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
print("\nMissing Values:\n", df.isnull().sum())
X = df[['horsepower']].values
y = df['mpg'].values.reshape(-1, 1)
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_scaled,
    test_size=0.2,
    random_state=42
)
degrees = [2, 3, 4]
results = {}
train_errors = []
test_errors = []
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    results[d] = {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, y_pred)
    }
    y_train_pred = model.predict(X_train_scaled)
    train_errors.append(
        mean_squared_error(y_train, y_train_pred)
    )
    test_errors.append(mse)
for d in degrees:
    print(f"\nDegree {d}:")
    print(f"MSE  = {results[d]['MSE']:.4f}")
    print(f"RMSE = {results[d]['RMSE']:.4f}")
    print(f"R2   = {results[d]['R2']:.4f}")

poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
print("\nRidge R2 :", r2_score(y_test, ridge_pred))
X_sorted = np.sort(X, axis=0)
plt.figure(figsize=(10,6))
plt.scatter(X, y_scaled, alpha=0.4, label="Actual Data")
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    X_sorted_poly = poly.transform(X_sorted)
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    X_sorted_scaled = scaler.transform(X_sorted_poly)
    model = LinearRegression()
    model.fit(X_poly_scaled, y_scaled)
    y_curve = model.predict(X_sorted_scaled)
    plt.plot(X_sorted, y_curve, linewidth=2, label=f"Degree {d}")

plt.xlabel("Horsepower")
plt.ylabel("Scaled MPG")
plt.title("Polynomial Curve Fitting")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='o', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE (0–1 Range)")
plt.title("Training vs Testing Error")
plt.legend()
plt.grid(True)
plt.text(2, train_errors[0], "Underfitting", ha='center')
plt.text(3, train_errors[1], "Good Fit", ha='center')
plt.text(4, train_errors[2], "Overfitting", ha='center')
plt.show()
labels_map = {
    2: "Underfitting",
    3: "Good Fit",
    4: "Overfitting"
}

plt.figure(figsize=(10,6))
plt.scatter(X_train, y_train, alpha=0.5, label="Training Data")
plt.scatter(X_test, y_test, alpha=0.5, label="Testing Data")

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_sorted_poly = poly.transform(X_sorted)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_sorted_scaled = scaler.transform(X_sorted_poly)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_curve = model.predict(X_sorted_scaled)
    plt.plot(X_sorted, y_curve, linewidth=2, label=labels_map[d])
plt.xlabel("Horsepower")
plt.ylabel("Scaled MPG")
plt.title("Underfitting vs Overfitting")
plt.legend()
plt.grid(True)
plt.show()