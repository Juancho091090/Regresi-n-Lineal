# Librerias Usadas
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Datos kilometraje y Combustible
kilometers = np.array([50, 120, 80, 150, 60, 90, 130, 100, 70, 140, 110, 130, 80, 160, 40, 190, 70, 150, 120, 100, 130, 160, 90, 80, 140])
fuel_consumption = np.array([200, 450, 300, 500, 220, 350, 470, 400, 250, 480, 390, 420, 270, 510, 180, 550, 240, 470, 400, 350, 430, 520, 320, 290, 490])

x = kilometers.reshape(-1, 1)
y = fuel_consumption

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modelo de regresión
model = LinearRegression()
model.fit(x_train, y_train)

# Predicción para todo el rango de datos (entrenamiento y prueba)
y_pred_full = model.predict(x)

# Predicción solo para el conjunto de prueba
y_pred_test = model.predict(x_test)

r2 = model.score(x_test, y_test)
coefficient = model.coef_[0]
intercept = model.intercept_

print("Linear Regression")
print("R2 Score:", r2)
print("Coefficient:", coefficient)
print("Intercept:", intercept)

# Gráfico entre el Combustible y el kilometraje se crea acá
plt.scatter(x, y, color='blue', label='Datos reales (kilómetros)')
plt.plot(x, y_pred_full, color='red', label='Línea de regresión')
plt.xlabel('Kilómetros')
plt.ylabel('Consumo de combustible (litros)')
plt.title(f'Regresión lineal: Kilómetros vs Consumo (R2: {r2:.2f})')
plt.legend()
plt.show()