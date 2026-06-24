import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data/wine.csv")

x = df["alcohol"]
y = df["quality"]
a, b = np.polyfit(x, y, 1)
y_pred = a * x + b

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("alcohol")
plt.ylabel("quality")
plt.title("Linear Regression")
plt.savefig("regression_result.png")
plt.show()
