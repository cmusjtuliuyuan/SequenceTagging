import numpy as np
import matplotlib.pyplot as plt



accuracys = [3.84, 89.32, 89.64, 88.93, 90.00, 89.78, 89.75, 89.67, 89.92, 89.57]
precisions = [1.75, 78.64, 79.48, 76.92, 79.90, 79.81, 79.83, 79.17, 79.64, 77.78]
recalls = [2.84, 82.60, 82.60, 82.14, 83.35, 83.10, 83.43, 83.15, 83.90, 83.36]
FB1s = [2.17, 80.57, 81.01, 79.45, 81.59, 81.42, 81.59, 81.11, 81.71, 80.47]

plt.figure()
plt.plot(accuracys,"g-",label="accuracy")
plt.plot(precisions,"r-.",label="precision")
plt.plot(recalls,"m-.",label="recalls")
plt.plot(FB1s,"k-.",label="FB1s")

plt.xlabel("epoches")
plt.ylabel("%")
plt.title("CONLL2000 dataset")

plt.grid(True)
plt.legend()
plt.show()