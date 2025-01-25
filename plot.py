import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', header=None,sep=";", names=['x', 'u'])



plt.figure(figsize=(10, 6))  
plt.plot(df['x'], df['u'], linestyle='-', marker='', color='black')
plt.title('Acoustic Material Vibrations')
plt.xlabel('X')
plt.ylabel('U(X)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()