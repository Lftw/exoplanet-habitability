import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/exoplanet_processed.csv')

data['hz_index'].hist(bins=30)
plt.xlabel('Hz Index')
plt.ylabel('Frequency')
plt.title('Distribution of Hz Index')
plt.show()

print('info:', data['hz_index'].describe())
