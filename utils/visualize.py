import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

data_df = pd.concat([train_df, test_df])

X = data_df.drop(columns=['label']).values
y = data_df['label'].values

x_col = X[:, 0]
y_col = X[:, 1]
z_col = X[:, 2] 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['b' if label == 0 else 'y' for label in y]
ax.scatter(x_col, y_col, z_col, c=colors, marker='o')

ax.set_xlabel('Acceleration X')
ax.set_ylabel('Gyroscope Y')
ax.set_zlabel('Water Level')
ax.set_title('Sensor Data Visualization')

normal_patch = plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='b', markersize=10)
urinating_patch = plt.Line2D([0], [0], marker='o', color='w', label='Urinating', markerfacecolor='y', markersize=10)
ax.legend(handles=[normal_patch, urinating_patch])

plt.show()
