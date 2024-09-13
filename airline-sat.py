import pandas as pd

df = pd.read_csv('/data/airline-passenger-satisfaction/train.csv')
print(df)
print(df.columns)

# ['Unnamed: 0', 'id', 'Gender', 'Customer Type', 'Age', 'Type of Travel',
#       'Class', 'Flight Distance', 'Inflight wifi service',
#       'Departure/Arrival time convenient', 'Ease of Online booking',
#       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
#       'Inflight entertainment', 'On-board service', 'Leg room service',
#       'Baggage handling', 'Checkin service', 'Inflight service',
#       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
#       'satisfaction']

df_sat = df[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
             'Class', 'Flight Distance', 'Food and drink', 'Online boarding', 'Seat comfort',
             'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Cleanliness',
             'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Gate location', 'On-board service',
             'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
             'satisfaction']]

print(df_sat)
for c in df_sat.columns:
    print(c, type(df_sat[c][0]))

# convert Class column to numeric using one-hot encoding
df_sat = pd.get_dummies(df_sat, columns=['Class', 'Gender', 'Customer Type', 'Type of Travel'])
# change satisfaction to numeric
df_sat['satisfaction'] = df_sat['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
print(df_sat)

# remove nans
df_sat = df_sat.dropna()

# cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

df_nosat = df_sat.drop(columns=['satisfaction'])

scaler = StandardScaler()
kmeans = KMeans(n_clusters=2)
pipeline = make_pipeline(scaler, kmeans)

pipeline.fit(df_nosat)
labels = pipeline.predict(df_nosat)

df_nosat['cluster'] = labels

# plot
plt.scatter(df_nosat['Inflight wifi service'], df_nosat['Departure/Arrival time convenient'], c=labels)
plt.xlabel('Inflight wifi service')
plt.ylabel('Departure/Arrival time convenient')
plt.savefig('airline-sat-cluster.png')

plt.clf()

# t-sne
from sklearn.manifold import TSNE
model = TSNE(n_components=2, n_jobs=-1, learning_rate=10)
transformed = model.fit_transform(df_nosat)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=labels, alpha=0.5, s=1)
plt.savefig('airline-sat-tsne.png')

# plot original labels with t-sne
plt.clf()
plt.scatter(xs, ys, c=df_sat['satisfaction'], alpha=0.5, s=1)
plt.savefig('airline-sat-tsne-sat.png')

# check accuracy of labels
import numpy as np
print(np.mean(df_sat['satisfaction'] == labels))

