import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

dataset_len = 180
dlen = int(dataset_len/2)
X_11 = pd.Series(np.random.normal(100,2,dlen))
X_12 = pd.Series(np.random.normal(90,2,dlen))
X_1 = pd.concat([X_11, X_12]).reset_index(drop=True)
X_21 = pd.Series(np.random.normal(10,30,dlen))
X_22 = pd.Series(np.random.normal(7,30,dlen))
X_2 = pd.concat([X_21, X_22]).reset_index(drop=True)
X_31 = pd.Series(np.random.normal(22,74,dlen))
X_32 = pd.Series(np.random.normal(52,52,dlen))
X_3 = pd.concat([X_31, X_32]).reset_index(drop=True)
Y_1 = pd.Series(np.random.normal(40,3,dlen))
Y_2 = pd.Series(np.random.normal(82,7,dlen))
Y = pd.concat([Y_1, Y_2]).reset_index(drop=True)
df = pd.concat([X_1, X_2, X_3, Y], axis=1)
df.columns = ['Advertisement Spent', 'Brand Equity Score', 'Market Share', 'Sales']

X = df.drop(['Sales'], axis = 1).values
Y = df['Sales']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(X_Train, Y_Train)

st.title('Business Insights')
st.subheader('Financial Analysis')
st.write('Welcome to our business dashboard, here you can analyse how'
         'changing one our 3 model parameters can affect will affect the predicted sales.')

ads = st.text_input('Advertisement Spent')
brand = st.text_input('Brand Equity Score')
market = st.text_input('Market Share')
ads = list(map(float, ads.split()))
brand = list(map(float, brand.split()))
market = list(map(float, market.split()))

if (ads and brand and market) and len(ads)==len(brand)==len(market):
    st.write("Thanks for adding input source")
    X_Test = np.array([ads, brand, market])
    X_Test = np.array(list(map(list, zip(*X_Test))))
else:
    X_Test = X_Test

st.write(df.head())

st.subheader('Feature Importance')
figure(num=None, figsize=(20, 22), dpi=80, facecolor='w', edgecolor='k')
feat_importances = pd.Series(regr.feature_importances_, index= df.drop(['Sales'], axis = 1).columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xticks(size = 15)
plt.yticks(size = 15)
st.pyplot()

#st.write(feat_importances.nlargest(10).index[0])

pred = regr.predict(X_Test)

st.subheader('Bayesian Belief Network')

df2 = pd.DataFrame({'from': ['Advertisement \nSpent',  'Brand \nEquity \nScore',
                             'Market \nShare'],
                   'to': ['Sales', 'Sales', 'Sales']})

carac = pd.DataFrame({'ID': ['Advertisement \nSpent',  'Brand \nEquity \nScore',
                             'Market \n Share', 'Sales'],
                      'values': ['group1', 'group2', 'group3', 'group4']})

G = nx.from_pandas_edgelist(df2, 'from', 'to', create_using=nx.Graph())
G.nodes()
carac = carac.set_index('ID')
carac = carac.reindex(G.nodes())
carac['values'] = pd.Categorical(carac['values'])

cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))
         }

colours = LinearSegmentedColormap('GYR', cdict)
fixed_positions = {'Advertisement \nSpent' : (0, 0),
                   'Brand \nEquity \nScore' : (-1, 2),
                   'Market \nShare' : (1, 2),
                   'Sales': (0, 1.2)}
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)
a = [int(np.mean(X_Test[:, 0])), int(np.mean(pred)), int(np.mean(X_Test[:, 1])), int(np.mean(X_Test[:, 2]))]


#st.write("Prediction Accuracy: ", regr.score(X_Test, Y_Test))
fig = plt.figure()
nx.draw(G, with_labels=True, node_color=carac['values'].cat.codes, cmap=colours, pos=pos,
        node_size=[v*200 for v in a], width=3)
plt.annotate(str(round(np.mean(X_Test[:, 1]), 3)), xy=(200, 880), xycoords='figure pixels', fontsize=15)
plt.annotate(str(round(np.mean(X_Test[:, 0]), 3)), xy=(220, 50), xycoords='figure pixels', fontsize=15)
plt.annotate(str(round(np.mean(X_Test[:, 2]), 3)), xy=(880, 900), xycoords='figure pixels', fontsize=15)
plt.annotate(str(round(np.mean(pred), 3)), xy=(300, 470), xycoords='figure pixels', fontsize=15)
st.pyplot()