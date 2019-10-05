import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA
from sklearn.manifold import LocallyLinearEmbedding
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
ops.reset_default_graph()

st.title('Mushrooms Feature Engineering')


@st.cache(ignore_hash=True)
def load_data(nrows):
    df = pd.read_csv('mushrooms.csv', nrows=nrows)
    pd.options.display.max_columns = None
    return df


def forest_test(X, Y, plot_name = ''):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
    predictionforest = trainedforest.predict(X_Test)
    st.subheader(plot_name + ' Accuracy = ' + str(round(accuracy_score(Y_Test,predictionforest),3)*100) + '%')


def complete_test_2D(X, Y, plot_name = ''):
    Small_df = pd.DataFrame(data = X, columns = ['C1', 'C2'])
    Small_df = pd.concat([Small_df, df['class']], axis = 1)
    Small_df['class'] = LabelEncoder().fit_transform(Small_df['class'])
    forest_test(X, Y, plot_name)
    data = []
    for clas, col, name in zip((1, 0), ['red', 'darkblue'], ['Poisonous', 'Edible']):

        trace = dict(
            type='scatter',
            x= Small_df.loc[Small_df['class'] == clas, 'C1'],
            y= Small_df.loc[Small_df['class'] == clas, 'C2'],
            mode= 'markers',
            name= name,
            marker=dict(
                color=col,
                size=12,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8)
        )
        data.append(trace)

    layout = dict(
            title= plot_name + ' Dimensionality Reduction',
            xaxis=dict(title='C1', showline=False),
            yaxis=dict(title='C2', showline=False)
    )
    fig = dict(data=data, layout=layout)
    st.plotly_chart(fig)


def complete_test_3D(X, Y, plot_name = ''):
    Small_df = pd.DataFrame(data = X, columns = ['C1', 'C2', 'C3'])
    Small_df = pd.concat([Small_df, df['class']], axis = 1)
    Small_df['class'] = LabelEncoder().fit_transform(Small_df['class'])
    forest_test(X, Y, plot_name)
    data = []
    for clas, col, name in zip((1, 0), ['red', 'darkblue'], ['Poisonous', 'Edible']):

        trace = dict(
            type='scatter3d',
            x= Small_df.loc[Small_df['class'] == clas, 'C1'],
            y= Small_df.loc[Small_df['class'] == clas, 'C2'],
            z= Small_df.loc[Small_df['class'] == clas, 'C3'],
            mode= 'markers',
            name= name
        )
        data.append(trace)

    layout = {
        "scene": {
          "xaxis": {
            "title": "C1",
            "showline": False
          },
          "yaxis": {
            "title": "C2",
            "showline": False
          },
          "zaxis": {
            "title": "C3",
            "showline": False
          }
        },
        "title": plot_name + ' Dimensionality Reduction'
    }
    fig = dict(data=data, layout=layout)
    st.plotly_chart(fig)


st.write('In this exercise, I am going to examine the Kaggle Mushrooms Dataset to predict '
         'if a Muchroom is Poisonous or Edible. This will be done by using just Feature Engineering '
         'Techniques such as PCA and Autoencoders '
         'in order to reduce our number of features from +20 to just 3. A Random Forest Classifier will be '
         'used to test our models accuracy throughout this whole notebook.')

st.write('This web app was created using the Streamlit library')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df = load_data(8000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# st.write('Done! (using st.cache)')

st.subheader('Raw data')
st.write(df)


st.subheader('Edible vs Poisonous Classes')
sns.set(style="ticks")
f = sns.countplot(x="class", data=df, palette="bwr")
st.pyplot()

X = df.drop(['class'], axis = 1)
X = pd.get_dummies(X, prefix_sep='_')
Y = df['class']
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)


# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
complete_test_3D(X_pca, Y, '3D PCA')

var_ratio = pca.explained_variance_ratio_
cum_var_ratio = np.cumsum(var_ratio)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,5)],
    y=var_ratio,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,5)],
    y=cum_var_ratio,
    name='Cumulative'
)

data = [trace1, trace2]

layout=dict(
    title='Explained variance Ratio by each principal components',
    yaxis=dict(
        title='Explained variance ratio in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)
st.plotly_chart(fig)


# LDA
def lda(X, Y):
    lda = LinearDiscriminantAnalysis(n_components=1)
    # run an LDA and use it to transform the features
    X_lda = lda.fit(X, Y).transform(X)

    forest_test(X_lda, Y, '1D LDA')

    LDA_df = pd.DataFrame(data = X_lda, columns = ['LDA1'])
    LDA_df = pd.concat([LDA_df, df['class']], axis = 1)
    LDA_df['class'] = LabelEncoder().fit_transform(LDA_df['class'])

    figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    sns.distplot(LDA_df.loc[LDA_df['class'] == 0]['LDA1'], label = 'Edible', hist=True, kde=True, rug=True)
    sns.distplot(LDA_df.loc[LDA_df['class'] == 1]['LDA1'], label = 'Poisonous', hist=True, kde=True, rug=True)
    st.pyplot()
    sns.jointplot(x="LDA1", y="class", data=LDA_df, kind="kde")
    st.pyplot()


lda(X, Y)

# ICA
ica = FastICA(n_components=3)
X_ica = ica.fit_transform(X)
complete_test_3D(X_ica, Y, '3D ICA')

# Locally Linear Embedding
embedding = LocallyLinearEmbedding(n_components=3, eigen_solver='dense')
X_lle = embedding.fit_transform(X)
complete_test_3D(X_lle, Y, '3D LLE')

# Autoencoders
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(3, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='softmax')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)

autoencoder.fit(X1, Y1,
                epochs=100,
                batch_size=300,
                shuffle=True,
                verbose = 0,
                validation_data=(X2, Y2))

encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
complete_test_3D(X_ae, Y, '3D AE')

# t-SNE
tsne = TSNE(n_components=3, verbose=1, perplexity=20, n_iter=250)
X_tsne = tsne.fit_transform(X)
complete_test_3D(X_tsne, Y, '3D t-SNE')


# st.subheader('Features Distribution')
# feat = st.slider('selected_feature', 1, 22, 1)
# f = sns.countplot(x=df.iloc[:, feat], data=df, palette="bwr")
# st.pyplot()
