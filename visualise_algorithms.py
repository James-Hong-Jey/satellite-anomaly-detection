from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def plot_scatter(df):
    df = df.reset_index()
    plt.figure(figsize=(14,8))
    for column in df.columns:
        # print(column)
        if column not in ['timestamp', 'anomaly', 'anomaly_predicted']:
            plt.scatter(x=df['timestamp'], y=df[column], label=column)

    for i, row in df.iterrows():
        if row['anomaly'] == 1 and 'anomaly_predicted' in row and row['anomaly_predicted'] == 1: # True positive
            plt.axvspan(row['timestamp'], row['timestamp'], color='red', alpha=0.3)
        elif row['anomaly'] == 0 and 'anomaly_predicted' in row and row['anomaly_predicted'] == 1: # False Positive
            plt.axvspan(row['timestamp'], row['timestamp'], color='green', alpha=0.3)
        elif row['anomaly'] == 1 and 'anomaly_predicted' in row and row['anomaly_predicted'] == 0: # False Negative
            plt.axvspan(row['timestamp'], row['timestamp'], color='purple', alpha=0.3)

    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_3d(inliers, outliers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    firmware_names = inliers.drop(columns='anomaly').columns.tolist()
    # Only accepts 3, higher dimensions need tSNE plot
    if(len(firmware_names) > 4):
        return None

    ax.scatter(inliers[firmware_names[0]], 
            inliers[firmware_names[1]],
            inliers[firmware_names[2]],
            c='blue', 
            label='Inliers')

    ax.scatter(outliers[firmware_names[0]], 
            outliers[firmware_names[1]],
            outliers[firmware_names[2]],
            c='red', 
            label='Outliers')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    print("Conventional 3D Plot:")
    plt.show()

def plot_tsne(df, drop=['timestamp', 'anomaly'], perplexity=5):
    anomaly = df['anomaly'].values
    anomaly_predicted = df['anomaly_predicted'].values
    tsne_df = df.drop(drop, axis=1).reset_index(drop=True)
    # m = TSNE(n_components=3, perplexity=perplexity)
    m = TSNE(n_components=2, perplexity=perplexity)
    tsne_features = m.fit_transform(tsne_df.values)
    colors = ['red' if x == 1 and y == 1 else 'purple' if x == 1 and y == 0 else 'green' if x == 0 and y == 1 else 'blue' for x, y in zip(anomaly, anomaly_predicted)]
    plt.scatter(tsne_features[:,0], tsne_features[:,1], c=colors)
    print("TSNE 2D Plot:")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # colors = ['red' if x == 1 and y == 1 else 'purple' if x == 1 and y == 0 else 'green' if x == 0 and y == 1 else 'blue' for x, y in zip(anomaly, anomaly_predicted)]
    # colors = ['red' if x == 1 else 'blue' for x in anomaly]
    m = TSNE(n_components=3, perplexity=perplexity)
    tsne_features = m.fit_transform(tsne_df.values)
    ax.scatter(tsne_features[:,0], tsne_features[:,1], tsne_features[:,2], c=colors, marker='o')
    print("TSNE 3D Plot:")
    plt.show()

def plot_pca(df, drop=['timestamp', 'anomaly'], alpha=0.7):
    anomaly = df['anomaly'].values
    anomaly_predicted = df['anomaly_predicted'].values
    df_features = df.drop(drop, axis=1).reset_index(drop=True)
    norm_df = MinMaxScaler().fit_transform(df_features)
    print("PCA 2D Plot:")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(norm_df)
    pca_df = pd.DataFrame(data=principal_components, columns = ['PC1', 'PC2'])
    pca_df['anomaly'] = anomaly
    pca_df['anomaly_predicted'] = anomaly_predicted
    # pca_df['error_type'] = 'true_positive' if pca_df['anomaly'] == 1 and pca_df['anomaly_predicted'] == 1 else 'true_negative' if pca_df['anomaly'] == 0 and pca_df['anomaly_predicted'] == 0 else 'false_positive' if pca_df['anomaly'] == 0 and pca_df['anomaly_predicted'] == 1 else 'false_negative' if pca_df['anomaly'] == 1 and pca_df['anomaly_predicted'] == 0 else 'null'
    pca_df['error_type'] = ['true_positive' if x == 1 and y == 1 else 'false_negative' if x == 1 and y == 0 else 'false_positive' if x == 0 and y == 1 else 'true_negative' for x, y in zip(anomaly, anomaly_predicted)]
    plt.figure
    plt.figure(figsize=(10, 7))
    # colors = {0: 'blue', 1: 'red'}
    colors = {'true_positive': 'red', 'true_negative': 'blue', 'false_positive': 'green', 'false_negative': 'purple'}
    plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                c=pca_df['error_type'].map(colors), alpha=alpha)

    plt.title('PCA of Temperature Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show() 