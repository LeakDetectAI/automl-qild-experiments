from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE

n_reduced = 15
# Create a dictionary to store the techniques and their options
reduction_techniques = {
    'select_from_model_et': SelectFromModel(ExtraTreesClassifier(), max_features=n_reduced),
    'select_from_model_rf': SelectFromModel(RandomForestClassifier(), max_features=n_reduced),
    'pca': PCA(n_components=n_reduced),
    'lda': LinearDiscriminantAnalysis(n_components=n_reduced),
    'tsne': TSNE(n_components=n_reduced),
    'nmf': NMF(n_components=n_reduced)
}
