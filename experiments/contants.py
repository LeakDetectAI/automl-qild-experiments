__all__ = ['SYNTHETIC_DATASET', 'MULTI_LAYER_PERCEPTRON', 'SGD_CLASSIFIER', 'RIDGE_CLASSIFIER', 'LINEAR_SVC',
           'DECISION_TREE', 'EXTRA_TREE', 'RANDOM_FOREST', 'EXTRA_TREES', 'ADA_BOOST_CLASSIFIER',
           'GRADIENT_BOOSTING_CLASSIFICATION', 'CLASSIFICATION', 'BAYES_PREDICTOR', 'MAJORITY_VOTING',
           'MUTUAL_INFORMATION', 'MUTUAL_INFORMATION_NEW', 'INFORMEDNESS', 'MCC', 'AUC_SCORE', 'ACCURACY', 'F_SCORE',
           'CONFUSION_MATRIX', 'SANTHIUB', 'HELLMANUB', 'FANOSLB', 'FANOS_ADJUSTEDLB', 'EMI', 'MISCORE', 'AUTO_SKLEARN',
           'AUTO_ML']

SYNTHETIC_DATASET = 'synthetic'
MULTI_LAYER_PERCEPTRON = "mlp"
SGD_CLASSIFIER = "sgd_classifier"
RIDGE_CLASSIFIER = "ridge_classifier"
LINEAR_SVC = "svm"
DECISION_TREE = "decision_tree"
EXTRA_TREE = "extra_tree"
RANDOM_FOREST = "random_forest"
EXTRA_TREES = "extra_trees"
ADA_BOOST_CLASSIFIER = "adaboost"
GRADIENT_BOOSTING_CLASSIFICATION = "gradient_boosting"
BAYES_PREDICTOR = 'bayes_predictor'
MAJORITY_VOTING = 'majority_voting'
AUTO_SKLEARN = 'auto_sklearn'

MUTUAL_INFORMATION = "mutual_information"
MUTUAL_INFORMATION_NEW = "mutual_information_new"
CLASSIFICATION = 'classification'
AUTO_ML = 'automl'
INFORMEDNESS = "Informedness"
MCC = "MathewsCorrelationCoefficient"
AUC_SCORE = "AucScore"
CONFUSION_MATRIX = "ConfusionMatrix"
F_SCORE = "F1Score"
ACCURACY = "Accuracy"
EMI = "EstimatedMutualInformation"
EMI_MIDPOINT = "EstimatedMutualInformationMidPoint"

MISCORE = "EstimatedMutualInformationScore"

SANTHI = "SanthiVardy"
HELLMAN = "HellmanRaviv"
FANOS = "Fanos"
FANOS_ADJUSTED = "FanosAdjusted"
UPPER = "UpperBound"
LOWER = "LowerBound"
SANTHIUB = SANTHI + UPPER
HELLMANUB = HELLMAN + UPPER
FANOSLB = FANOS + LOWER
FANOS_ADJUSTEDLB = FANOS_ADJUSTED + LOWER
