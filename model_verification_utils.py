import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def train_model(train, label, args):
    if not isinstance(train, np.ndarray):
        train = train.to_numpy()
    if not isinstance(label, np.ndarray):
        label = label.to_numpy()

    if args.model == 'LR':

        clf = LogisticRegression(max_iter=1000, random_state=42)

    elif args.model == 'RF':
        clf = RandomForestClassifier(random_state=42)

    else:
        clf = DecisionTreeClassifier(random_state=42)
    clf.fit(train, label.ravel())
    return clf

def generate_explanations(args):
    X_train, y_train, test_case, model = args
    explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode="classification")
    explanation = explainer.explain_instance(test_case, model.predict_proba, top_labels=1)
    return explanation
