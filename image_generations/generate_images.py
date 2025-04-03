import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt

# Generate synthetic data
X_class, y_class = make_classification(n_features=5, random_state=42)
X_reg, y_reg = make_regression(n_features=5, noise=0.1, random_state=42)

# 1. Decision Tree Visualization
def plot_decision_tree():
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_class, y_class)
    plt.figure(figsize=(10, 6))
    plot_tree(clf, feature_names=[f"Feature {i}" for i in range(X_class.shape[1])], filled=True)
    plt.title("Decision Tree")
    plt.savefig("images/decision_trees.png")
    plt.close()

# 2. Feature Importance (Random Forest)
def plot_feature_importance():
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    importance = rf.feature_importances_
    plt.bar(range(len(importance)), importance, tick_label=[f"Feature {i}" for i in range(len(importance))])
    plt.title("Feature Importance")
    plt.savefig("images/feature_importance.png")
    plt.close()

# 3. Partial Dependence Plot (PDP)
def plot_pdp():
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(rf, X_class, [0], ax=ax)
    plt.title("Partial Dependence Plot")
    plt.savefig("images/pdp.png")
    plt.close()

# 4. SHAP Values
def plot_shap_values():
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_class)
    shap.summary_plot(shap_values[1], X_class, show=False)
    plt.title("SHAP Values")
    plt.savefig("images/shap_values.png")
    plt.close()

# 5. LIME (Local Surrogate Model)
def plot_lime():
    lime_explainer = LimeTabularExplainer(X_class, feature_names=[f"Feature {i}" for i in range(X_class.shape[1])],
                                          class_names=["Class 0", "Class 1"], discretize_continuous=True)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    exp = lime_explainer.explain_instance(X_class[0], rf.predict_proba, num_features=5)
    fig = exp.as_pyplot_figure()
    plt.title("LIME Explanation")
    plt.savefig("images/lime.png")
    plt.close()

# Generate all images
plot_decision_tree()
plot_feature_importance()
plot_pdp()
plot_shap_values()
plot_lime()
