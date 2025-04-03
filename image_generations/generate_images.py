import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import os

import matplotlib.pyplot as plt

# 1. Decision Tree Visualization
def plot_decision_tree():
    """
    Trains a DecisionTreeClassifier on the provided dataset and visualizes the decision tree.

    The function fits a decision tree classifier with a maximum depth of 3 using the global 
    variables `X_class` (features) and `y_class` (target labels). It then plots the decision 
    tree with feature names labeled as "Feature 0", "Feature 1", etc., and saves the plot 
    as a PNG image in the "images" directory.

    The output image is titled "Decision Tree" and is saved with the filename 
    "decision_trees.png".

    Note:
        - Ensure that `X_class` and `y_class` are defined and accessible in the global scope.
        - The "images" directory must exist, or the function will raise an error when saving the file.

    Dependencies:
        - matplotlib.pyplot (imported as `plt`)
        - sklearn.tree.DecisionTreeClassifier
        - sklearn.tree.plot_tree
    """
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_class, y_class)
    plt.figure(figsize=(10, 6))
    plot_tree(clf, feature_names=[f"Feature {i}" for i in range(X_class.shape[1])], filled=True)
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig("images/decision_trees.png")
    plt.close()

# 2. Feature Importance (Random Forest)
def plot_feature_importance():
    """
    Trains a Random Forest Classifier on the provided dataset and plots the feature importance.

    The function fits a Random Forest Classifier using the global variables `X_class` (features) 
    and `y_class` (target labels). It calculates the feature importances, creates a bar plot 
    to visualize them, and saves the plot as "images/feature_importance.png".

    Note:
        - This function relies on the global variables `X_class` and `y_class` being defined 
          and accessible in the current scope.
        - The output plot is saved to the "images" directory, which must exist prior to execution.

    Dependencies:
        - matplotlib.pyplot as plt
        - sklearn.ensemble.RandomForestClassifier

    Raises:
        - NameError: If `X_class` or `y_class` is not defined.
        - FileNotFoundError: If the "images" directory does not exist.

    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    importance = rf.feature_importances_
    plt.bar(range(len(importance)), importance, tick_label=[f"Feature {i}" for i in range(len(importance))])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()

# 3. Partial Dependence Plot (PDP)
def plot_pdp():
    """
    Generates and saves a Partial Dependence Plot (PDP) for a Random Forest Classifier.

    This function trains a Random Forest Classifier on the provided dataset (`X_class` and `y_class`),
    computes the partial dependence for the first feature (index 0), and visualizes it using Matplotlib.
    The resulting plot is saved as "images/pdp.png".

    Note:
        - The dataset variables `X_class` and `y_class` must be defined in the global scope.
        - The directory "images" must exist, or the function will raise an error when saving the plot.

    Dependencies:
        - RandomForestClassifier from sklearn.ensemble
        - PartialDependenceDisplay from sklearn.inspection
        - Matplotlib for plotting

    Raises:
        FileNotFoundError: If the "images" directory does not exist.

    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(rf, X_class, [0], ax=ax)
    plt.title("Partial Dependence Plot")
    plt.tight_layout()
    plt.savefig("images/pdp.png")
    plt.close()

# 4. SHAP Values
def plot_shap_values():
    """
    Generates and saves a SHAP summary plot for a Random Forest Classifier.

    This function trains a Random Forest Classifier on the provided dataset,
    computes SHAP values using the TreeExplainer, and creates a summary plot
    of the SHAP values. The plot is saved as a PNG file in the "images" 
    directory with the filename "shap_values.png".

    Note:
        - The function assumes that `X_class` and `y_class` are predefined 
          global variables representing the feature matrix and target vector, 
          respectively.
        - The "images" directory must exist prior to saving the plot.

    Dependencies:
        - matplotlib.pyplot (imported as plt)
        - shap
        - sklearn.ensemble.RandomForestClassifier

    Raises:
        FileNotFoundError: If the "images" directory does not exist.

    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_class)
    shap.summary_plot(shap_values, X_class, show=False)
    plt.title("SHAP Values")
    plt.tight_layout()
    plt.savefig("images/shap_values.png")
    plt.close()

# 5. LIME (Local Surrogate Model)
def plot_lime():
    """
    Generates and saves a LIME (Local Interpretable Model-agnostic Explanations) explanation plot 
    for a Random Forest classifier.

    The function uses the LimeTabularExplainer to explain the predictions of a Random Forest 
    classifier trained on a given dataset. It visualizes the explanation as a matplotlib figure 
    and saves it as a PNG image.

    Steps:
    1. Initializes a LimeTabularExplainer with the provided dataset and feature names.
    2. Trains a Random Forest classifier on the dataset.
    3. Explains the prediction of the first instance in the dataset using LIME.
    4. Generates a matplotlib figure for the explanation.
    5. Saves the figure to the "images" directory as "lime.png".

    Note:
    - The dataset `X_class` and labels `y_class` must be defined globally or passed to the function.
    - The "images" directory must exist in the working directory, or the function will raise an error.

    Saves:
        A PNG image of the LIME explanation plot at "images/lime.png".
    """
    lime_explainer = LimeTabularExplainer(X_class, feature_names=[f"Feature {i}" for i in range(X_class.shape[1])],
                                          class_names=["Class 0", "Class 1"], discretize_continuous=True)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_class, y_class)
    exp = lime_explainer.explain_instance(X_class[0], rf.predict_proba, num_features=5)
    fig = exp.as_pyplot_figure()
    plt.title("LIME Explanation")
    plt.tight_layout()
    plt.savefig("images/lime.png")
    plt.close()

if __name__ == "__main__":    
    # Generate synthetic data
    X_class, y_class = make_classification(n_features=5, random_state=42)
    X_reg, y_reg = make_regression(n_features=5, noise=0.1, random_state=42)
    
    # Create directory for saving images
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    # Generate all images
    plot_decision_tree()
    plot_feature_importance()
    plot_pdp()
    plot_shap_values()
    plot_lime()
