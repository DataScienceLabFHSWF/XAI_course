import os
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer

# SHAP library
import shap

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
# Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# PyTorch imports
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from captum.attr import Saliency, IntegratedGradients

# Hugging Face Transformers
from transformers import pipeline

# PIL for image handling
from PIL import Image
import time
from transformers import AutoTokenizer, AutoModel, utils

# BertViz imports
from bertviz import head_view
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

from tqdm import tqdm  # Import tqdm for progress tracking

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
    # Generate synthetic data
    X_class, y_class = make_classification(n_features=5, random_state=42)

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
    # Generate synthetic data
    X_class, y_class = make_classification(n_features=5, random_state=42)
    X_reg, y_reg = make_regression(n_features=5, noise=0.1, random_state=42)

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
    # Generate synthetic data
    X_class, y_class = make_classification(n_features=5, random_state=42)
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
    # Generate synthetic data
    X_class, y_class = make_classification(n_features=5, random_state=42)
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
    # Generate synthetic data
    X_class, y_class = make_classification(n_features=5, random_state=42)
    
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

    # 6. CNN Feature Visualization
def plot_cnn_feature_maps():
    """
    Trains a simple CNN on the MNIST dataset and visualizes feature maps for a sample image.

    This function trains a CNN on the MNIST dataset, extracts feature maps from the first convolutional
    layer for a sample image, and saves the visualization as "images/cnn_feature_maps.png".

    Note:
        - The "images" directory must exist prior to saving the plot.

    Dependencies:
        - TensorFlow/Keras
        - Matplotlib

    Raises:
        FileNotFoundError: If the "images" directory does not exist.
    """
    # Load MNIST dataset
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train)

    # Define a simple CNN
    model = Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        Conv2D(8, (3, 3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0)

    # Extract feature maps
    sample_image = X_train[0:1]
    feature_map_model = Sequential(model.layers[:1])  # First convolutional layer
    feature_maps = feature_map_model.predict(sample_image)

    # Plot feature maps
    fig, axes = plt.subplots(1, feature_maps.shape[-1], figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.suptitle("CNN Feature Maps")
    plt.tight_layout()
    plt.savefig("images/cnn_feature_maps.png")
    plt.close()

# 7. LLM Explanation (Sentiment Analysis)
def plot_llm_explanation():
    """
    Generates and saves an explanation for a sentiment analysis task using a pre-trained LLM.

    This function uses the Hugging Face Transformers pipeline to perform sentiment analysis on a
    sample text and visualizes the explanation (e.g., token importance) as a bar plot. The plot
    is saved as "images/llm_explanation.png".

    Note:
        - The "images" directory must exist prior to saving the plot.

    Dependencies:
        - Transformers library
        - Matplotlib

    Raises:
        FileNotFoundError: If the "images" directory does not exist.
    """
    # Define sample text
    sample_text = "The movie was absolutely fantastic and I loved every moment of it!"

    # Use Hugging Face pipeline for sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(sample_text)

    # Tokenize and compute token importance (mock example)
    tokens = sample_text.split()
    importance = [len(token) for token in tokens]  # Mock importance based on token length

    # Plot token importance
    plt.bar(tokens, importance, color='skyblue')
    plt.title("LLM Explanation (Token Importance)")
    plt.xlabel("Tokens")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("images/llm_explanation.png")
    plt.close()

# 8. Saliency Map
def plot_saliency_map():
    """
    Generates and saves a saliency map for a CNN trained on the MNIST dataset.

    This function trains a simple CNN on the MNIST dataset, computes the gradient of the output
    with respect to the input image to generate a saliency map, and saves the visualization as
    "images/saliency_map.png".

    Note:
        - The "images" directory must exist prior to saving the plot.

    Dependencies:
        - TensorFlow/Keras
        - Matplotlib

    Raises:
        FileNotFoundError: If the "images" directory does not exist.
    """
    # Load CIFAR-10 dataset using PyTorch utilities
    cifar10_train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=256, shuffle=False
    )

    # Get the first image and label from the dataset
    sample_image, y_train = next(iter(cifar10_train))
    sample_image = sample_image[0].unsqueeze(0)  # Select the first image and add batch dimension
    y_train = y_train[0]  # Select the corresponding label

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the sample image
    sample_image = preprocess(sample_image)

    # Define a pre-trained model (e.g., ResNet18) and fine-tune it on CIFAR-10
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust the final layer for CIFAR-10 (10 classes)
    model = model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Check if the fine-tuned weights exist
    if not os.path.exists('cifar10_resnet18.pth'):
        # Fine-tune the model on CIFAR-10
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop (quick fine-tuning for a few epochs)
        
        model = model.to(device)  # Move model to GPU if available
        criterion = criterion.to(device)  # Move criterion to GPU if available

        for epoch in range(1):  # Only 1 epoch for quick fine-tuning
            model.train()
            epoch_loss = 0.0
            with tqdm(cifar10_train, desc=f"Epoch {epoch + 1}/x", unit="batch") as progress_bar:
                for images, labels in progress_bar:
                    images = preprocess(images).to(device)  # Preprocess and move to device
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=epoch_loss / (progress_bar.n + 1))
        # Save the fine-tuned weights
        torch.save(model.state_dict(), 'cifar10_resnet18.pth')

    # Load the fine-tuned weights
    model.load_state_dict(torch.load('cifar10_resnet18.pth'))

    # Compute saliency map using Captum
    saliency = Saliency(model)
    sample_image.requires_grad_()  # Enable gradients for the input
    sample_image = sample_image.to(device)  # Move to device
    model = model.to(device)  # Ensure model is on the same device as the input
    output = model(sample_image)
    target_class = torch.argmax(output, dim=1)
    saliency_map = saliency.attribute(sample_image, target=target_class.item()).squeeze().cpu().detach().numpy()

    # Plot the image and saliency map side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(sample_image[0].permute(1, 2, 0).cpu().detach().numpy() * 0.5 + 0.5)  # Denormalize for visualization
    axes[0].set_title(f"Original Image (Label: {cifar10_train.dataset.classes[y_train]})")
    axes[0].axis('off')

    # Saliency map
    axes[1].imshow(saliency_map.mean(axis=0), cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("images/saliency_map.png")
    plt.close()

# 9. Integrated Gradients with Captum
def plot_integrated_gradients():
    """
    Generates and saves an Integrated Gradients visualization for a CNN trained on the CIFAR-10 dataset.

    This function uses a pre-trained ResNet18 model fine-tuned on CIFAR-10, computes the Integrated Gradients
    for a sample image, and saves the visualization as "images/integrated_gradients.png".

    Note:
        - The "images" directory must exist prior to saving the plot.

    Dependencies:
        - PyTorch
        - Captum
        - Matplotlib

    Raises:
        FileNotFoundError: If the "images" directory does not exist.
    """
    # Load CIFAR-10 dataset using PyTorch utilities
    cifar10_train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=256, shuffle=False
    )

    # Get the first image and label from the dataset
    sample_image, y_train = next(iter(cifar10_train))
    sample_image = sample_image[0].unsqueeze(0)  # Select the first image and add batch dimension
    y_train = y_train[0]  # Select the corresponding label

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the sample image
    sample_image = preprocess(sample_image)

    # Define a pre-trained model (e.g., ResNet18) and fine-tune it on CIFAR-10
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust the final layer for CIFAR-10 (10 classes)
    model = model.eval()  # Set model to evaluation mode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Check if the fine-tuned weights exist
    if not os.path.exists('cifar10_resnet18.pth'):
        # Fine-tune the model on CIFAR-10
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop (quick fine-tuning for a few epochs)
        model = model.to(device)  # Move model to GPU if available
        criterion = criterion.to(device)  # Move criterion to GPU if available

        for epoch in range(1):  # Only 1 epoch for quick fine-tuning
            model.train()
            epoch_loss = 0.0
            with tqdm(cifar10_train, desc=f"Epoch {epoch + 1}/x", unit="batch") as progress_bar:
                for images, labels in progress_bar:
                    images = preprocess(images).to(device)  # Preprocess and move to device
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=epoch_loss / (progress_bar.n + 1))
        # Save the fine-tuned weights
        torch.save(model.state_dict(), 'cifar10_resnet18.pth')

    # Load the fine-tuned weights
    model.load_state_dict(torch.load('cifar10_resnet18.pth'))

    # Compute Integrated Gradients using Captum
    ig = IntegratedGradients(model)
    sample_image.requires_grad_()  # Enable gradients for the input
    sample_image = sample_image.to(device)  # Move to device
    model = model.to(device)  # Ensure model is on the same device as the input
    output = model(sample_image)
    target_class = torch.argmax(output, dim=1)
    attributions = ig.attribute(sample_image, target=target_class.item(), n_steps=50).squeeze().cpu().detach().numpy()

    # Plot the original image and attributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(sample_image[0].permute(1, 2, 0).cpu().detach().numpy() * 0.5 + 0.5)  # Denormalize for visualization
    axes[0].set_title(f"Original Image (Label: {cifar10_train.dataset.classes[y_train]})")
    axes[0].axis('off')

    # Integrated Gradients
    axes[1].imshow(attributions.mean(axis=0), cmap='hot')
    axes[1].set_title("Integrated Gradients")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("images/integrated_gradients.png")
    plt.close()

# 10. LLM Interpretation with BertViz
def plot_llm_interpretation_with_bertviz():
    """
    Generates and saves an LLM interpretation visualization using BertViz.

    This function uses the Hugging Face Transformers library to load a pre-trained BERT model,
    computes attention weights for a sample text, and visualizes the attention using BertViz's
    head_view. The visualization is saved as an HTML file in the "images" directory.

    Note:
        - The "images" directory must exist prior to saving the file.

    Dependencies:
        - Transformers library
        - BertViz

    Saves:
        An HTML file of the BertViz head view at "images/bertviz_llm_interpretation.html".
    """

    # Suppress standard warnings
    utils.logging.set_verbosity_error()

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

    # Define sample text
    sample_text = "The quick brown fox jumps over the lazy dog."

    # Tokenize input and compute attention
    inputs = tokenizer.encode(sample_text, return_tensors='pt')
    outputs = model(inputs)
    attention = outputs[-1]  # Output includes attention weights when output_attentions=True
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    # Generate BertViz head view
    html_head_view = head_view(attention, tokens, html_action='return')

    # Save the visualization as an HTML file
    output_path = "images/bertviz_llm_interpretation.html"
    with open(output_path, 'w') as file:
        file.write(html_head_view.data)
    # Generate BertViz neuron view
    model_type = 'bert'
    model_version = 'bert-base-uncased'
    do_lower_case = True
    sentence_a = "The cat sat on the mat"
    sentence_b = "The cat lay on the rug"

    # Load model and tokenizer for neuron view
    model = BertModel.from_pretrained(model_version, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

    # Generate neuron view visualization
    html_neuron_view = show(model, model_type, tokenizer, sentence_a, sentence_b, layer=2, head=0, html_action='return')

    # Save the visualization as an HTML file
    neuron_view_output_path = "images/bertviz_neuron_view.html"
    with open(neuron_view_output_path, 'w') as file:
        file.write(html_neuron_view.data)


def is_file_outdated(file_path, days=7):
    """
    Check if a file is outdated based on the last modification time.

    Args:
        file_path (str): Path to the file.
        days (int): Number of days to consider as the threshold.

    Returns:
        bool: True if the file is outdated or doesn't exist, False otherwise.
    """
    if not os.path.exists(file_path):
        return True
    last_modified_time = os.path.getmtime(file_path)
    return (time.time() - last_modified_time) > (days * 86400)

if __name__ == "__main__":  
    # Create directory for saving images
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    

    # Generate all images only if they don't exist or are outdated
    if is_file_outdated("images/decision_trees.png"):
        print("Generating Decision Tree visualization...")
        plot_decision_tree()
        print("Decision Tree visualization saved.")

    if is_file_outdated("images/feature_importance.png"):
        print("Generating Feature Importance plot...")
        plot_feature_importance()
        print("Feature Importance plot saved.")

    if is_file_outdated("images/pdp.png"):
        print("Generating Partial Dependence Plot (PDP)...")
        plot_pdp()
        print("Partial Dependence Plot saved.")

    if is_file_outdated("images/shap_values.png"):
        print("Generating SHAP Values plot...")
        plot_shap_values()
        print("SHAP Values plot saved.")

    if is_file_outdated("images/lime.png"):
        print("Generating LIME explanation plot...")
        plot_lime()
        print("LIME explanation plot saved.")

    if is_file_outdated("images/cnn_feature_maps.png"):
        print("Generating CNN Feature Maps visualization...")
        plot_cnn_feature_maps()
        print("CNN Feature Maps visualization saved.")

    if is_file_outdated("images/llm_explanation.png"):
        print("Generating LLM Explanation plot...")
        plot_llm_explanation()
        print("LLM Explanation plot saved.")

    if is_file_outdated("images/saliency_map.png"):
        print("Generating Saliency Map...")
        plot_saliency_map()
        print("Saliency Map saved.")

    if is_file_outdated("images/integrated_gradients.png"):
        print("Generating Integrated Gradients plot...")
        plot_integrated_gradients()
        print("Integrated Gradients plot saved.")

    if is_file_outdated("images/bertviz_llm_interpretation.hmtl"):
        print("Generating LLM Interpretation with BertViz...")
        plot_llm_interpretation_with_bertviz()
        print("LLM Interpretation with BertViz saved.")
    # Print completion message
    print("All images generated and saved in the 'images' directory.")
