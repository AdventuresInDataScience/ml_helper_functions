####
# Below is a Python function that encapsulates the knowledge distillation process from a complex base model 
# (like XGBoost) to a simpler new model (like a single decision tree). The function allows you to specify the 
# input data, random seed, train-test split percentage, base model, and new model.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def knowledge_distillation(X, y, random_seed, split_percentage, base_model, new_model, base_model_fit = False):
    """
    Perform knowledge distillation from a trained base model to a new model.
    
    Parameters:
    - X: Feature set
    - y: Target variable
    - random_seed: Random seed for reproducibility
    - split_percentage: Train-test split percentage
    - base_model: Instance of the base model (e.g., XGBoost). Can be an unfit base model, or an already fit model
    - new_model: Instance of the new model (e.g., Decision Tree)
    - base_model_fit: If set to False, then base model is treated as unfit, and is fit to new data. If True, then
    assumes model is already fit, and function will not attempt to retrain base model
    
    Returns:
    - base_model_accuracy: Accuracy of the base model on the test set
    - new_model_accuracy: Accuracy of the new model on the test set
    
    Explanation:
    - This function is a generalized approach to knowledge distillation. It first splits the data into training 
    and test sets, then trains the base model, generates soft labels (predictions) from the base model, trains 
    the new model on these soft labels, and finally evaluates both models on the test set.
    
    Example:
    # Example usage
    from sklearn.datasets import load_iris
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Load data
    X, y = load_iris(return_X_y=True)

    # Define models
    base_model = XGBClassifier(random_state=42)
    new_model = DecisionTreeClassifier(random_state=42)

    # Perform knowledge distillation
    base_acc, new_acc = knowledge_distillation(X, y, random_seed=42, split_percentage=0.8, base_model=base_model, new_model=new_model)

    print(f"Base Model Accuracy: {base_acc:.2f}")
    print(f"New Model Accuracy: {new_acc:.2f}")

    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_percentage), random_state=random_seed)

    # If the function receives an untrained model, then it fits the base model
    if base_model_fit == False:
        base_model.fit(X_train, y_train)

    # Generate "soft labels" using the base model
    soft_labels = base_model.predict(X_train)

    # Train the new model using the soft labels
    new_model.fit(X_train, soft_labels)

    # Evaluate both models on the test set
    base_model_preds = base_model.predict(X_test)
    new_model_preds = new_model.predict(X_test)

    base_model_accuracy = accuracy_score(y_test, base_model_preds)
    new_model_accuracy = accuracy_score(y_test, new_model_preds)

    return base_model_accuracy, new_model_accuracy



