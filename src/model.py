from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def train_model(model, X_train, y_train):
    """
    Trains a given machine learning model on the provided training data.
    
    Parameters:
    - model: scikit-learn compatible model
    - X_train: features for training
    - y_train: target labels for training

    Returns:
    - Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on test data and prints performance metrics.

    Parameters:
    - model: trained model
    - X_test: test features
    - y_test: true test labels

    Returns:
    - accuracy: float
    - y_pred: predictions
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, y_pred


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compares several machine learning models and returns the best one based on accuracy.

    Parameters:
    - X_train, X_test: feature sets
    - y_train, y_test: labels

    Returns:
    - best_model: model with the highest accuracy on test data
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=0),
        'Decision Tree': DecisionTreeClassifier(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'Support Vector Classifier': SVC(probability=True, random_state=0),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    model_accuracies = {}
    model_roc_auc = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_accuracies[name] = acc

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            model_roc_auc[name] = auc
            print(f"\n{name} - Accuracy: {acc:.4f}, AUC-ROC: {auc:.4f}")
        else:
            model_roc_auc[name] = None
            print(f"\n{name} - Accuracy: {acc:.4f} (AUC-ROC not available)")

    # Sort and display model performance by accuracy
    sorted_acc = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
    print("\nModel Accuracy Ranking:")
    for model_name, acc in sorted_acc:
        print(f"{model_name}: {acc:.4f}")

    # Select best model based on accuracy
    best_model_name = sorted_acc[0][0]
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name} with Accuracy: {model_accuracies[best_model_name]:.4f}")
    if model_roc_auc[best_model_name] is not None:
        print(f"AUC-ROC: {model_roc_auc[best_model_name]:.4f}")

    return best_model
