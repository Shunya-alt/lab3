import numpy as np  # <--- ВОТ ЭТА СТРОЧКА БЫЛА ПРОПУЩЕНА

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    ClassificationMetrics,
    RegressionInput,
    RegressionResult,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)

    n_samples, n_features = X_train.shape
    
    # инициализируем веса нулями и смещение нулем
    w = np.zeros(n_features)
    b = 0.0
    
    learning_rate = data.learning_rate
    epochs = data.epochs
    loss_history = []
    
    # цикл обучения (градиентный спуск)
    for epoch in range(epochs):
        # 1 Прямой проход: делаем предсказание
        y_pred = X_train.dot(w) + b
        
        # 2 считаем ошибку (MSE)
        error = y_pred - y_train
        loss = np.mean(error ** 2)
        loss_history.append(float(loss))
        
        # 3 считаем градиенты
        dw = (2.0 / n_samples) * X_train.T.dot(error)
        db = (2.0 / n_samples) * np.sum(error)
        
        # 4 обновляем веса и смещение
        w -= learning_rate * dw
        b -= learning_rate * db
        
    # делаем финальные предсказания для тестовой выборки
    test_predictions = X_test.dot(w) + b
    
    # возвращаем результат
    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        predictions=test_predictions
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    # Переводим в numpy
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)
    
    n_samples, n_features = X_train.shape
    
    # инициализируем веса
    w = np.zeros(n_features)
    b = 0.0
    
    learning_rate = data.learning_rate
    epochs = data.epochs
    loss_history = []
    
    # функция сигмоиды (локальная)
    def sigmoid(z):
        # np.clip защищает от переполнения (чтобы exp не ушел в бесконечность)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    # цикл обучения
    for epoch in range(epochs):
        # 1 линейная часть и сигмоида
        z = X_train.dot(w) + b
        y_pred_prob = sigmoid(z)
        
        # 2 считаем Log Loss (с защитой от log(0))
        epsilon = 1e-15
        y_pred_prob_safe = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        loss = -np.mean(y_train * np.log(y_pred_prob_safe) + (1 - y_train) * np.log(1 - y_pred_prob_safe))
        loss_history.append(float(loss))
        
        # 3 градиенты
        error = y_pred_prob - y_train
        dw = (1.0 / n_samples) * X_train.T.dot(error)
        db = (1.0 / n_samples) * np.sum(error)
        
        # 4 обновляем веса
        w -= learning_rate * dw
        b -= learning_rate * db
        
    # предсказания для теста
    test_probabilities = sigmoid(X_test.dot(w) + b)
    
    # бинарные предсказания: если вероятность >= 0.5, то 1, иначе 0
    test_predictions = (test_probabilities >= 0.5).astype(int)
    
    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    # Переводим в numpy
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)
    y_test = np.array(data.test_target)
    
    # Вспомогательная функция для подсчета метрик
    def evaluate_model(model):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        return ClassificationMetrics(
            roc_auc=roc_auc_score(y_test, probs),
            accuracy=accuracy_score(y_test, preds),
            precision=precision_score(y_test, preds, zero_division=0),
            recall=recall_score(y_test, preds, zero_division=0),
            f1=f1_score(y_test, preds, zero_division=0)
        )

    # Обучаем три модели
    lr_metrics = evaluate_model(LogisticRegression(max_iter=1000))
    dt_metrics = evaluate_model(DecisionTreeClassifier(random_state=42))
    knn_metrics = evaluate_model(KNeighborsClassifier(n_neighbors=5))
    
    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )