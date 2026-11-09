import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Импорты твоих функций
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models
from grader_contracts.linear_models import RegressionInput, ClassificationInput

def run_all():
    results = {"linear_regression": {}, "classification": {}}

    #1 ЛИНЕЙНАЯ РЕГРЕССЦИЯ
    print("1 ЛИНЕЙНАЯ РЕГРЕССИЯ (Student_Performance.csv)")
    try:
        df_reg = pd.read_csv("Student_Performance.csv")
        
        #подготовка: Yes/No -> 1/0
        df_reg["Extracurricular Activities"] = df_reg["Extracurricular Activities"].map({"Yes": 1, "No": 0})
        X = df_reg.drop(columns=["Performance Index"]).values
        y = df_reg["Performance Index"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #масштабирование 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        reg_data = RegressionInput(X_train_scaled, y_train, X_test_scaled, y_test, learning_rate=0.05, epochs=1000)
        reg_res = train_linear_regression(reg_data)
        test_mse = mean_squared_error(y_test, reg_res.predictions)
        results["linear_regression"]["test_mse"] = float(test_mse)
        
        print(f"Итоговый MSE на тесте: {test_mse:.4f}")
        print(f"Первые 5 предсказаний: {reg_res.predictions[:5].round(2)}")
        print(f"Первые 5 реальных ответов: {y_test[:5].round(2)}\n")
    except FileNotFoundError:
        print("Ошибка: Файл Student_Performance.csv не найден в папке!\n")

    #2 КЛАССИФИКАЦИЯ
    print("\n2 КЛАССИФИКАЦИЯ (german.csv)")
    try:
        df_cls = pd.read_csv("german.csv", sep=";")
        
        #ищем целевую колонку
        target_col = None
        for col in df_cls.columns:
            if "credit" in str(col).lower():
                target_col = col
                break
                
        if target_col is None:
            target_col = df_cls.columns[0]  # Если не нашли по имени, берем первую
            
        print(f"Целевая колонка: {target_col}")
        
        #кодируем категориальные признаки
        df_encoded = pd.get_dummies(df_cls, drop_first=True)
        
        #ищем закодированное имя целевой колонки (может измениться после get_dummies)
        true_target_col = [c for c in df_encoded.columns if target_col in c][0]
        y_cls = df_encoded[true_target_col].values
        
        #проверяем тип данных и переводим в числа, если это строки
        if y_cls.dtype == 'object' or y_cls.dtype == 'bool':
            y_cls = pd.factorize(y_cls)[0]
        elif y_cls.min() == 1:
            y_cls = y_cls - 1
        
        #проверяем, что у нас два класса
        unique_classes = np.unique(y_cls)
        print(f"Уникальные классы в y: {unique_classes}")
        if len(unique_classes) < 2:
            raise ValueError("В данных только один класс! Проверьте CSV файл.")
        X_cls = df_encoded.drop(columns=[true_target_col]).values
        
        #делаем стратифицированное разбиение (чтобы классы были в тесте и трейне)
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )
        
        #масштабирование
        scaler_cls = StandardScaler()
        Xc_train_scaled = scaler_cls.fit_transform(Xc_train)
        Xc_test_scaled = scaler_cls.transform(Xc_test)
        cls_data = ClassificationInput(Xc_train_scaled, yc_train, Xc_test_scaled, yc_test, learning_rate=0.1, epochs=1000)
        
        #2.1 самописная модель
        own_res = train_logistic_classifier(cls_data)
        results["classification"]["own_logistic"] = {
            "roc_auc": float(roc_auc_score(yc_test, own_res.probabilities)),
            "accuracy": float(accuracy_score(yc_test, own_res.predictions)),
            "precision": float(precision_score(yc_test, own_res.predictions, zero_division=0)),
            "recall": float(recall_score(yc_test, own_res.predictions, zero_division=0)),
            "f1": float(f1_score(yc_test, own_res.predictions, zero_division=0))
        }
        
        #2.2 готовые модели из sklearn
        baseline_res = compare_classical_models(cls_data)
        
        def metrics_to_dict(m):
            return {"roc_auc": float(m.roc_auc), "accuracy": float(m.accuracy), "precision": float(m.precision), "recall": float(m.recall), "f1": float(m.f1)}
            
        results["classification"]["logistic_regression"] = metrics_to_dict(baseline_res.logistic_regression)
        results["classification"]["decision_tree"] = metrics_to_dict(baseline_res.decision_tree)
        results["classification"]["knn"] = metrics_to_dict(baseline_res.knn)
        
        # Вывод для отчета
        print("Сравнение моделей (Метрика Accuracy):")
        print(f"  Своя логистическая: {results['classification']['own_logistic']['accuracy']:.4f}")
        print(f"  Sklearn Логистическая: {results['classification']['logistic_regression']['accuracy']:.4f}")
        print(f"  Дерево решений: {results['classification']['decision_tree']['accuracy']:.4f}")
        print(f"  KNN: {results['classification']['knn']['accuracy']:.4f}\n")
        
    except Exception as e:
        print(f"Ошибка при обработке german.csv: {e}\n")

    #3 СОХРАНЕНИЕ В JSON
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Результаты сохранены в experiment_results.json")

if __name__ == "__main__":
    run_all()