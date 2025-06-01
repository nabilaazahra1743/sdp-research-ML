import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

class Main:
    def __init__(self):
        self.model_path = './IrisClassifier.sav'
        self.model = None
        self.base_models = [
            AdaBoostClassifier(), RandomForestClassifier(), ExtraTreesClassifier(),
            GradientBoostingClassifier(), HistGradientBoostingClassifier(), XGBClassifier(),
            CatBoostClassifier(verbose=0)
        ]
        self.meta_model = LogisticRegression()

    def train(self, training_directory):
        train_file = os.path.join(training_directory, 'train.csv')
        df = pd.read_csv(train_file, delimiter=';')
        X = df.drop(columns=['bugs'])
        y = df['bugs']

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

        base_model_preds = []
        for model in self.base_models:
            model.fit(X_train_base, y_train_base)
            pred = model.predict(X_train_meta)
            base_model_preds.append(pred)
        
        stacking_dataset = np.column_stack(base_model_preds)
        self.meta_model.fit(stacking_dataset, y_train_meta)
        self.model = self.meta_model

    def evaluate(self, evaluation_directory):
        eval_file = os.path.join(evaluation_directory, 'evaluate.csv')
        df = pd.read_csv(eval_file, delimiter=';')
        X_eval = df.drop(columns=['bugs'])
        y_eval = df['bugs']

        holdout_preds = []
        for model in self.base_models:
            pred = model.predict(X_eval)
            holdout_preds.append(pred)
        
        stacking_holdout_dataset = np.column_stack(holdout_preds)
        meta_model_holdout_preds = self.meta_model.predict(stacking_holdout_dataset)

        accuracy = accuracy_score(y_eval, meta_model_holdout_preds)
        precision = precision_score(y_eval, meta_model_holdout_preds)
        recall = recall_score(y_eval, meta_model_holdout_preds)
        f1score = f1_score(y_eval, meta_model_holdout_preds)
        roc_auc = roc_auc_score(y_eval, meta_model_holdout_preds)

        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision * 100:.2f}%')
        print(f'Recall: {recall * 100:.2f}%')
        print(f'F1score: {f1score * 100:.2f}%')
        print(f'Roc_auc: {roc_auc * 100:.2f}%')
        print(classification_report(y_eval, meta_model_holdout_preds))
        
        return accuracy

    def save(self):
        joblib.dump(self.model, self.model_path)

    def process_data(self, input_directory):
        input_file = os.path.join(input_directory, 'dataset.csv')
        df = pd.read_csv(input_file, delimiter=';')

        training_data_directory = os.getenv('training_data_directory', './train_data')
        evaluation_data_directory = os.getenv('evaluation_data_directory', './eval_data')
        os.makedirs(training_data_directory, exist_ok=True)
        os.makedirs(evaluation_data_directory, exist_ok=True)

        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv(os.path.join(training_data_directory, 'train.csv'), index=False, sep=';')
        eval_df.to_csv(os.path.join(evaluation_data_directory, 'evaluate.csv'), index=False, sep=';')

        print(f"Data processed and saved: {training_data_directory}/train.csv, {evaluation_data_directory}/evaluate.csv")
