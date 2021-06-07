import pandas as pd
from sklearn.model_selection import cross_val_score

def get_majority_class_percentage(dataset, class_name):
    count = dataset[class_name].value_counts(normalize=True)
    if len(count>0):
        return count.iloc[0]
    else:
        return float('nan')
    
def get_normalized_improvement(accuracy, baseline_accuracy):
    return ((accuracy / baseline_accuracy) - 1) * 100

def evaluate_model(projects, non_features_columns, model, drop_na=True):
    results = []
    for project in projects:
        project = project.replace("/", "__")
        project_dataset = f"../../data/projects/{project}-training.csv"
        df = pd.read_csv(project_dataset)
        if drop_na:
            df_clean = df.dropna()
        else:
            df_clean = df
        majority_class = get_majority_class_percentage(df_clean, 'developerdecision')
        if len(df_clean) >= 10:
            y = df_clean["developerdecision"].copy()
            df_clean = df_clean.drop(columns=['developerdecision'])
            df_clean = df_clean.drop(columns=non_features_columns)
            features = list(df_clean.columns)
            X = df_clean[features]
    #         print(f"project: {project} \t len df: {len(df)} \t len df clean: {len(df_clean)} \t len x: {len(X)}  \t len y: {len(y)}")
            scores = cross_val_score(model, X, y, cv=10)
            accuracy = scores.mean()
            std_dev = scores.std()
            
            normalized_improvement = get_normalized_improvement(accuracy, majority_class)
            
            results.append([project, len(df), len(df_clean), accuracy, std_dev, majority_class, normalized_improvement])
        else:
            results.append([project, len(df), len(df_clean), 0, 0, 0, 0])
    accuracy_results = pd.DataFrame(results, columns=['project', 'observations', 'observations (without NaN)', 'accuracy', 'std_dev', 'baseline (majority)', 'improvement'])
    accuracy_results = accuracy_results.round(3)
    return accuracy_results.sort_values('improvement', ascending=False)