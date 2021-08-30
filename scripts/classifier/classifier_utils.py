import pandas as pd
import math
from sklearn.model_selection import cross_val_score, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV, SelectFromModel
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt
import configs
from MDLP import MDLP_Discretizer

class ProjectResults:
    def __init__(self, project_name, results, scores, scores_text, confusion_matrix, target_names):
        self.project_name = project_name
        self.results = results
        self.scores = scores
        self.scores_text = scores_text
        self.confusion_matrix = confusion_matrix
        self.target_names = target_names

    def get_scores_df(self):
        return pd.DataFrame(self.scores).transpose()

    def get_confusion_matrix_df(self):
        print('Columns = predicted label')
        print('Rows = true label')
        return pd.DataFrame(self.confusion_matrix, index=self.target_names, columns=self.target_names)

class ProjectsResults:
    def __init__(self, algorithm, projects, non_feature_columns, projects_data_path=configs.PROJECTS_DATA, drop_na=True, replace_na=False, ablation=False, ablation_group='', ablation_mode='remove'):
        self.results = {}
        self.algorithm = algorithm
        self.evaluated_projects=0
        self.evaluate_projects(projects, non_feature_columns, algorithm, projects_data_path, drop_na, replace_na, ablation, ablation_group, ablation_mode)
    
    def add_project_result(self, project_result):
        self.results[project_result.project_name] = project_result

    def get_report_df(self, include_overall=False, sort_by='improvement'):
        results_df = []
        for project_name, project_results in self.results.items():
            results_df.append(project_results.results)
        df = pd.concat(results_df, ignore_index=True)
        df = df.sort_values(sort_by, ascending=False)
        if include_overall:
            df = pd.concat([df, get_overall_accuracy(df)], ignore_index=True)
        return df

    def evaluate_projects(self, projects, non_features_columns, algorithm, projects_data_path, drop_na, replace_na, ablation, ablation_group, ablation_mode):
        for project in projects:
            project_results = evaluate_project(project, non_features_columns, algorithm, projects_data_path, drop_na, replace_na, ablation, ablation_group, ablation_mode)
            if not np.isnan(project_results.results.iloc[0]['accuracy']):
                self.evaluated_projects+=1
            self.add_project_result(project_results)
    
    def get_project(self, project_name):
        return self.results[project_name]

    def get_class_score_df(self, target_name, score_metric, include_overall=False):
        rows = []
        columns = ['project']
        columns.append(target_name)
        for project_name, project in self.results.items():
            row = []
            row.append(project_name)
            project_df = project.get_scores_df()
            if score_metric in project_df:
                project_df_classes_recall = project_df[score_metric]
                if target_name in project_df_classes_recall:
                    row.append(project_df_classes_recall[target_name])
                else:
                    row.append(np.nan)
                rows.append(row)
        df = pd.DataFrame(rows, columns=columns)
        if include_overall:
            df = pd.concat([df, get_overall_accuracy_per_class(df)], ignore_index=True)
        return df

class IgnoreChunkAttributes:
    def __init__(self, attributes_group, project):
        self.ignored_columns = []
        self.get_ignored_attributes(attributes_group, project)
    
    def get_ignored_attributes(self, attributes_group, project):
        if attributes_group == 'complexity' or attributes_group == 'all':
            self.ignored_columns.extend(['leftCC', 'rightCC'])
        if attributes_group == 'size' or attributes_group == 'all':
            self.ignored_columns.extend(['chunkAbsSize', 'chunkRelSize', 'chunk_left_abs_size', 'chunk_left_rel_size',
             'chunk_right_abs_size', 'chunk_right_rel_size', 'chunkPosition'])
        if attributes_group == 'position' or attributes_group == 'all':
            self.ignored_columns.extend(['chunkPosition'])

        if attributes_group == 'authorship' or attributes_group == 'content' or attributes_group == 'all':
            project = project.replace("/", "__")
            projects_data_path = configs.PROJECTS_DATA
            project_dataset = f"{projects_data_path}/{project}-training.csv"
            df = pd.read_csv(project_dataset)
            if attributes_group == 'authorship' or attributes_group == 'all':
                self.ignored_columns.append('self_conflict_perc')
                
                # use an heuristic to identify column names related to the authors involved in the conflict
                # none of the other column names have '.' or '@', which most of the author' columns have
                for column in list(df.columns):
                    if '.' in column or '@' in column:
                        self.ignored_columns.append(column)
            
            if attributes_group == 'content' or attributes_group == 'all':
                all_possible_constructors = ['Class declaration', 'Return statement', 'Array access', 'Cast expression', 
                'Attribute', 'Array initializer', 'Do statement', 'Case statement', 'Other', 'Method signature', 'Break statement',
                'TypeDeclarationStatement', 'Comment', 'Method invocation', 'Package declaration', 'While statement', 
                'Interface signature', 'Variable', 'Enum value', 'Class signature', 'Annotation', 'Method interface',
                'Interface declaration', 'Synchronized statement', 'Throw statement', 'Switch statement', 'Catch clause',
                'Try statement', 'Annotation declaration', 'For statement', 'Enum declaration', 'Enum signature', 'Assert statement',
                'Static initializer', 'If statement', 'Method declaration', 'Continue statement', 'Import', 'Blank']
                for column in list(df.columns):
                    for constructor in all_possible_constructors:
                        if constructor in column:
                            self.ignored_columns.append(column)


def get_overall_accuracy_per_class(df):
    values = ['Overall']
    for column in df.columns:
        if column != 'project':
            values.append(df[column].mean(skipna=True))
    rows = [values]
    result = pd.DataFrame(rows, columns=df.columns)
    return result

def get_majority_class_percentage(dataset, class_name):
    count = dataset[class_name].value_counts(normalize=True)
    if len(count>0):
        return count.iloc[0]
    else:
        return float('nan')
    
def get_normalized_improvement(accuracy, baseline_accuracy):
    if accuracy > baseline_accuracy:
        return (accuracy - baseline_accuracy) / (1 - baseline_accuracy)
    return (accuracy - baseline_accuracy) / baseline_accuracy

def get_project_class_distribution(project_dataset, project_name, normalized=True, drop_na=True):
    developer_decisions = ['Version 1', 'Version 2', 'Combination', 'ConcatenationV1V2', 
    'ConcatenationV2V1', 'Manual', 'None']
    
    row = [project_name]
    df = project_dataset
    if drop_na:
        df_clean = df.dropna()
    else:
        df_clean = df
    count = df_clean['developerdecision'].value_counts(normalize=normalized)
    for decision in developer_decisions:
        value = 0
        try:
            if normalized:
                value = round(count[decision]*100,2)
            else:
                value = int(count[decision])
        except KeyError:
            pass
        row.append(value)
    df_columns = ['Project']
    df_columns.extend(developer_decisions)
    return pd.DataFrame([row], columns=df_columns)

def get_projects_class_distribution(projects, normalized=True, drop_na=True, include_overall=False):
    results = []
    for project in projects:
        project_name = project.replace("/", "__")
        project_dataset_path = f"../../data/projects/{project_name}-training.csv"
        df = pd.read_csv(project_dataset_path)
        results.append(get_project_class_distribution(df, project_name, normalized, drop_na))
    if include_overall:
            project_name = 'Overall'
            dataset_path = f"../../data/dataset-training.csv"
            df = pd.read_csv(dataset_path)
            results.append(get_project_class_distribution(df, project_name, normalized, drop_na))
    results = pd.concat(results, ignore_index=True)
    return results
    # return pd.DataFrame(results, columns=df_columns)

# ignores projects that were not evaluated (accuracy = np.NaN)
def get_overall_accuracy(results):
    sum_observations = results['observations'].sum()
    sum_observations_wt_nan = results['observations (wt NaN)'].sum()
    mean_precision = results['precision'].mean()
    mean_recall = results['recall'].mean()
    mean_f1_score = results['f1-score'].mean()
    mean_accuracy = results['accuracy'].mean()
    mean_baseline = results['baseline (majority)'].mean()
    mean_improvement = results['improvement'].mean()
    rows = [['Overall', sum_observations, sum_observations_wt_nan, mean_precision,
    mean_recall, mean_f1_score, mean_accuracy, mean_baseline, mean_improvement]]
    result = pd.DataFrame(rows, columns=results.columns)
    return result

def predict(algorithm, X, y):
    y_pred = cross_val_predict(algorithm, X, y, cv=10)
    return y_pred

def get_all_involved_classes(y, y_pred):
    class_names = set(y)
    predicted_class_names = set(y_pred)
    class_names = class_names.union(predicted_class_names)
    class_names = sorted(list(class_names))
    return class_names

def get_prediction_scores(y, y_pred, output_dict=True):
    class_names = get_all_involved_classes(y, y_pred)
    scores = classification_report(y, y_pred, target_names=class_names, digits=3, output_dict=output_dict)
    return scores

def get_average_tree_depth(estimator, projects, non_features_columns):
    tree_depths = []
    if type(estimator) == type(DecisionTreeClassifier()) or type(estimator) == type(RandomForestClassifier()):
        for project in projects:
            proj = project.replace("/", "__")
            proj_dataset = f"../../data/projects/{proj}-training.csv"
            df_proj = pd.read_csv(proj_dataset)
            df_clean = df_proj.dropna()
            if len(df_clean) >= 10:
                y = df_clean["developerdecision"].copy()
                df_clean_features = df_clean.drop(columns=['developerdecision']) \
                                            .drop(columns=non_features_columns)
                features = list(df_clean_features.columns)
                X = df_clean_features[features]
                estimator.fit(X,y)
                if type(estimator) == type(DecisionTreeClassifier()):
                    tree_depths.append(estimator.get_depth())
                elif type(estimator) == type(RandomForestClassifier()):
                    for estimator_ in estimator.estimators_:
                        tree_depths.append(estimator_.get_depth())
    if len(tree_depths) > 0:
        return sum(tree_depths)/len(tree_depths), tree_depths
    else:
        return None

'''
Compare models overall scores (among all projects given by the parameter)
'''
def compare_models(models, models_names, projects, non_features_columns):
    models_results = []
    for model in models:
        models_results.append(ProjectsResults(model, projects, non_features_columns))
    reports = []
    for model_results in models_results:
        reports.append(model_results.get_report_df(include_overall=True))
    if len(reports) > 0:
        df_result = reports[0].loc[(reports[0]['project']=='Overall')].copy()
        df_result['model'] = None
        for i in range(1, len(reports)):
            df_model_overall_report = reports[i].loc[(reports[i]['project']=='Overall')]
            df_result = pd.concat([df_result, df_model_overall_report], ignore_index=True)
        
        model_index = 0
        for index, row in df_result.iterrows():
            df_result.at[index, 'model'] = models_names[model_index]
            model_index+=1
        
        return df_result

'''
Compare models precision, recall and f-measure for a specified class (target_name)
'''
def compare_models_per_class(target_name, models_results, models_names):
    data = []
    columns = ['model', 'precision', 'recall', 'f1-score']
    for model_name, model_results in models_results.items():
        if model_name in models_names:
            precision_df = model_results.get_class_score_df(target_name, 'precision', include_overall=True)
            precision = precision_df[precision_df['project'] == 'Overall'].iloc[0][target_name]
            recall_df = model_results.get_class_score_df(target_name, 'recall', include_overall=True)
            recall = recall_df[recall_df['project'] == 'Overall'].iloc[0][target_name]
            f1score_df = model_results.get_class_score_df(target_name, 'f1-score', include_overall=True)
            f1score = f1score_df[f1score_df['project'] == 'Overall'].iloc[0][target_name]
            data.append([model_name, precision, recall, f1score])
    return pd.DataFrame(data, columns=columns)

'''
Compare models overall scores (among all projects given by the parameter) assigning medals to top3
'''
def compare_models_medals(models, models_names, projects, non_features_columns):
    results = {}
    results_columns = ['model_name', 'mean_accuracy', 'sum_accuracy', 'sum_rank', 'total_medals', 'gold_medals', 'silver_medals', 'bronze_medals']
    rows = []

    # initialize the result dataframe
    for model, model_name in zip(models, models_names):
        row = [model_name, 0.0,0.0,0,0,0,0,0]
        rows.append(row)

    results = pd.DataFrame(rows, columns=results_columns)
    total_evaluated_projects = 0
    
    for project in projects:
        model_results_project = {}
        columns = ['model_name', 'accuracy']
        rows = []
        
        evaluated_project = False

        # execute each model for this project
        for model, model_name in zip(models, models_names):
            model_results = ProjectsResults(model, [project], non_features_columns)
            model_results_df = model_results.get_report_df(include_overall=True)
            model_accuracy = model_results_df[model_results_df['project']=='Overall'].iloc[0]['accuracy']
            if not np.isnan(model_accuracy) and not evaluated_project:
                evaluated_project = True
            rows.append([model_name, model_accuracy])
        
        if evaluated_project:
            total_evaluated_projects+=1
        
        model_results_project = pd.DataFrame(rows, columns=columns)
        model_results_project['rank'] = model_results_project['accuracy'].rank(method='min', ascending=False)
        
        # assign medals to the top-3 models
        for index, model_results in model_results_project.iterrows():
            if not np.isnan(model_results['accuracy']):
                row = results[results['model_name'] == model_results['model_name']].iloc[0]
                sum_accuracy = row['sum_accuracy']
                sum_rank = row['sum_rank']
                gold_medals = row['gold_medals']
                silver_medals = row['silver_medals']
                bronze_medals = row['bronze_medals']
                
                results.at[row.name, 'sum_accuracy'] = sum_accuracy + model_results['accuracy']
                results.at[row.name, 'sum_rank'] = sum_rank + model_results['rank']
                if model_results['rank'] == 1:
                    results.at[row.name, 'gold_medals'] = gold_medals + 1
                elif model_results['rank'] == 2:
                    results.at[row.name, 'silver_medals'] = silver_medals + 1
                elif model_results['rank'] == 3:
                    results.at[row.name, 'bronze_medals'] = bronze_medals + 1
                
                if model_results['rank'] <= 3:
                    results.at[row.name, 'total_medals'] = gold_medals + silver_medals + bronze_medals + 1
    
    if total_evaluated_projects != 0:
        results['mean_accuracy'] = results['sum_accuracy'] / total_evaluated_projects
        results['mean_rank'] = results['sum_rank'] / total_evaluated_projects
    else:
        results['mean_accuracy'] = 0
        results['mean_rank'] = 0
    results = results.drop(['sum_accuracy'], axis=1)
    results = results.drop(['sum_rank'], axis=1)

    return results

def replace_na_values(df):
    imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
    df_constant = pd.DataFrame(imp_constant.fit_transform(df),
                           columns = df.columns)
    return df_constant

# obs about metrics used:
# weighted metrics: precision, recall, and f1-score
# Calculate metrics for each label (class), and find their average weighted by support (the number of true instances for each label).
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
# macro metrics: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
def evaluate_project(project, non_features_columns, algorithm, projects_data_path, drop_na=True, replace_na=False, ablation=False, ablation_group='', ablation_mode='remove'):
    results = []
    class_names = []
    project = project.replace("/", "__")
    project_dataset = f"{projects_data_path}/{project}-training.csv"
    df = pd.read_csv(project_dataset)
    if replace_na:
        df = replace_na_values(df)
    if drop_na:
        df_clean = df.dropna()
    else:
        df_clean = df
    if ablation:
        if ablation_mode == 'add':
            chunk_attributes = IgnoreChunkAttributes('all', project).ignored_columns
            include_attributes = IgnoreChunkAttributes(ablation_group, project).ignored_columns
            ignored_attributes = list(set(chunk_attributes) - set(include_attributes))
        else:
            ignored_attributes = IgnoreChunkAttributes(ablation_group, project).ignored_columns
        df_clean = df_clean.drop(columns=ignored_attributes)
        # print(ignored_attributes.ignored_columns)
    # print(len(df_clean.columns))
    majority_class = get_majority_class_percentage(df_clean, 'developerdecision')
    scores = {}
    scores_text= ''
    conf_matrix = []
    target_names = sorted(df['developerdecision'].unique())
    if len(df_clean) >= 10:
        y = df_clean["developerdecision"].copy()
        df_clean = df_clean.drop(columns=['developerdecision'])
        df_clean = df_clean.drop(columns=non_features_columns)
        features = list(df_clean.columns)
        X = df_clean[features]
#         print(f"project: {project} \t len df: {len(df)} \t len df clean: {len(df_clean)} \t len x: {len(X)}  \t len y: {len(y)}")
        # scores = cross_val_score(model, X, y, cv=10)
        # accuracy = scores.mean()
        # std_dev = scores.std()
        y_pred = predict(algorithm, X, y)
        scores = get_prediction_scores(y, y_pred)
        scores_text = get_prediction_scores(y, y_pred, False)
        class_names = get_all_involved_classes(y,y_pred)
        conf_matrix = confusion_matrix(y, y_pred, labels=class_names)
        
        accuracy = scores['accuracy']
        precision = scores['weighted avg']['precision']
        recall = scores['weighted avg']['recall']
        f1_score = scores['weighted avg']['f1-score']
        normalized_improvement = get_normalized_improvement(accuracy, majority_class)
        
        results.append([project, len(df), len(df_clean), precision, recall, f1_score, accuracy, majority_class, normalized_improvement])
    else:
        results.append([project, len(df), len(df_clean), np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    results = pd.DataFrame(results, columns=['project', 'observations', 'observations (wt NaN)', 'precision', 'recall', 'f1-score', 'accuracy', 'baseline (majority)', 'improvement'])
    
    results = results.round(3)
    return ProjectResults(project, results, scores, scores_text, conf_matrix, class_names)

# adapted from https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

# adapted from https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# adapted from https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False, color='black')
    ax.set_yticklabels(yticklabels, minor=False, color='black')

    fig.patch.set_facecolor('white')

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))


# adapted from https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''

    plotMat = []
    support = []
    class_names = []

    not_classes = ['accuracy', 'macro avg', 'weighted avg']
    for class_name, metrics in classification_report.items():
            row_values = []
            if class_name not in not_classes:
                class_names.append(class_name)
                for metric, value in classification_report[class_name].items():
                    if metric != 'support':
                        row_values.append(float(value))
                    else:
                        support.append(value)
                plotMat.append(row_values)


    # print('plotMat: {0}'.format(plotMat))
    # print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

'''
    Assigns gold, silver, and bronze medals for the top-3 combination of parameters in each project.
'''
def grid_search_all(projects, estimator, parameters, non_features_columns):
    import itertools
    results = {}
    results_columns = list(parameters.keys())
    results_columns.extend(['mean_accuracy', 'sum_accuracy', 'sum_rank', 'total_medals', 'gold_medals', 'silver_medals', 'bronze_medals'])
    combinations = []

    for combination in itertools.product(*parameters.values()):
        row = []
        key=''
        for parameter_value in combination:
            row.append(parameter_value)
            key+=str(parameter_value)
        row.extend([0,0,0,0,0,0,0])
        combinations.append(row)
    results = pd.DataFrame(combinations, columns=results_columns)
    
    total_evaluated_projects = 0
    for project in projects:
        project_results = grid_search(project, estimator, parameters, non_features_columns)
        if project_results != None:
            df_gridsearch_dt = pd.DataFrame(project_results)\
                .filter(regex=("param_.*|mean_test_score|std_test_score|rank_test_score"))\
                .sort_values(by=['rank_test_score'])
            
            for index, combination in df_gridsearch_dt.iterrows():
                filtered_rows = results
                for parameter in list(parameters.keys()):
                    parameter_key = f'param_{parameter}'
                    combination_value = combination[parameter_key]
                    if combination[parameter_key] == None:
                        filtered_rows = filtered_rows[filtered_rows[parameter].isnull()]
                    else:
                        filtered_rows = filtered_rows[filtered_rows[parameter]==combination_value]

                if len(filtered_rows) > 0:
                    row = results.loc[filtered_rows.index]
                    sum_accuracy = row['sum_accuracy']
                    sum_rank = row['sum_rank']
                    gold_medals = row['gold_medals']
                    silver_medals = row['silver_medals']
                    bronze_medals = row['bronze_medals']
                    
                    results.at[filtered_rows.index, 'sum_accuracy'] = sum_accuracy + combination['mean_test_score']
                    results.at[filtered_rows.index, 'sum_rank'] = sum_rank + combination['rank_test_score']
                    if combination['rank_test_score'] == 1:
                        results.at[filtered_rows.index, 'gold_medals'] = gold_medals + 1
                    elif combination['rank_test_score'] == 2:
                        results.at[filtered_rows.index, 'silver_medals'] = silver_medals + 1
                    elif combination['rank_test_score'] == 3:
                        results.at[filtered_rows.index, 'bronze_medals'] = bronze_medals + 1
                    
                    if combination['rank_test_score'] <= 3:
                        results.at[filtered_rows.index, 'total_medals'] = gold_medals + silver_medals + bronze_medals + 1
            total_evaluated_projects+=1
    results['mean_accuracy'] = results['sum_accuracy'] / total_evaluated_projects
    results['mean_rank'] = results['sum_rank'] / total_evaluated_projects
    
    results = results.drop(['sum_accuracy'], axis=1)
    results = results.drop(['sum_rank'], axis=1)

    return results



def grid_search(project, estimator, parameters, non_features_columns):
    proj = project.replace("/", "__")
    proj_dataset = f"../../data/projects/{proj}-training.csv"
    df_proj = pd.read_csv(proj_dataset)
    df_clean = df_proj.dropna()
    # print(f"Length of df_clean: {len(df_clean)}")
    if len(df_clean) >= 10:
        # majority_class = get_majority_class_percentage(df_clean, 'developerdecision')
        y = df_clean["developerdecision"].copy()
        df_clean_features = df_clean.drop(columns=['developerdecision']) \
                                    .drop(columns=non_features_columns)
        features = list(df_clean_features.columns)
        X = df_clean_features[features]
        clf = GridSearchCV(estimator, parameters, verbose=0, cv=10)
        clf.fit(X, y)
        # print("Best params and score:", clf.best_params_, clf.best_score_, '\n',
              # clf.cv_results_,
            #   sep='\n')
        return clf.cv_results_
    else:
        return None

def get_validation_curve_all(projects, estimator, param_name, param_range, non_features_columns):
    train_scores_mean = []
    train_scores_std = []
    test_scores_mean = []
    test_scores_std = []
    number_projects = 0
    for project in projects:
        proj = project.replace("/", "__")
        proj_dataset = f"../../data/projects/{proj}-training.csv"
        df_proj = pd.read_csv(proj_dataset)
        df_clean = df_proj.dropna()
        # print(f"Length of df_clean: {len(df_clean)}\n")
        if len(df_clean) >= 10:
            y = df_clean["developerdecision"].copy()
            df_clean_features = df_clean.drop(columns=['developerdecision']) \
                                        .drop(columns=non_features_columns)
            features = list(df_clean_features.columns)
            X = df_clean_features[features]
            train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=10)

            train_scores_mean.append(np.mean(train_scores, axis=1).tolist())
            train_scores_std.append(np.std(train_scores, axis=1).tolist())
            test_scores_mean.append(np.mean(test_scores, axis=1).tolist())
            test_scores_std.append(np.std(test_scores, axis=1).tolist())
            number_projects+=1


    train_scores_mean= np.mean(train_scores_mean, axis=0)
    train_scores_std= np.mean(train_scores_std, axis=0)
    test_scores_mean= np.mean(test_scores_mean, axis=0)
    test_scores_std= np.mean(test_scores_std, axis=0)

    plt.title(f"Accumulated Validation Curve with {type(estimator).__name__}.\n Number of projects: {number_projects}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    if None in param_range:
        param_range.remove(None)
        param_range.insert(0,-1)
    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2,
                        color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.2,
                        color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def plot_validation_curve(project, estimator, param_name, param_range, non_features_columns, ax):
    proj = project.replace("/", "__")
    proj_dataset = f"../../data/projects/{proj}-training.csv"
    df_proj = pd.read_csv(proj_dataset)
    df_clean = df_proj.dropna()
    if len(df_clean) >= 10:
        # majority_class = get_majority_class_percentage(df_clean, 'developerdecision')
        y = df_clean["developerdecision"].copy()
        df_clean_features = df_clean.drop(columns=['developerdecision']) \
                                    .drop(columns=non_features_columns)
        features = list(df_clean_features.columns)
        X = df_clean_features[features]
        train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=10)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.set_title(f"{project} \n n={len(df_clean)}", wrap=True)
        ax.set_xlabel(param_name)
        # plt.xticks(param_range)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.1)
        lw = 2
        if None in param_range:
            param_range = param_range.copy()
            param_range.remove(None)
            param_range.insert(0,-1)
        ax.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
        ax.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        ax.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
        ax.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        ax.legend(loc="best")
        return True
    return False

def plot_validation_curves(projects, estimator, parameter, param_range, non_features_columns):
    import math
   
    N = len(projects)
    cols = 4
    rows = int(math.ceil(N / cols))
    plt.figure(figsize=(15,4*rows))
    
    plot_index = 1
    for n in range(N):
        ax = plt.subplot(rows,cols, plot_index)
        has_data = plot_validation_curve(projects[n], estimator, parameter,
                                                param_range,
                                                non_features_columns, ax)
        if not has_data:
            ax.remove()
        else:
            plot_index+=1
    plt.tight_layout()
    plt.savefig(f'validation_curves_{type(estimator).__name__}_{parameter}.png', bbox_inches='tight')

def projects_feature_selection(projects, non_features_columns, algorithm, feature_selection_strategy):
    results = []
    selected_features_records = []
    for project in projects:
        project = project.replace("/", "__")
        project_dataset = f"{configs.PROJECTS_DATA}/{project}-training.csv"
        df = pd.read_csv(project_dataset)
        df_clean = df.dropna()
        
        if len(df_clean) >= 10:
            y = df_clean["developerdecision"].copy()
            df_clean = df_clean.drop(columns=['developerdecision'])
            df_clean = df_clean.drop(columns=non_features_columns)
            features = list(df_clean.columns)
            X = df_clean[features]
            scores = cross_val_score(algorithm, X, y, cv=10)
            default_accuracy = scores.mean()

            # feature selection code
            if feature_selection_strategy == 'tree':
                algorithm.fit(X,y)
                selection_algorithm = SelectFromModel(algorithm, prefit=True)
                selection_algorithm.transform(X)
                selected_features = get_list_selected_features(selection_algorithm, X)
            elif feature_selection_strategy == 'recursive':
                min_features_to_select = 1
                selection_algorithm = RFECV(estimator=algorithm, step=1, cv=5,
                  scoring='accuracy', n_jobs=5,
                  min_features_to_select=min_features_to_select)
                selection_algorithm.fit(X, y)
                selection_algorithm.transform(X)
                selected_features = get_list_selected_features(selection_algorithm, X)
            elif feature_selection_strategy == 'IGAR':
                # TODO: implement the IGAR using sklearn interface (to use the fit/transform functions)
                # uses the n value found in the IGAR tuning notebook
                n = 65
                selected_features, attributes_ranking = IGAR(n, X, y)
            else:
                print('Invalid feature selection strategy')
                return None

            X_selected = get_selected_data(X, selected_features)
            original_features = X.shape[1]
            nr_selected_features = X_selected.shape[1]
            scores = cross_val_score(algorithm, X_selected, y, cv=10)
            new_accuracy = scores.mean()
            normalized_improvement = get_normalized_improvement(new_accuracy, default_accuracy)

            for selected_feature in selected_features:
                information_gain = get_information_gain(list(y), list(X_selected[selected_feature]))
                selected_features_records.append([project, selected_feature, information_gain, feature_selection_strategy])

            results.append([project, len(df_clean), original_features, nr_selected_features, default_accuracy, new_accuracy, normalized_improvement])
        else:
            results.append([project, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    
    results = pd.DataFrame(results, columns=['project', 'N', '# attr.', '# attr. fs', 'accuracy', 'accuracy_fs', 'improvement'])
    results = pd.concat([results, get_overall_feature_selection(results)], ignore_index=True)
    results = results.round(3)
    
    return results, selected_features_records

def get_list_selected_features(selection_algorithm, X):
    selected_features= X.columns[(selection_algorithm.get_support())]
    return list(selected_features)

def get_overall_feature_selection(df):
    n = df['N'].sum()
    nr_attributes = df['# attr.'].mean(skipna=True)
    nr_selected_attributes = df['# attr. fs'].mean(skipna=True)
    default_accuracy = df['accuracy'].mean(skipna=True)
    new_accuracy = df['accuracy_fs'].mean(skipna=True)
    improvement = get_normalized_improvement(new_accuracy, default_accuracy)
    
    values = ['Overall', n, nr_attributes, nr_selected_attributes, default_accuracy, new_accuracy, improvement]
    rows = [values]
    result = pd.DataFrame(rows, columns=df.columns)
    return result

'''
    CREDITS TO https://stackoverflow.com/a/64914824
'''
def get_information_gain(class_, feature):
  classes = set(class_)

  Hc = 0
  for c in classes:
    pc = class_.count(c)/len(class_) # probability of the class
    Hc += - pc * math.log(pc, 2) # accumulate the entropy for this class

  feature_values = set(feature)
  Hc_feature = 0
  for feat in feature_values:
    pf = feature.count(feat)/len(feature) # probability of the feature
    indices = [i for i in range(len(feature)) if feature[i] == feat] # indices where the feature happens
    clasess_of_feat = [class_[i] for i in indices] # classes assigned to each occurrence of this feature
    for c in classes:
        pcf = clasess_of_feat.count(c)/len(clasess_of_feat) # probability of having this class and this feature
        if pcf != 0:
            temp_H = - pf * pcf * math.log(pcf, 2)
            Hc_feature += temp_H
    # overall entropy (among the classes) - entropy of the feature  
    ig = Hc - Hc_feature
    return ig

'''
    Applies MLDP discretization to numeric values.
    MLDP Implementation: https://github.com/navicto/Discretization-MDLPC 
'''
def get_mldp_discretization(X, y, ignore_columns=[]):
    X_orig = X.copy()
    X = X.to_numpy()
    y = y.to_numpy()
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_features_names = list(X_orig.select_dtypes(include=numerics)) # columns names
    numeric_features = [X_orig.columns.get_loc(col) for col in numeric_features_names] # columns indexes
    discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer.fit(X, y)
    X_discretized = discretizer.transform(X)
    return pd.DataFrame(X_discretized, columns=numeric_features_names)

'''
    Feature selection method based on the information gain ranking of the attributes. 
    Returns the n attributes with the highest information gain values and their information gain (2 lists)
'''
def IGAR(n, X, y):
    features = {}
    selected_features = []
    information_gains = []
    # it is necessary to discretize numeric values 
    #  before calculating the information gain
    X_discretized = get_mldp_discretization(X, y)
    for column in X_discretized.columns:
        feature = X_discretized[column]
        information_gain = get_information_gain(list(y), list(feature))
        features[column] = information_gain

    # order the attributes according to descending information gain
    for name, ig in sorted(features.items(), key=lambda x: x[1], reverse=True):
        selected_features.append(name)
        information_gains.append(ig)
    if n == 0:
        return selected_features, information_gains
    if len(selected_features) >= n:
        return selected_features[:n], information_gains[:n]
    else:
        return [],[]

'''
    returns only the selected data based on the original dataset and a list of selected features
'''
def get_selected_data(original_X, selected_features):
    new_selected_attributes = []

    # we need to make sure that the order of the columns is not changed
    # https://github.com/scikit-learn/scikit-learn/issues/5394
    for column in original_X.columns:
        if column in selected_features:
            new_selected_attributes.append(column)
    return original_X[new_selected_attributes]

'''
Collect accuracy for each n
Vary the n parameter from min_n to max_n
    For each n:
        For each project:
            Execute the IGAR using n
            Execute the prediction (calculate accuracy and normalized improvement)
'''
def IGAR_tuning(prediction_algorithm, projects, non_features_columns, min_n, max_n, save=False):
    columns = ['project', 'n', 'accuracy', 'accuracy_selected', 'orig_attr']
    data = []
    for n in range(min_n, max_n+1):
        for project in projects:
            project = project.replace("/", "__")
            project_dataset = f"{configs.PROJECTS_DATA}/{project}-training.csv"
            df = pd.read_csv(project_dataset)
            df_clean = df.dropna()
            
            scores = {}
            if len(df_clean) >= 10:
                y = df_clean["developerdecision"].copy()
                df_clean = df_clean.drop(columns=['developerdecision'])
                df_clean = df_clean.drop(columns=non_features_columns)
                features = list(df_clean.columns)
                X = df_clean[features]

                scores = cross_val_score(prediction_algorithm, X, y, cv=5)
                default_accuracy = scores.mean()

                selected_attributes, attributes_ranking = IGAR(n, X, y)
                X_selected = get_selected_data(X, selected_attributes)

                scores = cross_val_score(prediction_algorithm, X_selected, y, cv=5)
                new_accuracy = scores.mean()

                data.append([project, n, default_accuracy, new_accuracy, X.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    if save:
        df.to_csv('IGAR_tuning.csv', index=False)
    return df

def compute_IGAR_tuning_summary(IGAR_results, min_n, max_n):
    wins = {}
    ranking = {}
    for n in range(min_n, max_n+1):
        wins[n] = 0
        ranking[n] = 0

    for project in IGAR_results['project'].unique():
        
        # find number of wins
        project_results = IGAR_results[IGAR_results['project']==project].copy()
        best_result = project_results.loc[project_results['accuracy_selected'].idxmax()]
        n = best_result['n']
        wins[n] += 1
        
        # accumulate ranking 
        project_results['ranking'] = project_results['accuracy_selected'].rank(ascending=False)
        for index, row in project_results.iterrows():
            ranking[row['n']] += row['ranking']

    # average the accumulated rankings
    n_projects = len(IGAR_results['project'].unique())
    for n, accumulated_rank in ranking.items():
        average = accumulated_rank/n_projects
        ranking[n] = average
            
    columns = ['n', 'average_default_accuracy', 'average_accuracy', 'improvement', 'average_ranking', 'number_wins']
    data = []
    for n in range(min_n, max_n+1):
        row = [n]
        
        n_results = IGAR_results[IGAR_results['n'] == n]
        average_accuracy = n_results['accuracy_selected'].mean(skipna=True)
        default_accuracy = n_results['accuracy'].mean(skipna=True)
        improvement = get_normalized_improvement(average_accuracy, default_accuracy)
        
        row.append(default_accuracy)
        row.append(average_accuracy)
        row.append(improvement)
        row.append(ranking[n])
        row.append(wins[n])
        data.append(row)

    return pd.DataFrame(data, columns=columns)

def is_authorship_feature(feature_name):
    if '.' in feature_name or '@' in feature_name:
        return True

def is_content_feature(feature_name):
    all_constructors = ['Class declaration', 'Return statement', 'Array access', 'Cast expression', 
                            'Attribute', 'Array initializer', 'Do statement', 'Case statement', 'Other', 'Method signature', 'Break statement',
                            'TypeDeclarationStatement', 'Comment', 'Method invocation', 'Package declaration', 'While statement', 
                            'Interface signature', 'Variable', 'Enum value', 'Class signature', 'Annotation', 'Method interface',
                            'Interface declaration', 'Synchronized statement', 'Throw statement', 'Switch statement', 'Catch clause',
                            'Try statement', 'Annotation declaration', 'For statement', 'Enum declaration', 'Enum signature', 'Assert statement',
                            'Static initializer', 'If statement', 'Method declaration', 'Continue statement', 'Import', 'Blank']
    return feature_name in all_constructors

#   input list columns = project | attribute | information gain | selection method
def get_attribute_selection_ranking(attributes_stats, method):
    results = []
    df = pd.DataFrame(attributes_stats, columns=['project', 'attribute', 'information_gain', 'method'])
    selected_df = df[df['method'] == method].copy()
    attributes = list(selected_df['attribute'].unique())
    selected_df['ranking'] = selected_df.groupby('project')['information_gain'].rank('dense', ascending=False)
    for attribute in attributes:
        attribute_results = selected_df[selected_df['attribute'] == attribute].copy()
        times_selected = len(attribute_results)
        average_information_gain = attribute_results['information_gain'].mean(skipna=True)
        average_ranking = attribute_results['ranking'].mean(skipna=True)
        results.append([attribute, times_selected, average_information_gain, average_ranking])
    
    # process authorship related attributes grouping
    information_gain_sum = 0
    times_selected_sum = 0
    ranking_sum = 0
    count = 0
    for result in results:
        if is_authorship_feature(result[0]):
            times_selected_sum += result[1]
            information_gain_sum += result[2]
            ranking_sum += result[3]
            count+=1
    information_gain_average = information_gain_sum / count
    times_selected_average = times_selected_sum / count
    ranking_average = ranking_sum / count
    results.append(['chunk_author', times_selected_average, information_gain_average, ranking_average])

    # process content related attributes grouping
    information_gain_sum = 0
    times_selected_sum = 0
    ranking_sum = 0
    count = 0
    for result in results:
        if is_content_feature(result[0]):
            times_selected_sum += result[1]
            information_gain_sum += result[2]
            ranking_sum += result[3]
            count+=1
    information_gain_average = information_gain_sum / count
    times_selected_average = times_selected_sum / count
    ranking_average = ranking_sum / count
    results.append(['content_constructor', times_selected_average, information_gain_average, ranking_average])
   
    return pd.DataFrame(results, columns=['attribute', 'count_selected', 'average_information_gain', 'average_ranking'])
