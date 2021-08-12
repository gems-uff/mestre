import pandas as pd
import exploratory_analysis.utils as utils
import os

def add_remaining_columns(original_df, discretized_df):
    original_columns = original_df.columns
    discretized_df_columns = discretized_df.columns

    for column in original_columns:
        if column not in discretized_df_columns:
            discretized_df[column] = original_df[column]
    return discretized_df

def discretize_df(df, type='log10'):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(df.select_dtypes(include=numerics))
    not_attributes_columns = ['chunk_id', 'line_start', 'line_end', 'line_separator']
    float_columns = ['chunkRelSize','self_conflict_perc','chunk_left_rel_size', 'chunk_right_rel_size']
    numeric_columns = [elem for elem in numeric_columns if elem not in not_attributes_columns]
    numeric_columns = [elem for elem in numeric_columns if elem not in float_columns]
    last_not_boolean_column = "self_conflict_perc"
    last_not_boolean_column_index = list(df.columns).index(last_not_boolean_column)
    boolean_columns = list(df.columns)[last_not_boolean_column_index+1:]
    numeric_columns = [elem for elem in numeric_columns if elem not in boolean_columns]

    # clean invalid values
    for column in numeric_columns:
        if df[column].min() < 0:
            df = df[(df[column] >= 0) | (df[column].isna())]
    df = df[df['developerdecision']!='UnknownConcatenation']
    discretized_df = utils.get_discretized_df_new(df, numeric_columns, type)
    
    discretized_df = add_remaining_columns(df, discretized_df)
    return discretized_df

def print_dataset_info(df):
    total_chunks = len(df)
    total_projects = len(pd.unique(df['project']))
    total_merges = len(pd.unique(df['sha']))
    print(f"The dataset has {total_merges} merges with {total_chunks} chunks from {total_projects} projects.")

df_training = pd.read_csv("../data/dataset-training.csv")
df_test = pd.read_csv("../data/dataset-test.csv")

print_dataset_info(df_training)
print_dataset_info(df_test)

projects = list(df_training['project'].unique())
if not os.path.exists('../data/projects/discretized_log10'):
    os.mkdir('../data/projects/discretized_log10')

if not os.path.exists('../data/projects/discretized_log2'):
    os.mkdir('../data/projects/discretized_log2')

for project in projects:
    print(f'Processing project {project}')
    project_name = project.replace('/','__')
    df_project_training = pd.read_csv(f'../data/projects/{project_name}-training.csv')
    df_project_test = pd.read_csv(f'../data/projects/{project_name}-test.csv')

    df_project_training_log10 = discretize_df(df_project_training, 'log10')
    df_project_training_log2 = discretize_df(df_project_training, 'log2')
    df_project_training_log10.to_csv(f'../data/projects/discretized_log10/{project_name}-training.csv', index=False)
    df_project_training_log2.to_csv(f'../data/projects/discretized_log2/{project_name}-training.csv', index=False)

    df_project_test_log10 = discretize_df(df_project_test, 'log10')
    df_project_test_log2 = discretize_df(df_project_test, 'log2')
    df_project_test_log10.to_csv(f'../data/projects/discretized_log10/{project_name}-test.csv', index=False)
    df_project_test_log2.to_csv(f'../data/projects/discretized_log2/{project_name}-test.csv', index=False)

print('Processing complete dataset...')
df_training_log10 = discretize_df(df_training, 'log10')
df_training_log2 = discretize_df(df_training, 'log2')
df_training_log10.to_csv(f'../data/dataset-training_log10.csv', index=False)
df_training_log2.to_csv(f'../data/dataset-training_log2.csv', index=False)

df_test_log10 = discretize_df(df_test, 'log10')
df_test_log2 = discretize_df(df_test, 'log2')
df_test_log10.to_csv(f'../data/dataset-test_log10.csv', index=False)
df_test_log2.to_csv(f'../data/dataset-test_log2.csv', index=False)
