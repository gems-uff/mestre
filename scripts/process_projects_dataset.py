import pandas as pd
import configs
import os
import math

RANDOM_SEED = 19052021

def get_project_authors(df_projects, df_authors, project):
    project_chunks = df_projects[df_projects['project'] == project]
    project_chunks_id = list(project_chunks['chunk_id'].unique())
    chunk_authors = df_authors[df_authors['chunk_id'].isin( project_chunks_id)]
    authors = set()
    for index, chunk in chunk_authors.iterrows():
        chunk_authors = get_chunk_authors(chunk['authors_left'], chunk['authors_right'])
        authors.update(chunk_authors)
    return list(authors)

def get_chunk_authors(authors_left, authors_right):
    authors = set()
    authors_left = eval(authors_left)
    authors_right = eval(authors_right)
    authors.update(authors_left.keys())
    authors.update(authors_right.keys())
    return list(authors)

def set_chunks_authors(df, all_authors, df_authors):
    for author in all_authors:
        df[author] = 0
        
    for index, row in df.iterrows():
        row_authors = df_authors[df_authors['chunk_id'] == row['chunk_id']]
        if len(row_authors) > 0:
            row_authors = row_authors.iloc[0]
            authors = get_chunk_authors(row_authors['authors_left'], row_authors['authors_right'])
            for author in authors:
                df.loc[index, author.strip()] = 1
    return df


df = pd.read_csv(f"{configs.DATA_PATH}/selected_dataset_2.csv")
df_authors = pd.read_csv(f"{configs.DATA_PATH}/chunk_authors.csv")
projects_dataset_path = f"{configs.DATA_PATH}/projects"
if not os.path.exists(projects_dataset_path):
    os.mkdir(projects_dataset_path)

projects = list(df['project'].unique())
chunks_training = []
chunks_test = []
print('Starting...')
for index,project in enumerate(projects):
    print(f'Processing project {project}.')
    project_chunks = df[df['project'] == project]
    project_authors = get_project_authors(df, df_authors, project)
    total_chunks = len(project_chunks)
    training_size = math.ceil(total_chunks * 0.8)
    project_chunks_training = project_chunks.sample(n=training_size, random_state=RANDOM_SEED)
    chunks_training.append(project_chunks_training.copy())
    project_chunks_test = project_chunks.drop(project_chunks_training.index)
    chunks_test.append(project_chunks_test.copy())
    project_chunks_training = set_chunks_authors(project_chunks_training, project_authors, df_authors)
    project_chunks_test = set_chunks_authors(project_chunks_test, project_authors, df_authors)
    project_name = project.replace("/","__")
    project_chunks_training.to_csv(f"{projects_dataset_path}/{project_name}-training.csv", index=False)
    project_chunks_test.to_csv(f"{projects_dataset_path}/{project_name}-test.csv", index=False)
print("Finished.")
chunks_training = pd.concat(chunks_training)
chunks_test = pd.concat(chunks_test)
chunks_training.to_csv(f"{configs.DATA_PATH}/dataset-training.csv", index=False)
chunks_test.to_csv(f"{configs.DATA_PATH}/dataset-test.csv", index=False)


