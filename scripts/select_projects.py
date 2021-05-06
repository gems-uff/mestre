import pandas as pd
import configs

# criteria do select projects: 
#   at least 1000 chunks

projects = pd.read_csv(f'{configs.DATA_PATH}/number_conflicting_chunks.csv')
selected_projects = list(projects[projects['chunks'] >= 1000]['project'])

chunks = pd.read_csv(f'{configs.LABELLED_DATASET_PATH}')
selected_chunks = []
print('Processing....')
for index, row in chunks.iterrows():
    if row['project'] in selected_projects:
        selected_chunks.append(row)
selected_chunks = pd.DataFrame(selected_chunks, columns=chunks.columns)
selected_chunks.to_csv(f'{configs.SELECTED_PROJECTS_DATASET_PATH}', index=False)
print(f'File {configs.SELECTED_PROJECTS_DATASET_PATH} generated.')