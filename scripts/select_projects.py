import pandas as pd
import configs
import os
import subprocess

# criteria do select projects: 
#   at least 1000 chunks

'''
    Filters projects that have any degree of intersection between their commits (implicit forks)
'''
def filter_intersection(selected_projects):
    projects = list(selected_projects)
    projects_commits = {}
    projects_intersection = {}
    for project in projects:
        project_path = f'{configs.REPOS_PATH}/{project}'
        project_commits = set()
        if os.path.exists(project_path):
            command = 'git rev-list --all'
            all_commits = execute_command(command, project_path)
            for commit in all_commits.split():
                project_commits.add(commit)
            projects_commits[project] = project_commits
    data = []
    columns = ['project1', 'project2', 'intersection_perc']
    for project, commits in projects_commits.items():
        for project2, commits2 in projects_commits.items():
            if project != project2:
                intersection = commits.intersection(commits2)
                project_project2 = f'{project} -> {project2}'
                project2_project = f'{project2} -> {project}'
                if project_project2 not in projects_intersection:
                    intersection_perc = len(intersection) / len(commits)
                    projects_intersection[project_project2] = intersection_perc
                    data.append([project, project2, intersection_perc])
                        
                if project2_project not in projects_intersection:
                    intersection_perc = len(intersection) / len(commits2)
                    projects_intersection[project2_project] = intersection_perc
                    data.append([project2, project, intersection_perc])
    intersection = pd.DataFrame(data, columns=columns)
    intersection.to_csv('../data/projects_intersection.csv', index=False)
    projects_that_intersect = intersection[intersection['intersection_perc'] > 0]
    filtered_projects = set()
    for index, row in projects_that_intersect.iterrows():
        filtered_projects.add(row['project1'])
        filtered_projects.add(row['project2'])
    filtered_projects.remove('android/platform_frameworks_base') # we keep the biggest of the projects that intersect
    print(f'Filtered {len(filtered_projects)} of {len(selected_projects)} projects for being implicit forks: {filtered_projects}')
    selected_projects = set(selected_projects)
    selected_projects = selected_projects - filtered_projects
    return list(selected_projects)

'''
    Filters out projects that do not have enough chunks with enough data to be used in the study (missing attributes)
'''
def filter_projects_missing_data(selected_projects):
    new_selected_projects = []
    excluded_projects = ['elastic/elasticsearch', 'eclipse/jetty.project', 'revolsys/com.revolsys.open']
    for project in selected_projects:
        if project not in excluded_projects:
            new_selected_projects.append(project)
    return new_selected_projects            
    

def execute_command(command, path):
    try:
        my_env = os.environ.copy()
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, cwd=path, encoding="latin-1")
        return result.strip()
    except subprocess.CalledProcessError as e:
        pass
    return ''

projects = pd.read_csv(f'{configs.DATA_PATH}/number_conflicting_chunks.csv')
selected_projects = list(projects[projects['chunks'] >= 1000]['project'])
selected_projects = filter_intersection(selected_projects)
selected_projects = filter_projects_missing_data(selected_projects)
chunks = pd.read_csv(f'{configs.LABELLED_DATASET_PATH}')
selected_chunks = []
print('Processing labelled dataset....')
for index, row in chunks.iterrows():
    if row['project'] in selected_projects:
        selected_chunks.append(row)
selected_chunks = pd.DataFrame(selected_chunks, columns=chunks.columns)
selected_chunks.to_csv(f'{configs.SELECTED_PROJECTS_DATASET_PATH}', index=False)
print(f'File {configs.SELECTED_PROJECTS_DATASET_PATH} generated.')

chunks = pd.read_csv(f'{configs.DATA_PATH}/dataset.csv')
selected_chunks = []
print('Processing dataset....')
for index, row in chunks.iterrows():
    if row['project'] in selected_projects:
        selected_chunks.append(row)
selected_chunks = pd.DataFrame(selected_chunks, columns=chunks.columns)
selected_chunks.to_csv(f'{configs.DATA_PATH}/selected_dataset.csv', index=False)
print(f'File {configs.DATA_PATH}/selected_dataset.csv generated.')