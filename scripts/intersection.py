import os
import configs
import subprocess
import pandas as pd

# given a folder with repos, investigate if and how much intersection there is between them

repos_folder = '../repos'

def execute_command(command, path):
    try:
        my_env = os.environ.copy()
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, cwd=path, encoding="latin-1")
        return result.strip()
    except subprocess.CalledProcessError as e:
        pass
    return ''
    
if __name__ == "__main__":
    df = pd.read_csv('../data/SELECTED_LABELLED_DATASET.csv')
    projects = df['project'].unique()
    projects_commits = {}
    projects_intersection = {}
    for project in projects:
        project_path = f'{repos_folder}/{project}'
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
    pd.DataFrame(data, columns=columns).to_csv('../data/projects_intersection.csv', index=False)
    print(len(projects))