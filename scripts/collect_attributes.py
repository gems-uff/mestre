import pandas as pd
import configs
import subprocess
import os
import time
import sys
import pathlib
from datetime import datetime

# Keywords count in commit messages: fix, bug, feature, improve, document, refactor, update, add, remove, use, delete, and change.
def get_keywords_frequency(parent1, parent2, base_commit):
    commits_messages = {}
    keywords_frequency = {
        'fix':0, 'bug':0, 'feature':0, 'improve':0, 'document':0, 'refactor':0,
        'update':0, 'add':0, 'remove':0, 'use':0, 'delete':0, 'change':0
    }
    commits_left = get_commits_between(base_commit, parent1)
    commits_right = get_commits_between(base_commit, parent2)
    commits_list = []
    # print(f'Commits left: {len(commits_left)}  |  Commits right: {len(commits_right)}')
    commits_list = commits_left + commits_right
    for commit in commits_list:
        commit_message = get_commit_message(commit)
        if(commit not in commits_messages):
            commits_messages[commit] = commit_message
        
    # print(commits_messages)
    for commit, commit_message in commits_messages.items():
        for keyword, _ in keywords_frequency.items():
            if keyword.lower() in commit_message.lower():
                keywords_frequency[keyword]+=1
    # print(keywords_frequency)
    return keywords_frequency


# get commit message for a commit:
    # git show -s --format=%B commit_SHA
def get_commit_message(commit_SHA):
    command = f'git show -s --format=%B {commit_SHA}'
    return execute_command(command).strip()


# get commits between two commits:
    # git log --oneline base_commit..parent
def get_commits_between(commit1, commit2):
    command = f'git log --oneline {commit1}..{commit2}'
    commits = []
    output = execute_command(command)
    output = output.strip().split('\n')
    if(len(output)>0):
        for commit in output:
            commits.append(commit.split(' ')[0])
    return commits

# Number of changed lines: git diff --shortstat mergeSHA parentSHA
    #  output: 6 files changed, 401 insertions(+), 24 deletions(-)
def get_number_changed_lines(mergeSHA, parentSHA):
    command = f"git diff --shortstat {mergeSHA} {parentSHA}"
    output = execute_command(command).strip()
    insertions = '0'
    deletions = '0'
    if 'changed' in output:
        for output_part in output.split(','):
            if 'insertion' in output_part:
                insertions = output_part.strip().split()[0]
            if 'deletion' in output_part:
                deletions = output_part.strip().split()[0]
    return insertions, deletions

def get_commit_date(SHA):
    command = f"git show -s --format=%ci {SHA}"
    output = execute_command(command).strip()
    date = datetime.strptime(output, '%Y-%m-%d %H:%M:%S %z') 
    return date


# Conclusion delay: difference in days between the dates of the merge parents
    # date: git show -s --format=%ci parentSHA
    # check which one is greater and subtract one from the other
def get_conclusion_delay(parent1SHA, parent2SHA):
    parent1_date = get_commit_date(parent1SHA)
    parent2_date = get_commit_date(parent2SHA)
    if(parent1_date > parent2_date):
        difference = parent1_date - parent2_date
    else:
        difference = parent2_date - parent1_date
    # print(f'parent1: {parent1_date}  | parent2: {parent2_date}  | difference: {difference.days}')
    return difference.days
    


def execute_command(command):
    try:
        my_env = os.environ.copy()
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, encoding='latin-1')
        return result
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return ''

# this configuration is necessary for some extra large projects like android/platform_framework_base
def config_limits():
    command = 'git config diff.renamelimit 15345'
    execute_command(command)
    command = 'git config merge.renamelimit 15345'
    execute_command(command)

def check_commits_consistency(commits):
    for commit in commits:
        command = f'git show {commit}'
        output = execute_command(command)
        if '' in output:
            return False
    return True

def write_failed_chunks(failed):
    with open('failed_chunks.txt', 'w') as file:
        for failed_chunk in failed:
            file.write(f"{failed_chunk[0]}:{failed_chunk[1]}\n")

def main():
    df = pd.read_csv(configs.INITIAL_DATASET_PATH, encoding="utf-8")
    starting_folder = pathlib.Path(__file__).parent.absolute()
    collected_commits = set()
    extracted_data = []
    print(f'Starting the collection process for {len(df)} chunks...')
    config_limits()
    current_index = 0
    columns = ['chunk_id', 'sha', 'project', 'left_lines_added', 'left_lines_removed', 'right_lines_added', 'right_lines_removed', 'conclusion_delay']
    columns.extend(['keyword_fix', 'keyword_bug', 'keyword_feature', 'keyword_improve', 'keyword_document', 'keyword_refactor', 'keyword_update'])
    columns.extend(['keyword_add', 'keyword_remove', 'keyword_use', 'keyword_delete', 'keyword_change'])
    save_every = 50
    failed_chunks = []
    for index, row in df.iterrows():
        current_index +=1
        status = (current_index / len(df)) * 100
        print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}", flush=True)
        commit_index = f"{row['project']}-{row['sha']}"
        if(commit_index not in collected_commits): # since these attributes are related to the commit and not to the chunk, we only collect it if the merge commit (sha) is not already collected
            data = []
            data.append(row['chunk_id'])
            data.append(row['sha'])
            data.append(row['project'])
            project_folder = f"{configs.REPOS_PATH}/{row['project']}"
            if os.path.exists(project_folder):
                os.chdir(project_folder)
                if not check_commits_consistency([row['sha'], row['leftsha'], row['rightsha'], row['basesha']]):
                    failed_chunks.append([row['chunk_id'], 'BAD_COMMIT'])
                else:
                    left_insertions, left_deletions = get_number_changed_lines(row['sha'], row['leftsha'])
                    right_insertions, right_deletions = get_number_changed_lines(row['sha'], row['rightsha'])
                    # print(f'left insertions: {left_insertions}  / left deletions: {left_deletions} / right insertions: {right_insertions}  /right deletions: {right_deletions}')
                    conclusion_delay = get_conclusion_delay(row['leftsha'], row['rightsha'])
                    keywords_frequency = get_keywords_frequency(row['leftsha'], row['rightsha'], row['basesha'])
                    # print(f'conclusion delay: {conclusion_delay} | keywords frequency: {keywords_frequency}')
                    data.extend([left_insertions, left_deletions, right_insertions, right_deletions,conclusion_delay])
                    for keyword, frequency in keywords_frequency.items():
                        data.append(frequency)
                    collected_commits.add(commit_index)
                    extracted_data.append(data)
                os.chdir(starting_folder)
            else:
                failed_chunks.append([row['chunk_id'], 'REPO_NOT_AVAILABLE'])
        if current_index % save_every == 0:
            pd.DataFrame(extracted_data, columns = columns).to_csv(f'{configs.DATA_PATH}/collected_attributes1.csv', index=False)
            write_failed_chunks(failed_chunks)
    os.chdir(starting_folder)

    pd.DataFrame(extracted_data, columns = columns).to_csv(f'{configs.DATA_PATH}/collected_attributes1.csv', index=False)
    write_failed_chunks(failed_chunks)

main()