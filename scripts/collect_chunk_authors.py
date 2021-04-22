import pandas as pd
import configs
import subprocess
import os
import time
import sys
import pathlib
from datetime import datetime
import re

starting_folder = ''

# get commit message for a commit:
    # git show -s --format=%B commit_SHA
def get_commit_message(commit_SHA):
    command = f'git show -s --format=%B {commit_SHA}'
    return execute_command(command).strip()

def reset():
    command = "git reset --hard"
    output = execute_command(command)
    if "HEAD is now at" in output.strip():
        return True
    return False

def checkout(sha):
    if reset():
        command = f"git checkout {sha}"
        output = execute_command(command)
        if "HEAD is now at" in output.strip():
            return True
    return False

def merge(left_sha, right_sha):
    if checkout(left_sha):
        command = f"git merge {right_sha}"
        output = execute_command(command)
        return True
    return False
        
def blame(file_path, line_start, line_end):
    command = f"git blame --show-email -L {line_start},{line_end} {file_path}"
    return execute_command(command)

def show_file_lines(file_path, start, end):
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines[start:end+1]:
        print(line)

# strategy:
# redo the merge
# git blame the conflicted file
# inspect the result
def get_chunks_author(file_path, line_start, line_end, line_separator, merge_base):
    blame_output_left = blame(file_path, line_start, line_separator)
    # print(f'blame left:  \n {blame_output_left}')
    authors_left = extract_authors(blame_output_left)
    if len(authors_left) == 0:
        authors_left = find_deleted_lines_authors(merge_base, line_start, line_separator, file_path)
    print(f'authors_left: {authors_left}')
    print('------------------')
    blame_output_right = blame(file_path, line_separator+1, line_end)
    # print(f'blame right: \n {blame_output_right}')
    authors_right = extract_authors(blame_output_right)
    if len(authors_right) == 0:
        authors_right = find_deleted_lines_authors(merge_base, line_separator+1, line_end, file_path)
    print(f'authors_right: {authors_right}')
        

def extract_authors(blame_output):
    authors = set()
    lines = blame_output.split('\n')
    insideChunk = False
    # print('------------------------ AAA')
    for line in lines:
        # print(f'line: {line}')
        if '<<<<<<<' in line or '=======' in line:
            insideChunk = True
        elif insideChunk:
            if '(' in line:
                line_parts = line.strip().split('(')[1].split()
                # print(line_parts)
                author = line_parts[0].replace('<','').replace('>','').replace('(','')
                authors.add(author)
    return authors


# how we find who deleted a line:
# use difflame tool (https://github.com/eantoranz/difflame)
#   ./difflame.py {merge_base_sha} -e -- {file_path}
# example:
#   ./difflame.py 708e2db0bf5e3bfbb48bf94d604ef883970a2b92 -e -- app/src/main/java/org/connectbot/HostListActivity.java
# example of a line outputed by difflame:
#     12ef92bf (<kenny@the-b.org>       2008-11-14 10:07:29  29  20) import java.util.List;
# last commit where the line was modified | email of the author | timestamp of the commit | line number in the merge base | line number in the conflictingfile
# example of a deleted line and information about the commit that deleted it:
# -0f293ab7 (<rhansby@gmail.com>     2015-09-22 14:41:37  26    ) import android.support.v7.widget.LinearLayoutManager;
# retrieve all lines between the lines for the chunk in the current version
# if any deleted lines are present, collect the authors names and how many lines they deleted.
# regex for matching the line information from difflame:
#   (-|\+|\s)?\S{8} \(<(.*?)>\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\d+)?\s+(\d+)?
def find_deleted_lines_authors(merge_base_sha, line_start, line_end, file_path):
    command = f"./{starting_folder}/difflame.py {merge_base_sha} -e -- {file_path}"
    output = execute_command(command).split('\n')
    pattern = re.compile(r"(-|\+|\s)?\S{8} \(<(.*?)>\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\d+)?\s+(\d+)?\)")
    start = False
    authors = {}
    for line in output:
        # print(line)
        x = pattern.match(line)
        if x is not None:
            line_type = x.group(1)
            author = x.group(2)
            # original_line = x.group(3)
            merge_line = x.group(4)
            # print(f"line type: {line_type}  author: {author} original line: {original_line} merge_line: {merge_line}  line_end: {line_end}")
            # print(line)
            if merge_line is not None and int(merge_line) > int(line_end):
                break
            if merge_line is not None and int(merge_line) <= int(line_start):
                start = True
            if start:
                if line_type == '-' and merge_line is None:
                    if author in authors:
                        authors[author]+=1
                    else:
                        authors[author]=1

    return authors

# NOT USED \/
# use git blame using the chunk information (line_start, line_end, HEAD) and the merge base commit 
#   git blame -L line_start,line_end --show-email --reverse commit_base..HEAD file_path
# as a result, each line of the output will display whats the last commit (last_commit) where such line appeared
# example: 
    # d2164043d (<rhansby@gmail.com> 2015-09-21 12:04:45 -0700  20) import android.content.res.TypedArray;
    # in this case, last_commit is d2164043d
# for each blame output line, find what is the next commit after last_commit where the line appeared. i.e. the commit where the line as deleted
    # git log --reverse --ancestry-path last_commit..HEAD
    # get the commit author
# assumption: the chunk delimitation (line_start, line_end) is valid for the current replayed merge. However,
# such line numbers might be different in the previous version. 
# We assume that such delimitation entails at least one line which was deleted
        

def execute_command(command):
    try:
        my_env = os.environ.copy()
        print(command)
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env)
        # print(result)
        return result
    except subprocess.CalledProcessError as e:
        pass
        # print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return ''

def main():
    df = pd.read_csv(configs.INITIAL_DATASET_PATH)
    starting_folder = pathlib.Path(__file__).parent.absolute()
    # left_sha = "afd2d45cacac80d86d7920788cf5a16a4d3c3bbc"
    # right_sha = "98e035abc3c37c07e866c8dc0416cc60caa694e2"
    # project = "cwensel/cascading"
    # file_path = "cwensel/cascading/src/core/cascading/flow/planner/FlowStepJob.java"
    # file_path = file_path.replace(f"{project}/", "")
    # line_start = 219
    # line_end = 224 
    # line_separator = 221
    # project_folder = "/mnt/c/Users/HelenoCampos/Documents/workspace_test/analysis/cascading"
    for index, row in df.iterrows():
        left_sha = row['leftsha']
        right_sha = row['rightsha']
        project = row['project']
        merge_base = row['basesha']
        file_path = row['path'].replace(f"{project}/", "")
        line_start = row['line_start']
        line_end = row['line_end']
        line_separator = row['line_separator']
        project_folder = f"{configs.REPOS_PATH}/{project}"
        if os.path.exists(project_folder):
            os.chdir(project_folder)
            if merge(left_sha, right_sha):
                print('=============================')
                print(f"{row['chunk_id']}: {project} {row['sha']} {file_path} ")
                print('--------WHOLE CONFLICT----------')
                show_file_lines(f'{os.getcwd()}/{file_path}', line_start, line_end)
                print('------------------')
                get_chunks_author(file_path, line_start, line_end, line_separator, merge_base)
                input()
        os.chdir(starting_folder)

# def main():
#     df = pd.read_csv(configs.INITIAL_DATASET_PATH)
#     starting_folder = pathlib.Path(__file__).parent.absolute()
#     collected_commits = set()
#     extracted_data = []
#     print(f'Starting the collection process for {len(df)} chunks...')
#     current_index = 0
#     for index, row in df.iterrows():
#         current_index +=1
#         status = (current_index / len(df)) * 100
#         print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}")
#         commit_index = f"{row['project']}-{row['sha']}"
#         if(commit_index not in collected_commits): # since these attributes are related to the commit and not to the chunk, we only collect it if the merge commit (sha) is not already collected
#             data = []
#             data.append(row['chunk_id'])
#             data.append(row['sha'])
#             data.append(row['project'])
#             project_folder = f"{configs.REPOS_PATH}/{row['project']}"
#             if os.path.exists(project_folder):
#                 os.chdir(project_folder)
#                 left_insertions, left_deletions = get_number_changed_lines(row['sha'], row['leftsha'])
#                 right_insertions, right_deletions = get_number_changed_lines(row['sha'], row['rightsha'])
#                 # print(f'left insertions: {left_insertions}  / left deletions: {left_deletions} / right insertions: {right_insertions}  /right deletions: {right_deletions}')
#                 conclusion_delay = get_conclusion_delay(row['leftsha'], row['rightsha'])
#                 keywords_frequency = get_keywords_frequency(row['leftsha'], row['rightsha'], row['basesha'])
#                 # print(f'conclusion delay: {conclusion_delay} | keywords frequency: {keywords_frequency}')
#                 data.extend([left_insertions, left_deletions, right_insertions, right_deletions,conclusion_delay])
#                 for keyword, frequency in keywords_frequency.items():
#                     data.append(frequency)
#                 collected_commits.add(commit_index)
#                 extracted_data.append(data)
#                 os.chdir(starting_folder)

#     os.chdir(starting_folder)
#     columns = ['chunk_id', 'sha', 'project', 'left_lines_added', 'left_lines_removed', 'right_lines_added', 'right_lines_removed', 'conclusion_delay']
#     columns.extend(['keyword_fix', 'keyword_bug', 'keyword_feature', 'keyword_improve', 'keyword_document', 'keyword_refactor', 'keyword_update'])
#     columns.extend(['keyword_add', 'keyword_remove', 'keyword_use', 'keyword_delete', 'keyword_change'])

#     pd.DataFrame(extracted_data, columns = columns).to_csv(f'{configs.DATA_PATH}/collected_attributes1.csv', index=False)

# main()
main()