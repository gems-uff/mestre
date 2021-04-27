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
        authors_left = get_empty_chunk_authors(merge_base, line_start, line_separator, file_path)
    # print(f'authors_left: {authors_left}')
    # print('------------------')
    blame_output_right = blame(file_path, line_separator+1, line_end)
    # print(f'blame right: \n {blame_output_right}')
    authors_right = extract_authors(blame_output_right)
    if len(authors_right) == 0:
        authors_right = get_empty_chunk_authors(merge_base, line_separator+1, line_end, file_path)
    # print(f'authors_right: {authors_right}')
    return authors_left, authors_right
        

def extract_authors(blame_output):
    authors = {}
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
                if author in authors:
                    authors[author]['modified']+=1
                else:
                    authors[author]={'modified': 1, 'deleted': 0, 'moved/renamed':False}
    return authors


# how we find who participated in an empty side of a chunk:
# use difflame tool (https://github.com/eantoranz/difflame)
#   ./difflame.py {merge_base_sha} -e -- {file_path}
# example:
#   ./difflame.py 708e2db0bf5e3bfbb48bf94d604ef883970a2b92 -e -- app/src/main/java/org/connectbot/HostListActivity.java
# example of a line outputed by difflame:
#     12ef92bf (<kenny@the-b.org>       2008-11-14 10:07:29  29  20) import java.util.List;
# last commit where the line was modified | email of the author | timestamp of the commit | line number in the merge base | line number in the conflictingfile
# example of a deleted line and information about the commit that deleted it:
    # -0f293ab7 (<rhansby@gmail.com>     2015-09-22 14:41:37  26    ) import android.support.v7.widget.LinearLayoutManager;
# example of an added line:
#   +f7beb3b8 (<kenny@the-b.org>       2013-04-12 04:13:42    51) import android.widget.SeekBar;
# retrieve all lines between the lines for the empty chunk in the current version
# if any deleted lines are present, collect the authors names and how many lines they deleted.
# if the chunk is empty in the merge and an added line is shown in blame, it means the file was renamed or moved.
#   in this case, we collect the authors who authored such lines.
# regex for matching the line information from difflame:
#   (-|\+|\s)?\S{8} \(<(.*?)>\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\d+)?\s+(\d+)?
# returns authors who participated in the empty chunk
def get_empty_chunk_authors(merge_base_sha, line_start, line_end, file_path):
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
            original_line = x.group(3)
            merge_line = x.group(4)
            # print(f"line type: {line_type}  author: {author} original line: {original_line} merge_line: {merge_line}  line_end: {line_end}  line_start: {line_start}")
            # print(line)
            if merge_line is not None and int(merge_line) > int(line_end):
                break
            if merge_line is not None and int(merge_line) >= int(line_start):
                start = True
            if start:
                if line_type == '-' and merge_line is None:
                    if author in authors:
                        authors[author]['deleted']+=1
                    else:
                        authors[author]={'modified': 0, 'deleted': 1, 'moved/renamed':False}
                if line_type == '+' and original_line is None:
                    if not author in authors:
                        authors[author]={'modified': 0, 'deleted': 0, 'moved/renamed':True}

    return authors

def execute_command(command):
    try:
        my_env = os.environ.copy()
        # print(command)
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
    data = []
    columns = ["chunk_id", "left_size", "right_size", "authors_left", "authors_right"]
    current_index = 0
    for index, row in df.iterrows():
        current_index +=1
        status = (current_index / len(df)) * 100
        print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}")
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
                # print('=============================')
                # print(f"{row['chunk_id']}: {project} {row['sha']} {file_path} ")
                # print('--------WHOLE CONFLICT----------')
                # show_file_lines(f'{os.getcwd()}/{file_path}', line_start, line_end)
                # print('------------------')
                left_size = line_separator - line_start - 1
                right_size = line_end - line_separator - 1
                authors_left_dict, authors_right_dict = get_chunks_author(file_path, line_start, line_end, line_separator, merge_base)
                data.append([row['chunk_id'], left_size, right_size, authors_left_dict, authors_right_dict])
                # input()
        os.chdir(starting_folder)
    pd.DataFrame(data, columns=columns).to_csv(f"{configs.DATA_PATH}/chunk_authors.csv", index=False)

main()