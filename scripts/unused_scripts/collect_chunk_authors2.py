# this version of the collect_chunk_authors script uses the base lines as context to locate the chunk in difflame output
# it does not work for all cases (some lines deletion cases are not retrieved). 


import pandas as pd
import configs
import subprocess
import os
import time
import sys
import pathlib
from datetime import datetime
import re
import blame_parser

starting_folder = ''
pattern = re.compile(r"(-|\+|\s)?\S{8} \(<(.*?)>\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\d+)?\s+(\d+)?\) (.*?)\n")
problems_log_file_path = ''
current_chunk = ''


class DifflameLine:
    def __init__(self, line_type, author, original_line, merge_line, content):
        self.line_type = line_type
        self.author = author
        self.original_line = original_line
        self.merge_line = merge_line
        self.content = content

# get commit message for a commit:
    # git show -s --format=%B commit_SHA
def get_commit_message(commit_SHA):
    command = f'git show -s --format=%B {commit_SHA}'
    return execute_command(command).strip()

def reset():
    command = "git reset --hard"
    output = execute_command( command)
    if "HEAD is now at" in output.strip():
        return True
    else:
        log_problem('INCONSISTENT REPOSITORY STATE')
    return False

def checkout(sha):
    if reset():
        command = f"git checkout {sha}"
        output = execute_command(command)
        if "HEAD is now at" in output.strip():
            return True
        else:
            log_problem('INCONSISTENT REPOSITORY STATE')
    return False

def merge(left_sha, right_sha):
    if checkout(left_sha):
        command = f"git merge {right_sha}"
        output = execute_command(command)
        return True
    return False

def get_commit_date(SHA):
    command = f"git show -s --format=%ci {SHA}"
    output = execute_command(command).strip()
    output = output.split('\n')
    if len(output) > 0:
        date = output[0] # takes only the date and discard possible warnings
    else:
        date = None
    return date

def checkout_diff3_version(file_path):
    command = f"git checkout --conflict=diff3 {file_path}"
    execute_command(command)

def checkout_default_version(file_path):
    command = f"git checkout --conflict=merge {file_path}"
    execute_command(command)

def blame_file(file_path):
    command = f"git blame --show-email {file_path}"
    return execute_command(command)

def blame_lines(file_path, line_start, line_end):
    command = f"git blame --show-email -L {line_start},{line_end} {file_path}"
    return execute_command(command)

def show_file_lines(file_path, start, end):
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines[start:end+1]:
        print(line)

def get_file_lines_by_commit(commit_sha, file_path):
    command = f"git show {commit_sha}:{file_path}"
    print(command)
    return execute_command(command)

def get_difflame_lines(difflame_output):
    global pattern
    lines = []
    for line in difflame_output:
        x = pattern.match(line+"\n")
        if x is not None:
            line_type = x.group(1)
            author = x.group(2)
            original_line = x.group(3)
            merge_line = x.group(4)
            content = x.group(5)+"\n"
            lines.append(DifflameLine(line_type, author, original_line, merge_line, content))
    return lines

def setup_log_file():
    global problems_log_file_path
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H-%M-%S")
    problems_log_file_path = os.path.abspath(f'{configs.LOGS_PATH}/collect_chunk_authors-log-{now}.txt')
    if not os.path.exists(configs.LOGS_PATH):
        os.mkdir(configs.LOGS_PATH)

def log_problem(problem):
    global problems_log_file_path
    global current_chunk
    print(problems_log_file_path)
    with open(problems_log_file_path,'a') as f:
        f.write(f'{current_chunk}: {problem}')


# strategy:
# redo the merge
# git blame the conflicted file
# inspect the result
def get_chunks_author(file_path, line_start, line_end, line_separator, merge_base):
    blame_output_left = blame_lines(file_path, line_start, line_separator)
    # print(f'blame left:  \n {blame_output_left}')
    authors_left = extract_authors(blame_output_left)
    if len(authors_left) == 0:
        authors_left = get_empty_chunk_authors(merge_base, line_start, line_separator, file_path, line_start)
    # print(f'authors_left: {authors_left}')
    # print('------------------')
    blame_output_right = blame_lines(file_path, line_separator+1, line_end)
    # print(f'blame right: \n {blame_output_right}')
    authors_right = extract_authors(blame_output_right)
    if len(authors_right) == 0:
        authors_right = get_empty_chunk_authors(merge_base, line_separator+1, line_end, file_path, line_start)
    # print(f'authors_right: {authors_right}')
    return authors_left, authors_right
        

def extract_authors(blame_output):
    authors = {}
    lines = blame_output.split('\n')
    # print('blame lines:')
    # print(lines)
    # input()
    blame = blame_parser.Blame(lines)
    # print(blame.lines)
    insideChunk = False
    # print('------------------------ AAA')
    for line in blame.lines:
        # print(f'line: {line.line_content}')
        if '<<<<<<<' in line.line_content or '=======' in line.line_content:
            insideChunk = True
        elif insideChunk:
            if line.author != 'not.committed.yet':
                if line.author in authors:
                    authors[line.author]['modified']+=1
                else:
                    authors[line.author]={'modified': 1, 'deleted': 0, 'moved/renamed':False}
    # print(authors)
    # input()
    return authors

# if the searched file is present in diff, return its previous name
def inspect_diff(diff_lines, searched_file):
    searched_file = f"b/{searched_file}"
    for line in diff_lines:
        if searched_file in line: # diff --git a/src/org/connectbot/GeneratePubkeyActivity.java b/app/src/main/java/org/connectbot/GeneratePubkeyActivity.java
            parts = line.split()
            if len(parts) == 4:
                file_name = parts[2]
                return file_name.replace("a/",'')
    return None

# performs diffs between each version between the current_commit and merge_base to find the name of a file in the base version
def get_base_file_name(merge_base, file_name, current_commit):
    # get the list of commits between the commit where we know the file name and the merge common ancestor
    #   git rev-list --ancestry-path merge_base..current_commit
    command = f"git rev-list --ancestry-path {merge_base}..{current_commit}"
    log_output = execute_command(command)
    commits = log_output.strip().split('\n')[::-1]
    commits.insert(0, merge_base)
    print(commits)
    if len(commits) > 1:
        current_file = file_name        
        # for each pair of commits, until we reach the common ancestor, perform diffs
        for previous, current in zip(commits, commits[1:]):
            command = f'git diff {previous} {current}'
            diff_output = execute_command(command)
            previous_name = inspect_diff(diff_output.split('\n'), current_file)
            if previous_name !=None:
                current_file = previous_name
        return current_file

# returns content of the file in the merge base version
def get_base_file_lines(merge_base_sha, file_path):
    command = f"git show {merge_base_sha}:{file_path}"
    output = execute_command(command)
    if 'exists on disk, but not in' in output: # the file has been renamed or does not exist in base version
        file_path = get_base_file_name(merge_base_sha, file_path, "HEAD")
        command = f"git show {merge_base_sha}:{file_path}"
        output = execute_command(command)
        if 'exists on disk, but not in' in output:
            return None
    return output.split('\n')

def normalize_strings(strings_list):
    for i in range(len(strings_list)):
        strings_list[i] = strings_list[i].replace('\t', '').replace(' ','').replace('\n','')
    return strings_list

# find the minimal elements sequence match between base and target lists,
#  such that there is only one such sequence in base: 
# example:
#   base = 'abcdefghijklmnopqefhrstuvabxyzefgh1230212'
#   target = 'abc'
#   returns 0,3 [a,b,c]
# example 2:
#   base = 'abcdefghijklmnopqefhrstuvabxyzefgh1230212'
#   target = 'ghij'
#   returns 6,9 [g,h,i]
def search_context(base, target, reverse_target):
    size = 1
    target_position_start = 0
    candidates = [1]

    while len(candidates) > 0:
        candidates.clear()
        target_sequence = target[target_position_start:target_position_start+size]
        if reverse_target:
            target_sequence = target_sequence[::-1]
        target_sequence = normalize_strings(target_sequence)
        for i in range(len(base)):
            base_sequence = base[i:i+size]
            base_sequence = normalize_strings(base_sequence)
            if base_sequence == target_sequence:
                #start, end
                candidates.append([i,i+size])
        # found the match
        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1]
        
        size+=1
    return None,None

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
def get_empty_chunk_authors(merge_base_sha, line_start, line_end, file_path, whole_chunk_start):
    authors = {}
    global starting_folder
    # get blame output for the conflicted file
    blame_output = blame_file(file_path)

    # parse the blame output
    
    blame = blame_parser.Blame(blame_output.split('\n'))
    
    # find all lines that appear before the conflict chunk and that existed in the base version (we check this by date)
    base_date = get_commit_date(merge_base_sha)
    if base_date != None:
        before_lines = blame.find_lines_by_date_before_line_limit(base_date, whole_chunk_start)
        after_lines = blame.find_lines_by_date_after_conflict_end_mark(base_date, whole_chunk_start)
    else:
        log_problem('PROBLEM WITH BASE DATE')
        return {}

    # get all blame lines that existed in the base version and are not in a conflict chunk
    # TODO: getting those lines from the blame output is not 100% reliable. In some cases this 
    #   strategy will fail. Example: chunk id 776777, case 5.
    base_lines = blame.get_all_lines_content_before_date(base_date)
    # base_lines = get_base_file_lines(merge_base_sha, file_path)
    # print(base_date)
    print('base lines')
    print(base_lines)
    # input()
    if len(base_lines) == 0:
        log_problem(f'PROBLEM RETRIEVING BASE FILE. base_sha: {merge_base_sha}')
        return {}
    # print('base_lines')
    # print(base_lines)
    # input()
    # print()
    print('before_lines')
    before_lines_content = []
    for line in before_lines:
        before_lines_content.append(line.line_content)
    print(before_lines_content)
    input()

    # print('after_lines')
    after_lines_content = []
    for line in after_lines:
        after_lines_content.append(line.line_content)
    # print(after_lines_content)
    # input()
    
    # find the minimal number of lines that is enough to locate the conflicting chunk in the base version
    #   those are the context lines
    before_context_start, before_context_end = search_context(base_lines, before_lines_content, True)
    after_context_start, after_context_end = search_context(base_lines, after_lines_content, False)

    # find the region of interest (location of the conflict chunk) in the base file using the context lines
    before_context_lines = base_lines[before_context_start:before_context_end]
    after_context_lines = base_lines[after_context_start:after_context_end]
    print('before context')
    print(before_context_lines)

    print('after context')
    print(after_context_lines)
    input()
    
    # finally, using the before and after context lines, search those lines in difflame output
    # the content between the context lines is what we are searching for
    selected_lines = []
    command = f"{starting_folder}/difflame.py {merge_base_sha} -e -- {file_path}"
    print(command)
    difflame_output = execute_command(command).split('\n')
    difflame_lines = get_difflame_lines(difflame_output)
    if len(difflame_output) > 1:
        if 'new file mode' in difflame_output[1]:
            for difflame_line in difflame_lines:
                if int(difflame_line.merge_line) >= whole_chunk_start +1 and int(difflame_line.merge_line) < line_end-1:
                    # print(difflame_line.content)
                    selected_lines.append(difflame_line)
        else:
            content = []
            
            for line in difflame_lines:
                content.append(line.content)
            
            content = normalize_strings(content)
            before_context_lines = normalize_strings(before_context_lines)
            after_context_lines = normalize_strings(after_context_lines)
            
            start = end = 0
            for i in range(len(content)):
                if content[i] == before_context_lines[0] and content[i:i+len(before_context_lines)] == before_context_lines:
                    start = i+len(before_context_lines)
                if content[i] == after_context_lines[0] and content[i:i+len(after_context_lines)] == after_context_lines:
                    end = i
            # print('start:')
            # print(start)
            # print('end')
            # print(end)
            selected_lines = difflame_lines[start:end]
            print(selected_lines)
            
        # input()
        # # print(content)
        # # input()
        # print(base_content)
        # input()
        # print('selected_lines')
        
        # analyze the difflame selected lines to extract author information
        for line in selected_lines:
            # print(line.content)
            if line.line_type == '-' and line.merge_line is None:
                if line.author in authors:
                    authors[line.author]['deleted']+=1
                else:
                    authors[line.author]={'modified': 0, 'deleted': 1, 'moved/renamed':False}
            if line.line_type == '+' and line.original_line is None:
                if not line.author in authors:
                        authors[line.author]={'modified': 0, 'deleted': 0, 'moved/renamed':True}
        
        # input()
        # print('authors')
        # print(authors)
        # input()
    else:
        # print(difflame_output)
        log_problem(f'POSSIBLY DIFFLAME ERROR command: {command}')
        return {}
    return authors

def execute_command(command):
    try:
        my_env = os.environ.copy()
        # print(command)
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, encoding="latin-1")
        # print(result)
        return result
    except subprocess.CalledProcessError as e:
        pass
        # print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return ''

def main():
    global starting_folder
    global current_chunk
    df = pd.read_csv(configs.INITIAL_DATASET_PATH)
    df = df[df['chunk_id'] == 776777]
    starting_folder = pathlib.Path(__file__).parent.absolute()
    data = []
    columns = ["chunk_id", "left_size", "right_size", "authors_left", "authors_right"]
    current_index = 0
    save_every = 50
    setup_log_file()
    for index, row in df.iterrows():
        current_index +=1
        status = (current_index / len(df)) * 100
        print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}", flush=True)
        left_sha = row['leftsha']
        right_sha = row['rightsha']
        project = row['project']
        merge_base = row['basesha']
        file_path = row['path'].replace(f"{project}/", "")
        line_start = row['line_start']
        line_end = row['line_end']
        line_separator = row['line_separator']
        project_folder = f"{configs.REPOS_PATH}/{project}"
        current_chunk = row['chunk_id']
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
                if len(authors_left_dict) > 0 and len(authors_right_dict) > 0:
                    data.append([row['chunk_id'], left_size, right_size, authors_left_dict, authors_right_dict])
                    input()
                else:
                    log_problem('empty authors')
        else:
            log_problem('REPOSITORY NOT FOUND')
        os.chdir(starting_folder)
        if current_index % save_every == 0:
            print('Saving collected data so far.')
            pd.DataFrame(data, columns=columns).to_csv(f"{configs.DATA_PATH}/chunk_authors.csv", index=False)        
    pd.DataFrame(data, columns=columns).to_csv(f"{configs.DATA_PATH}/chunk_authors.csv", index=False)

main()