# this version of the collect_chunk_authors script uses the diff3 format
# when one side of the chunk is empty.
# the diff3 format includes the base content, which is used to locate the chunk
# lines in the base file and then the lines are used to locate the author information
# in the difflame output.
# Unfortunately, we found that in some cases diff3 produces a different number of chunks than
# the default merge in git. This makes this strategy unfeasible.
# Case (case 3 in documentation) where it fails: 
#   chunk id 777400 project connectbot/connectbot 
#   merge commit48e9c2f37ca757c3dfbd882417670ed67192725f
#   file: app/src/main/java/org/connectbot/util/UberColorPickerDialog.java
#   begin_line 85   end_line 88   separator_line 87

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
pattern = re.compile(r"(-|\+|\s)?\S{8} \(<(.*?)>\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\d+)?\s+(\d+)?\) (.*?)\n")

class DifflameLine:
    # line_type = ''
    # author = ''
    # original_line = -1
    # merge_line = -1
    # content = ''

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

def checkout_diff3_version(file_path):
    command = f"git checkout --conflict=diff3 {file_path}"
    execute_command(command)

def checkout_default_version(file_path):
    command = f"git checkout --conflict=merge {file_path}"
    execute_command(command)

def blame(file_path, line_start, line_end):
    command = f"git blame --show-email -L {line_start},{line_end} {file_path}"
    return execute_command(command)

def show_file_lines(file_path, start, end):
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines[start:end+1]:
        print(line)

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

# strategy:
# redo the merge
# git blame the conflicted file
# inspect the result
def get_chunks_author(file_path, line_start, line_end, line_separator, merge_base):
    blame_output_left = blame(file_path, line_start, line_separator)
    print(blame_output_left)
    # print(f'blame left:  \n {blame_output_left}')
    authors_left = extract_authors(blame_output_left)
    if len(authors_left) == 0:
        authors_left = get_empty_chunk_authors(merge_base, line_start, line_separator, file_path, line_start)
    # print(f'authors_left: {authors_left}')
    # print('------------------')
    blame_output_right = blame(file_path, line_separator+1, line_end)
    # print(f'blame right: \n {blame_output_right}')
    authors_right = extract_authors(blame_output_right)
    if len(authors_right) == 0:
        authors_right = get_empty_chunk_authors(merge_base, line_separator+1, line_end, file_path, line_start)
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
                if(line_parts != []):
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
def get_empty_chunk_authors(merge_base_sha, line_start, line_end, file_path, whole_chunk_start):
    authors = {}

    file_default = file_diff3 = None
    # salva o arquivo em conflito em uma lista
    with open(file_path, 'r') as f:
        file_default = f.readlines()

    # pede o arquivo no formato do diff3 e salva numa lista
    checkout_diff3_version(file_path)
    with open(file_path, 'r') as f:
        file_diff3 = f.readlines()

    # retorna o arquivo para o formato original
    checkout_default_version(file_path)
    
    # pega o mapeamento entre os arquivos
    import conflict_map
    chunks = conflict_map.get_chunks_mapping(file_default, file_diff3)
    
    # procura o chunk que começa na linha_start no default_file e pega o conteúdo da base
    chunk = None
    for c in chunks:
        if line_start+1 >= c.default_line_start and line_start+1 <= c.default_line_end:
            chunk = c
    
    if chunk != None:
        base_content = chunk.base_content
    else:
        base_content = []

    # procura o conteúdo da base no difflame e seleciona as linhas que têm esse conteúdo
    
    selected_lines = []
    command = f"./{starting_folder}/difflame.py {merge_base_sha} -e -- {file_path}"
    # print(command)
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
            
            start = end = 0
            for i in range(len(content)):
                if content[i] == base_content[0] and content[i:i+len(base_content)] == base_content:
                    start = i
                    end = i + len(base_content)
            selected_lines = difflame_lines[start:end]
        # input()
        # # print(content)
        # # input()
        # print(base_content)
        # input()
        # print(selected_lines)
        # input()
        # analisa as linhas selecionadas    
        

        for line in selected_lines:

            if line.line_type == '-' and line.merge_line is None:
                if line.author in authors:
                    authors[line.author]['deleted']+=1
                else:
                    authors[line.author]={'modified': 0, 'deleted': 1, 'moved/renamed':False}
            if line.line_type == '+' and line.original_line is None:
                if not line.author in authors:
                        authors[line.author]={'modified': 0, 'deleted': 0, 'moved/renamed':True}
        # print(authors)
        # input()
    else:
        print(command)
        print(difflame_output)
        print('possivel erro')
        input()
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
    df = pd.read_csv(configs.INITIAL_DATASET_PATH)
    starting_folder = pathlib.Path(__file__).parent.absolute()
    data = []
    columns = ["chunk_id", "left_size", "right_size", "authors_left", "authors_right"]
    current_index = 0
    save_every = 50
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
        if os.path.exists(project_folder):
            os.chdir(project_folder)
            if merge(left_sha, right_sha):
                print('=============================')
                print(f"{row['chunk_id']}: {project} {row['sha']} {file_path} ")
                print('--------WHOLE CONFLICT----------')
                show_file_lines(f'{os.getcwd()}/{file_path}', line_start, line_end)
                print('------------------')
                left_size = line_separator - line_start - 1
                right_size = line_end - line_separator - 1
                authors_left_dict, authors_right_dict = get_chunks_author(file_path, line_start, line_end, line_separator, merge_base)
                data.append([row['chunk_id'], left_size, right_size, authors_left_dict, authors_right_dict])
                # input()
        os.chdir(starting_folder)
        if current_index % save_every == 0:
            pd.DataFrame(data, columns=columns).to_csv(f"{configs.DATA_PATH}/chunk_authors.csv", index=False)        
    pd.DataFrame(data, columns=columns).to_csv(f"{configs.DATA_PATH}/chunk_authors.csv", index=False)

main()