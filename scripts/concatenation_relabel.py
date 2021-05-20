import database
import pandas as pd
import os
import subprocess
import time
import configs

def execute_command(command):
    output = ''
    try:
        my_env = os.environ.copy()
        output = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env)
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return output

def get_v1(conflict):
    V1 = []
    start = False
    for line in conflict:
        if("<<<<<<<" in line):
            start = True
            continue
        if("=======" in line):
            break
        if(start == True):
            V1.append(line)
    return V1

def get_v2(conflict):
    start = False
    V2 = []
    for line in conflict:
        if("=======" in line):
            start = True
            continue
        if(">>>>>>>" in line and start):
            break
        if(start == True):
            V2.append(line)
    return V2   

#returns the context before and after the conflict
def get_context(conflict):
    before = []
    is_before = True
    after = []
    is_after = False
    for line in conflict:
        if("<<<<<<<" in line):
            is_before = False
            continue
        if(">>>>>>>" in line and not is_before):
            is_after = True
            continue
        if is_before:
            before.append(line)
        if is_after:
            after.append(line)
    return before, after

def print_lines(lines):
    print(*lines, sep ="\n")

def write_list_to_file(content, file_name):
    with open(file_name, 'w') as file:
        for line in content:
            file.write(f"{line}\n")


def main():
    df = pd.read_csv(configs.INITIAL_DATASET_PATH, header=0)
    count = 0
    print('Analyzing...')
    for index, row in df.iterrows():
        print(f"{time.ctime()} #### {(index/len(df) * 100):.2f}%")
        if(row['developerdecision'] == 'Concatenation'):
            count+=1
            chunk_id = row['chunk_id']
            # print(chunk_id)
            conflict = database.get_conflict(chunk_id)
            v1 = get_v1(conflict)
            v2 = get_v2(conflict)
            context_before, context_after = get_context(conflict)
            solution = database.get_solution(chunk_id)
            
            # print_lines(conflict)
            # print('--------------------------')

            # print_lines(v1)
            # print('------------------')
            # print_lines(v2)
            # print('-----------solution---------------')
            # print_lines(solution)
            
            write_list_to_file(v1, 'v1')
            write_list_to_file(v2, 'v2')
            write_list_to_file(context_before, 'context1')
            write_list_to_file(context_after, 'context2')
            write_list_to_file(solution, 'solution')
            concatenation_type = execute_command('java -jar classifyConcatenation.jar v1 v2 context1 context2 solution')
            #print(f"{chunk_id}: {concatenation_type}")
            df.at[index, 'developerdecision'] = concatenation_type.strip()
            os.remove("v1")
            os.remove("v2")
            os.remove("context1")
            os.remove("context2")
            os.remove("solution")
    df.to_csv(configs.LABELLED_DATASET_PATH, index=None)

main()