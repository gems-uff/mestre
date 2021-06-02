import pandas as pd
import configs
import subprocess
import os
import time
import re


def execute_command(command, path):
    try:
        my_env = os.environ.copy()
        # print(command)
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, cwd=path, encoding="latin-1")
        # print(result)
        return result
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
        return '_command_error'

def get_commit_message(commit_SHA, path):
    command = f'git show -s --format=%B {commit_SHA}'
    return execute_command(command, path).strip()

def write_failed_chunks(failed):
    with open(f'{configs.LOGS_PATH}/merge_types_failed.txt', 'w') as file:
        for failed_chunk in failed:
            file.write(f"{failed_chunk[0]}:{failed_chunk[1]}\n")

def main():
    df = pd.read_csv(f"{configs.DATA_PATH}/macTool_output.csv")
    data = []
    chunks_failed = []
    current_index = 0
    branch_merge_pattern = re.compile(r"[m|M]erge[d]?(.*)( [b|B]ranch)?(.*) ((into (.*)|onto (.*)|from (.*) to (.*)))", re.IGNORECASE)
    for index, row in df.iterrows():
        current_index +=1
        status = (current_index / len(df)) * 100
        print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}", flush=True)
        chunk_id = row['chunk_id']
        project = row['project']
        project_folder = f"{configs.REPOS_PATH}/{project}"
        
        if os.path.exists(project_folder):
            devs1 = row['Devs 1']
            devs2 = row['Devs 2']
            if pd.isnull(devs1) or pd.isnull(devs2):
                chunks_failed.append([chunk_id, "UNKNOWN_DEVS"])
            else:
                merge_message = get_commit_message(row['sha'], project_folder)
                if merge_message == '_command_error':
                    chunks_failed.append([chunk_id, "CANNOT_RETRIEVE_COMMIT_MSG"])
                else:
                    has_branch_merge_message_indicator = branch_merge_pattern.search(merge_message) != None
                    has_multiple_devs_on_each_side = int(devs1) >=2 and int(devs2) >= 2 
                    data.append([chunk_id, row['project'], row['sha'][:10], devs1, devs2, merge_message[:255], has_branch_merge_message_indicator, has_multiple_devs_on_each_side])
        else:
            chunks_failed.append([chunk_id, "REPO_NOT_FOUND"])
    
    new_df = pd.DataFrame(data, columns=['chunk_id', 'project', 'merge_SHA', 'devs1', 'devs2', 'commit_message', 'has_branch_merge_message_indicator', 'has_multiple_devs_on_each_side'])
    new_df.to_csv(f"{configs.DATA_PATH}/merge_types_data.csv", index=False)
    write_failed_chunks(chunks_failed)

main()