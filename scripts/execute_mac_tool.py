import pandas as pd
import configs
import subprocess
import os
import sys
import time

def execute_command(command):
    executed = False
    try:
        my_env = os.environ.copy()
        subprocess.check_call(command, stdout=sys.stdout, stderr=subprocess.STDOUT, shell=True, env=my_env)
        executed = True
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return executed

def get_macTool_ouput(project_name, sha):
    project_file = f"{configs.MAC_TOOL_OUTPUT}/{project_name}-general.csv"
    if(os.path.exists(project_file)):
        macTool_output = pd.read_csv(project_file, delimiter=",")
        commit = macTool_output[macTool_output['Hash'] == sha]
        return commit
    else:
        return []

def main():
    mac_tool_command = f'java -jar {configs.MAC_TOOL_PATH} {configs.INITIAL_DATASET_PATH} {configs.REPOS_PATH} {configs.MAC_TOOL_OUTPUT}'
    execute_command(mac_tool_command)
    labelled_dataset = pd.read_csv(configs.LABELLED_DATASET_PATH)

    columns = ["chunk_id", "project", "Branching time", "Merge isolation time", "Devs 1", "Devs 2", "Different devs", "Same devs",	"Devs intersection", "Commits 1", "Commits 2", "Changed files 1", "Changed files 2", "Changed files intersection"]
    current_index = 0
    dataset = []
    for index, row in labelled_dataset.iterrows():
        data = []
        current_index +=1
        status = (current_index / len(labelled_dataset)) * 100
        print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}")
        sha = row['sha']
        chunk_id = row['chunk_id']
        commit = get_macTool_ouput(row['project_name'], sha)
        data.append(chunk_id)
        data.append(row['project'])
        if len(commit) == 1:
            for column in columns:
                if column != 'chunk_id' and column != 'project':
                    data.append(commit.iloc[0][column])
        else:
            data.extend([None] * (len(columns)-2))
        dataset.append(data)
    pd.DataFrame(dataset, columns = columns).to_csv(f'{configs.DATA_PATH}/macTool_output.csv', index=False)

main()