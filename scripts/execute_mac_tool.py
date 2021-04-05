import pandas as pd
import configs
import subprocess
import os
import time
import sys


def execute_command(command):
    executed = False
    try:
        my_env = os.environ.copy()
        subprocess.check_call(command, stdout=sys.stdout, stderr=subprocess.STDOUT, shell=True, env=my_env)
        executed = True
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return executed

def move_output_files():
    if(not os.path.exists(configs.MAC_TOOL_OUTPUT)):
        os.mkdir(configs.MAC_TOOL_OUTPUT)
    command = f"mv *.csv {configs.MAC_TOOL_OUTPUT}"
    execute_command(command)
    

def main():
    df = pd.read_csv('../data/INITIAL_DATASET.csv')
    repos_path = set()
    for index, row in df.iterrows():
        repos_path.add(f"{configs.REPOS_PATH}/{row['project']}")
        if(index > 1000):
            break
    mac_tool_command = f'java -jar {configs.MAC_TOOL_PATH}'

    current_index = 0
    for repo in repos_path:
        current_index +=1
        status = (current_index / len(repos_path)) * 100
        print('{} ### {:.1f}% of repos processed. Processing repo: {}'.format(time.ctime(), status, repo))    
        execute_command(f"{mac_tool_command} {repo}")
        move_output_files()
    

main()