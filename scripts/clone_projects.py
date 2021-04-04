# -*- coding: UTF-8 -*-
import os
import time
import subprocess
import pandas as pd
import configs


def clone_projects():
    if(not os.path.exists(configs.REPOS_PATH)):
        os.mkdir(configs.REPOS_PATH)
    current_index = 0
    df = pd.read_csv(configs.INITIAL_DATASET_PATH_TEST, header=0)
    print('Starting the clone process...')
    for index, row in df.iterrows():
        current_index +=1
        status = (current_index / len(df)) * 100
        print('{} ### {:.1f}% of rows processed. Processing project: {}'.format(time.ctime(), status, row['project']))
        folder = os.path.join(configs.REPOS_PATH, row['project'])
        if(not os.path.exists(folder)):
            command = 'git clone {} {}'.format(row['url'], folder)
            execute_command(command)

def execute_command(command):
    executed = False
    try:
        my_env = os.environ.copy()
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env)
        executed = True
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return executed
    
if __name__ == "__main__":
    clone_projects()
    