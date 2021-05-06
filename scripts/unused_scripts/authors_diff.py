# UNFINISHED
# the idea of this script is to perform diffs between each pair of commits between the merge parents and the merge base
#   on each diff, we need to keep track of the lines involved in the conflict and if it touches the conflict chunk
#   we collect the author information and how many lines he added/deleted to the conflict chunk region.

import subprocess
import os



def execute_command(command, working_directory):
    try:
        my_env = os.environ.copy()
        print(command)
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, encoding="latin-1", cwd=working_directory)
        # print(result)
        return result
    except subprocess.CalledProcessError as e:
        # pass
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return ''

conflict_start = 56
conflict_separator = 60
conflict_end = 61

conflict_lines = []
with open('replay.txt') as f:
    conflict_lines = f.readlines()

file_name = "app/src/main/java/org/connectbot/GeneratePubkeyActivity.java"
repo_path = "/mnt/c/Users/HelenoCampos/Dropbox/SHARED_PC_LAB/Doutorado/colaboracao/conflicts_classifier/conflict-resolution-mining/repos/connectbot/connectbot/"

def inspect_diffs(merge_base, file_name, current_commit, repo_path):
    # get the list of commits between the commit where we know the file name and the merge common ancestor
    #   git rev-list --ancestry-path merge_base..current_commit
    command = f"git rev-list --ancestry-path {merge_base}..{current_commit}"
    log_output = execute_command(command, repo_path)
    commits = log_output.strip().split('\n')
    commits.append(merge_base)
    print(commits)
    if len(commits) > 1:
        current_file = file_name        
        # for each pair of commits, until we reach the common ancestor, perform diffs
        for i in range(len(commits) - 1):
            commit1 = commits[i]
            commit2 = commits[i+1]
            command = f'git diff {commit1} {commit2} -- {file_name}'
            print(command)
            # diff_output = execute_command(command, repo_path)
            # previous_name = inspect_diff(diff_output.split('\n'), current_file)
            # if previous_name !=None:
            #     current_file = previous_name
        return current_file

merge_base = "3adb2f6dd0ea442facd89869e237887c432c6795"
current_commit = "ab3c048f47c6390ea1b6e42d09055f70938ae1ee"
file_path = "cascading-core/src/main/java/cascading/flow/planner/process/ProcessEdge.java"
repo_path = "/mnt/c/Users/HelenoCampos/Documents/workspace_test/conflict-resolution-mining/repos/cwensel/cascading"
# repo_path = "C:/Users/HelenoCampos/Documents/workspace_test/conflict-resolution-mining/repos/cwensel/cascading"

print(inspect_diffs(merge_base, file_path, current_commit, repo_path))