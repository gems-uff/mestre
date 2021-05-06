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

def get_file_name(conflict_lines, line):
    if ':' in conflict_lines[line]:
        parts = conflict_lines[line].split()
        if len(parts) > 0:
            parts = parts[1].split(':')
            if len(parts) > 0:
                return parts[1]
                
    return None

left_file = get_file_name(conflict_lines, conflict_start)
right_file = get_file_name(conflict_lines, conflict_end)

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




def get_base_file_name(merge_base, file_name, current_commit, repo_path):
    # get the list of commits between the commit where we know the file name and the merge common ancestor
    #   git rev-list --ancestry-path merge_base..current_commit
    command = f"git rev-list --ancestry-path {merge_base}..{current_commit}"
    log_output = execute_command(command, repo_path)
    commits = log_output.strip().split('\n')[::-1]
    commits.insert(0, merge_base)
    print(commits)
    if len(commits) > 1:
        current_file = file_name        
        # for each pair of commits, until we reach the common ancestor, perform diffs
        for previous, current in zip(commits, commits[1:]):
            command = f'git diff {previous} {current}'
            diff_output = execute_command(command, repo_path)
            previous_name = inspect_diff(diff_output.split('\n'), current_file)
            if previous_name !=None:
                current_file = previous_name
        return current_file


print(get_base_file_name("e48fcfd18fb8108e97bb3e116459b549bc0d2d4e", "app/src/main/java/org/connectbot/GeneratePubkeyActivity.java", "HEAD", repo_path))