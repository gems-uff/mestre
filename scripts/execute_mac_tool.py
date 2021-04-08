import pandas as pd
import configs
import subprocess
import os
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

def main():
    mac_tool_command = f'java -jar {configs.MAC_TOOL_PATH} {configs.INITIAL_DATASET_PATH} {configs.REPOS_PATH} {configs.MAC_TOOL_OUTPUT}'
    execute_command(mac_tool_command)

main()