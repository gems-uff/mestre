#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import time
import datetime
import configs
import database
import traceback
import pathlib
import os
import subprocess

def getCyclomaticComplexity(lines):
    ifs = whiles = fors = cases = logicalOperators = 0
    for line in lines:
        ifs += line.count("if")
        whiles += line.count("while")
        fors += line.count("for")
        cases += line.count("case")
        logicalOperators += line.count("&&") + line.count("||")
    CC = ifs + whiles + fors + logicalOperators + cases + 1
    #print("ifs: {}  whiles: {}  fors: {}  cases: {}  logical: {}".format(ifs,whiles,fors,cases,logicalOperators))
    return CC

def getFileSize(lines):
    return len(lines)

def getChunkContent(chunkId):
    return database.get_conflict(chunkId)

# chunk size relative to the file size
def getChunkRelativeSize(chunkSize, fileSize):
    if fileSize!= 0:
        return chunkSize/fileSize 
    else:
        return 0

def getChunkStartPosition(beginLine, fileSize): # return the quarter in which the chunk is located in the file
    # |----|----|----|----|
    #  1st  2nd  3rd  4th
    # suppose the file size is 100, beginLine is 70
    # it should return 3
    quarterSize = fileSize//4
    firstQuarter = quarterSize
    secondQuarter = firstQuarter+quarterSize
    thirdQuarter = secondQuarter+quarterSize
    if(beginLine<firstQuarter):
        return 1
    elif(beginLine<secondQuarter):
        return 2
    elif(beginLine<thirdQuarter):
        return 3
    else:
        return 4
    
def reset():
    command = "git reset --hard"
    output = execute_command(command)
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

def execute_command(command):
    try:
        my_env = os.environ.copy()
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env, encoding="latin-1")
        return result
    except subprocess.CalledProcessError as e:
        pass
        # print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return ''

def get_file_content(file_path):
    try:
        with open(file_path, encoding='latin-1') as f:
            return f.readlines()
    except:
        pass
    return []

def getLeftChunkCode(lines):
    leftChunk = []
    start = False
    for line in lines:
        if("<<<<<<<" in line):
            start = True
            continue
        if("=======" in line):
            break
        if(start == True):
            leftChunk.append(line)
    return leftChunk
    
def getRightChunkCode(lines):
    rightChunk = []
    start = False
    for line in lines:
        if("=======" in line):
            start = True
            continue
        if(">>>>>>>" in line):
            break
        if(start == True):
            rightChunk.append(line)
    return rightChunk    

# size of one side of the chunk relative to the other side of the chunk
def get_chunk_relative_size(chunk_side_code, other_side_chunk_code):
    if len(chunk_side_code)+len(other_side_chunk_code) != 0:
        return len(chunk_side_code)/(len(chunk_side_code)+len(other_side_chunk_code))
    return 0

def write_failed_chunks(failed):
    with open(f'{configs.LOGS_PATH}/collect_attributes_db_failed.txt', 'w') as file:
        for failed_chunk in failed:
            file.write(f"{failed_chunk[0]}:{failed_chunk[1]}\n")

df = pd.read_csv(configs.INITIAL_DATASET_PATH)

repos = {}
data = []
start_time = time.time()
counter = 0
chunk_count = 0
print_every = 20
failed_chunks = []

starting_folder = pathlib.Path(__file__).parent.absolute()
print("Processing start at %s" % (datetime.datetime.now()))
grouped_df = df.groupby('project')
for group_name, df_group in grouped_df:
    for index, row in df_group.iterrows():
        chunkCount+=1
        project_folder = f"{configs.REPOS_PATH}/{row['project']}"
        if os.path.exists(project_folder):
            os.chdir(project_folder)
            if merge(row['leftsha'], row['rightsha']):
                row2 = []
                beginLine, endLine = database.get_conflict_position(row['chunk_id'])
                sha = row['sha']
                repoName = row['project']
                file_path = row['path'].replace(row['project']+"/", '')
                fileContent = get_file_content(file_path)
                if fileContent != []:
                    chunkContent = getChunkContent(row['chunk_id'])
                    fileSize = getFileSize(fileContent)
                    leftChunk = getLeftChunkCode(chunkContent)
                    rightChunk = getRightChunkCode(chunkContent)
                    leftCC = getCyclomaticComplexity(leftChunk)
                    rightCC = getCyclomaticComplexity(rightChunk)
                    fileCC = getCyclomaticComplexity(fileContent)
                    chunkAbsSize = len(leftChunk) + len(rightChunk)
                    chunkRelSize = getChunkRelativeSize(chunkAbsSize, fileSize)
                    chunkPosition = getChunkStartPosition(beginLine, fileSize)
                    
                    left_chunk_absolute_size = len(leftChunk)
                    left_chunk_relative_size = get_chunk_relative_size(leftChunk, rightChunk)
                    right_chunk_absolute_size = len(rightChunk)
                    right_chunk_relative_size = get_chunk_relative_size(rightChunk, leftChunk)

                    percentage = chunkCount/df.size
                    row2.append(row['chunk_id'])
                    row2.extend([leftCC, rightCC, fileCC, fileSize, chunkAbsSize, chunkRelSize, chunkPosition])
                    row2.extend([left_chunk_absolute_size, left_chunk_relative_size])
                    row2.extend([right_chunk_absolute_size, right_chunk_relative_size])
                else:
                    failed_chunks.append([row['chunk_id'], 'INVALID_FILE'])
                
                #print('{} --- {:.2f}% done... Requests remaining: {}'.format(datetime.datetime.now(),percentage, requestsRemaining), end="\r")
                print("chunk_id: %d LeftCC: %d  RightCC: %d  FileCC: %d Chunk Absolute size: %d  Relative size: %.2f   fileSize: %.2f   #Position: %d  "% (row['chunk_id'], leftCC, rightCC, fileCC, chunkAbsSize, chunkRelSize, fileSize, chunkPosition), flush=True)
            else:
                failed_chunks.append([row['chunk_id'], 'CANT_MERGE'])
            if(counter >= print_every):
                size = len(df)
                percentage = (chunkCount/size)*100
                intermediary_time = time.time() - start_time
                estimated = ((intermediary_time * 100)/percentage)/60/60
                print('{} --- {:.2f}% done... estimated time to finish: {:.2f} hours. {} of {} rows processed.'.format(datetime.datetime.now(),percentage, estimated, chunkCount, size), flush=True)
                counter = 0
            counter = counter+1
            data.append(row2)
        else:
            failed_chunks.append([row['chunk_id'], 'REPO_NOT_AVAILABLE'])
        os.chdir(starting_folder)

elapsed_time = time.time() - start_time

print()
write_failed_chunks(failed_chunks)
print("Processed in %d seconds. Exporting csv..." % (elapsed_time))
columns = ['chunk_id','leftCC', 'rightCC', 'fileCC', 'fileSize' 'chunkAbsSize', 'chunkRelSize', 'chunkPosition']
columns.extend(["chunk_left_abs_size", "chunk_left_rel_size", "chunk_right_abs_size", "chunk_right_rel_size"])
df2 = pd.DataFrame(data, columns = columns)
result_df = pd.merge(df,df2, on='chunk_id')
result_file = f"{configs.DATA_PATH}/collected_attributes2.csv"
result_df.to_csv(result_file, index=False)