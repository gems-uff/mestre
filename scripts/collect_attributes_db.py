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

def getCyclomaticComplexity(code):
    ifs = code.count("if")
    whiles = code.count("while")
    fors = code.count("for")
    cases = code.count("case")
    logicalOperators = code.count("&&") + code.count("||")
    CC = ifs + whiles + fors + logicalOperators + cases + 1
    #print("ifs: {}  whiles: {}  fors: {}  cases: {}  logical: {}".format(ifs,whiles,fors,cases,logicalOperators))
    return CC

def getFileSize(code):
    return len(code.split("\n"))

def getChunkContent(chunkId):
    rows = database.get_conflict(chunkId)
    if(rows is not None):
        content = ""
        for row in rows:
            content += row + "\n"
        return content
    return rows


def getChunkAbsoluteSize(beginLine, endLine):
    return (endLine - beginLine) + 1
    
def getChunkRelativeSize(chunkSize, fileSize):
    return chunkSize/fileSize 

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
    
def getLeftChunkCode(chunkCode):
    lines = chunkCode.split("\n")
    leftChunk = ""
    start = False
    for line in lines:
        if("<<<<<<<" in line):
            start = True
            continue
        if("=======" in line):
            break
        if(start == True):
            leftChunk+=line+"\n"
    return leftChunk
    
def getRightChunkCode(chunkCode):
    lines = chunkCode.split("\n")
    rightChunk = ""
    start = False
    for line in lines:
        if("=======" in line):
            start = True
            continue
        if(">>>>>>>" in line):
            break
        if(start == True):
            rightChunk+=line+"\n"
    return rightChunk    

def execute_command(command):
    try:
        my_env = os.environ.copy()
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, env=my_env)
        return result
    except subprocess.CalledProcessError as e:
        print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
    return ''

def get_file_content(commitSHA, file_path):
    command = f"git show {commitSHA}:{file_path}"
    print(command)
    return execute_command(command)


df = pd.read_csv(configs.INITIAL_DATASET_PATH)

repos = {}
data = []
start_time = time.time()
counter = 0
print_every = 20

starting_folder = pathlib.Path(__file__).parent.absolute()
print("Processing start at %s" % (datetime.datetime.now()))
grouped_df = df.groupby('project')
for group_name, df_group in grouped_df:
    for index, row in df_group.iterrows():
        project_folder = f"{configs.REPOS_PATH}/{row['project']}"
        if os.path.exists(project_folder):
            os.chdir(project_folder)
            row2 = []
            beginLine, endLine = database.get_conflict_position(row['chunk_id'])
            sha = row['sha']
            repoName = row['project']
            file_path = row['path'].replace(row['project']+"/", '')
            fileContent = get_file_content(sha, file_path)
            chunkContent = getChunkContent(row['chunk_id'])
            fileSize = getFileSize(fileContent)
            chunkSize = getFileSize(chunkContent)
            leftChunk = getLeftChunkCode(chunkContent)
            rightChunk = getRightChunkCode(chunkContent)
            leftCC = getCyclomaticComplexity(leftChunk)
            rightCC = getCyclomaticComplexity(rightChunk)
            fileCC = getCyclomaticComplexity(fileContent)
            chunkAbsSize = getChunkAbsoluteSize(beginLine, endLine)
            chunkRelSize = getChunkRelativeSize(chunkSize, fileSize)
            chunkPosition = getChunkStartPosition(beginLine, fileSize)
            percentage = index/df.size
            row2.append(row['chunk_id'])
            row2.append(leftCC)
            row2.append(rightCC)
            row2.append(fileCC)
            row2.append(chunkAbsSize)
            row2.append(chunkRelSize)
            row2.append(chunkPosition)
            #print('{} --- {:.2f}% done... Requests remaining: {}'.format(datetime.datetime.now(),percentage, requestsRemaining), end="\r")
            print("LeftCC: %d  RightCC: %d  FileCC: %d Absolute size: %d  Relative size: %.2f      #Position: %d  "% (leftCC, rightCC, fileCC, chunkAbsSize, chunkRelSize, chunkPosition))
            if(counter >= print_every):
                size = len(df)
                percentage = (index/size)*100
                intermediary_time = time.time() - start_time
                estimated = ((intermediary_time * 100)/percentage)/60/60
                print('{} --- {:.2f}% done... estimated time to finish: {:.2f} hours. {} of {} rows processed.'.format(datetime.datetime.now(),percentage, estimated, index, size))
                counter = 0
            counter = counter+1
            data.append(row2)
        os.chdir(starting_folder)
            

elapsed_time = time.time() - start_time

print()
print("Processed in %d seconds. Exporting csv..." % (elapsed_time))
df2 = pd.DataFrame(data, columns = ['chunk_id','leftCC', 'rightCC', 'fileCC', 'chunkAbsSize', 'chunkRelSize', 'chunkPosition'])
result_df = pd.merge(df,df2, on='chunk_id')
result_file = f"{configs.DATA_PATH}/collected_attributes2.csv"
result_df.to_csv(result_file, index=False)