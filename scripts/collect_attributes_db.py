#!/usr/bin/env python
# coding: utf-8
from github import Github
import pandas as pd
import base64
import urllib.request
import time
import datetime
import configs
import database
import traceback

githubTokens = []
with open('github_keys') as file:
    githubTokens = [line.rstrip('\n') for line in file]

g=Github(githubTokens[0])

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



df = pd.read_csv(configs.INITIAL_DATASET_PATH)

repos = {}
data = []
start_time = time.time()
counter = 0
print_every = 20
token_index=0
g=Github(githubTokens[token_index])
token_index+=1
print("Processing start at %s" % (datetime.datetime.now()))
for index, row in df.iterrows():
    requestsRemaining = g.rate_limiting[0]
    if(requestsRemaining>2):
        row2 = []
        url = f"https://api.github.com/repos/{row['project']}"
        beginLine, endLine = database.get_conflict_position(row['chunk_id'])
        sha = row['sha']
        repoName = row['project']
        if(repoName in repos):
            repo = repos.get(repoName)
        else:
            repo = g.get_repo(repoName)
            repos[repoName] = repo
        try:   
            file = repo.get_contents(row['path'].replace(row['project'],''), ref=sha)
            fileContent = base64.b64decode(file.content).decode("utf-8")
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
                print('{} --- {:.2f}% done... estimated time to finish: {:.2f} hours. {} of {} rows processed. {} requests '.format(datetime.datetime.now(),percentage, estimated, index, size, requestsRemaining), end="\r")
                counter = 0
            counter = counter+1
            data.append(row2)
        except (Exception) as error:
            traceback.print_exc()
            print(f'main: {error}')

    else:
        print("Github API requests exhausted. Waiting 60 seconds and trying again....", end="\r")
        time.sleep(60)
        if(token_index < len(githubTokens)-1):
            g=Github(githubTokens[token_index])
            token_index+=1
        else:
            token_index=0
elapsed_time = time.time() - start_time
print()
print("Processed in %d seconds. Exporting csv..." % (elapsed_time))
df2 = pd.DataFrame(data, columns = ['chunk_id','leftCC', 'rightCC', 'fileCC', 'chunkAbsSize', 'chunkRelSize', 'chunkPosition'])
result_df = pd.merge(df,df2, on='chunk_id')
result_file = f"{configs.DATA_PATH}/collected_attributes2.csv"
result_df.to_csv(result_file, index=False)

