#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import time
import configs
import database

def getChunkContent(chunkId):
    rows = database.get_conflict(chunkId)
    if(rows is not None):
        content = ""
        for row in rows:
            content += row + "\n"
        return content
    return rows

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

def get_chunk_relative_size(chunk_side_code, other_side_chunk_code):
    if len(chunk_side_code)+len(other_side_chunk_code) != 0:
        return len(chunk_side_code)/(len(chunk_side_code)+len(other_side_chunk_code))
    return 0

df = pd.read_csv(configs.INITIAL_DATASET_PATH)

data = []

print("Processing start at %s" % (time.ctime()))
columns = ["chunk_id", "chunk_left_abs_size", "chunk_left_rel_size", "chunk_right_abs_size", "chunk_right_rel_size"]
current_index = 0
for index, row in df.iterrows():
    current_index +=1
    status = (current_index / len(df)) * 100
    print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}")
    chunkContent = getChunkContent(row['chunk_id'])
    leftChunk = getLeftChunkCode(chunkContent)
    rightChunk = getRightChunkCode(chunkContent)
    data.append([row['chunk_id'], len(leftChunk), get_chunk_relative_size(leftChunk, rightChunk), len(rightChunk), get_chunk_relative_size(rightChunk, leftChunk)])

result_file = f"{configs.DATA_PATH}/chunk_sizes.csv"
pd.DataFrame(data, columns=columns).to_csv(result_file, index=False)