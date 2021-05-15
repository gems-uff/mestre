import pandas as pd
import configs
import os
import time
import zipfile
import shutil

def unzip_macTool_output(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(configs.DATA_PATH)

def delete_macTool_files():
    for filename in os.listdir(configs.MAC_TOOL_OUTPUT):
        file_path = os.path.join(configs.MAC_TOOL_OUTPUT, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_macTool_ouput(project_name, sha):
    project_file = f"{configs.MAC_TOOL_OUTPUT}/{project_name}-general.csv"
    if(os.path.exists(project_file)):
        macTool_output = pd.read_csv(project_file, delimiter=",")
        commit = macTool_output[macTool_output['Hash'] == sha]
        return commit
    else:
        return []

def main():
    labelled_dataset = pd.read_csv(configs.LABELLED_DATASET_PATH)
    unzip_macTool_output(configs.MAC_TOOL_FILES)
    dataset = []

    collected_attributes1 = pd.read_csv(f"{configs.DATA_PATH}/collected_attributes1.csv")
    collected_attributes2 = pd.read_csv(f"{configs.DATA_PATH}/collected_attributes2.csv")
    collected_attributes3 = pd.read_csv(f"{configs.DATA_PATH}/authors_self_conflicts.csv")

    columns = list(labelled_dataset.columns)
    columns1 = ["left_lines_added",	"left_lines_removed",	"right_lines_added",	"right_lines_removed",	"conclusion_delay",	"keyword_fix",	"keyword_bug",	"keyword_feature",	"keyword_improve",	"keyword_document",	"keyword_refactor",	"keyword_update",	"keyword_add",	"keyword_remove",	"keyword_use",	"keyword_delete",	"keyword_change"]
    columns2 = ["leftCC",	"rightCC",	"fileCC",	"chunkAbsSize",	"chunkRelSize",	"chunkPosition", "fileSize", "chunk_left_abs_size",	"chunk_left_rel_size", "chunk_right_abs_size", "chunk_right_rel_size"]
    columns3 = ["Branching time",	"Merge isolation time",	"Devs 1",	"Devs 2",	"Different devs",	"Same devs",	"Devs intersection",	"Commits 1",	"Commits 2",	"Changed files 1",	"Changed files 2", "Changed files intersection"]
    columns4 = ["self_conflict_perc"]

    columns.extend(columns1)
    columns.extend(columns2)
    columns.extend(columns3)
    columns.extend(columns4)
    

    print(f'Starting the assemble process for {len(labelled_dataset)} chunks...')
    current_index = 0
    for index, row in labelled_dataset.iterrows():
        data = []
        current_index +=1
        status = (current_index / len(labelled_dataset)) * 100
        print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']} for project: {row['project']}")
        sha = row['sha']
        chunk_id = row['chunk_id']
        for column in labelled_dataset.columns:
            data.append(row[column])

        commit = collected_attributes1[collected_attributes1['sha'] == sha]
        if len(commit) > 0:
            for column in columns1:
                data.append(commit.iloc[0][column])
        else:
            data.extend([None] * len(columns1))
        chunk = collected_attributes2[collected_attributes2['chunk_id'] == chunk_id]
        if len(chunk) > 0:
            for column in columns2:
                data.append(chunk.iloc[0][column])
        else:
            data.extend([None] * len(columns2))
        
        commit = get_macTool_ouput(row['project_name'], sha)
        if len(commit) == 1:
            for column in columns3:
                data.append(commit.iloc[0][column])
        else:
            data.extend([None] * len(columns3))

        chunk = collected_attributes3[collected_attributes3['chunk_id'] == chunk_id]
        if len(chunk) > 0:
            for column in columns4:
                data.append(chunk.iloc[0][column])
        else:
            data.extend([None] * len(columns4))

        dataset.append(data)

    delete_macTool_files()
    pd.DataFrame(dataset, columns = columns).to_csv(f'{configs.DATA_PATH}/dataset.csv', index=False)

main()