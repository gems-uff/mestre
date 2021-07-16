# conflict-resolution-mining

## Dataset

There are two options for accessing the dataset used in this paper. You can either collect the data by yourself (takes a long time) or directly download the dataset files.

### Collect by yourself

We assume you have access to the conflicts database used in this [paper](https://ieeexplore.ieee.org/abstract/document/8468085). The database information can be configured in the scripts/database.py file.

Reproduce the scripts in the following order:

<table>
  <thead>
    <tr>
      <th>Script</th>
      <th>Input</th>
      <th>Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>extract_initial_dataset.py</td>
      <td>Conflicts database</td>
      <td>./data/INITIAL_DATASET.csv</td>
      <td>Extracts a csv with conflicting chunks and some descriptive attributes.</td>
    </tr>
    <tr>
      <td>concatenation_relabel.py</td>
      <td>./data/INITIAL_DATASET.csv, Conflicts database</td>
      <td>./data/LABELLED_DATASET.csv</td>
      <td>Relabels the developerdecision from each chunk that used the Concatenation strategy.</td>
    </tr>
    <tr>
      <td>clone_projects.py</td>
      <td>./data/INITIAL_DATASET.csv</td>
      <td>Repos folder</td>
      <td>Clones all projects into the ./repos folder.</td>
    </tr>
    <tr>
      <td>collect_chunk_authors.py</td>
      <td>./data/INITIAL_DATASET.csv, Repos folder</td>
      <td>./data/chunk_authors.csv</td>
      <td>Extracts a csv with information about all authors that contributed to a conflicting chunk. Detailed information can be found in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_chunk_authorspy">link</a>.</td>
    </tr>
    <tr>
      <td>collect_attributes.py</td>
      <td>./data/INITIAL_DATASET.csv, Repos folder</td>
      <td>./data/collected_attributes1.csv</td>
      <td>Extracts a csv with collected attributes from the conflicting chunks. Extracted attributes are described in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_attributespy">link</a>.</td>
    </tr>
    <tr>
      <td>execute_mac_tool.py</td>
      <td>./data/INITIAL_DATASET.csv, Repos folder</td>
      <td>Two csv files for each analyed repo, ./data/macTool_output.csv</td>
      <td>Executes a modified version of the macTool to extract merge attributes. More info in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#execute_mac_toolpy">link</a>.</td>
    </tr>
    <tr>
      <td>collect_merge_type.py</td>
      <td>./data/macTool_output.csv, Repos folder</td>
      <td>./data/merge_types_data.csv</td>
      <td>Extracts the merge commit message for each chunk merge commit, the merge branch message indicator, and the boolean attribute regarding the existence of multiple developers on each branch of the merge. More info in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_merge_typepy">link</a>.</td>
    </tr>
    <tr>
      <td>collect_attributes_db.py</td>
      <td>./data/INITIAL_DATASET.csv, Conflicts database, Repos folder</td>
      <td>./data/collected_attributes2.csv</td>
      <td>Extracts a csv with collected attributes from the conflicting chunks that can be calculated from the data in the database. Extracted attributes are described in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_attributes_dbpy">link</a>.</td>
    </tr>
    <tr>
      <td>extract_author_self_conflict.py</td>
      <td>./data/chunk_authors.csv</td>
      <td>./data/authors_self_conflicts.csv</td>
      <td>Extracts a csv with the calculated self_conflict_perc metric for each conflicting chunk.</td>
    </tr>
    <tr>
      <td>assemble_dataset.py</td>
      <td>./data/collected_attributes1.csv, ./data/collected_attributes2.csv, ./data/authors_self_conflicts.csv, ./data/merge_types_data.csv, ./data/macTool_output.csv</td>
      <td>./data/dataset.csv</td>
      <td>Combines all collected data from the previous scripts into a single csv.</td>
    </tr>
    <tr>
      <td>select_projects.py</td>
      <td>./data/LABELLED_DATASET, ./data/number_conflicting_chunks.csv, ./data/dataset.csv</td>
      <td>./data/selected_dataset.csv, ./data/SELECTED_LABELLED_DATASET.csv</td>
      <td>Extracts only the conflicting chunks that satisfy the criteria contained in the script (currently chunks from projects that have at least 1,000 conflicting chunks).</td>
    </tr>
    <tr>
      <td>github_api_data_preprocess.py</td>
      <td>./data/number_conflicting_chunks.csv, ./data/number_chunks__updated_repos.csv, ./data/projects_data_from_github_api.csv</td>
      <td>./data/api_data.csv</td>
      <td>This script joins the data about projects (collected from GitHub API) with the data of the number of chunks per project (extracted from Ghiotto's database) and the data of the new owner/names of the projects, as well the projects not found by the API.</td>
    </tr>
    <tr>
      <td>transform_boolean_attributes.py</td>
      <td>./data/selected_dataset.csv</td>
      <td>./data/selected_dataset2.csv</td>
      <td>Transforms the language construct column in each conflicting chunk into a boolean attribute.</td>
    </tr>
    <tr>
      <td>process_projects_dataset.py</td>
      <td>./data/selected_dataset2.csv, ./data/chunk_authors.csv</td>
      <td>Two csv files (training/test) for each analyzed selected repository put into .data/projects, .data/dataset-training.csv, .data/dataset-test.csv</td>
      <td>Splits the dataset into training/validation (80%) and test (20%) parts. Creates the boolean attribute for authors in each selected project. Details can be viewed in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#process_projects_datasetpy">link</a></td>
    </tr>
    <tr>
      <td>discretize_dataset.py</td>
      <td>./data/dataset-training.csv, ./data/dataset-test.csv, ./data/projects/{project}-training.csv, ./data/projects/{project}-test.csv </td>
      <td>Two csv files (training/test) for each analyzed selected repository put into .data/projects/discretized_log2 and .data/projects/discretized_log10, .data/dataset-training_log2.csv, .data/dataset-training_log10.csv, .data/dataset-test_log2.csv, .data/dataset-test_log10.csv</td>
      <td>Discretizes categorical attributes from the dataset using log2 and log10 functions.</td>
    </tr>
  </tbody>
</table>


### Download dataset:

Execute the script download_dataset_files.py. All data files will be put into the ./data folder.
