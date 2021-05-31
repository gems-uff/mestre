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
      <td>./data/INITIAL_DATASET.csv</td>
      <td>./data/LABELLED_DATASET.csv</td>
      <td>Relabels the developerdecision from each chunk that used the Concatenation strategy.</td>
    </tr>
    <tr>
      <td>clone_projects.py</td>
      <td>./data/INITIAL_DATASET.csv</td>
      <td>Repos folder populated</td>
      <td>Clones all projects into the ./repos folder.</td>
    </tr>
    <tr>
      <td>collect_attributes.py</td>
      <td>./data/INITIAL_DATASET.csv</td>
      <td>./data/collected_attributes1.csv</td>
      <td>Extracts a csv with collected attributes from the conflicting chunks. Extracted attributes are described in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_attributespy">link</a>.</td>
    </tr>
    <tr>
      <td>collect_attributes_db.py</td>
      <td>./data/INITIAL_DATASET.csv, Conflicts database</td>
      <td>./data/collected_attributes2.csv</td>
      <td>Extracts a csv with collected attributes from the conflicting chunks that can be calculated from the data in the database. Extracted attributes are described in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_attributes_dbpy">link</a>.</td>
    </tr>
    <tr>
      <td>collect_chunk_authors.py</td>
      <td>./data/INITIAL_DATASET.csv</td>
      <td>./data/chunk_authors.csv</td>
      <td>Extracts a csv with information about all authors that contributed to a conflicting chunk. Detailed information can be found in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#collect_chunk_authorspy">link</a>.</td>
    </tr>
    <tr>
      <td>extract_author_self_conflict.py</td>
      <td>./data/chunk_authors.csv</td>
      <td>./data/authors_self_conflicts.csv</td>
      <td>Extracts a csv with the calculated self_conflict_perc metric for each conflicting chunk.</td>
    </tr>
    <tr>
      <td>execute_mac_tool.py</td>
      <td>./data/INITIAL_DATASET.csv</td>
      <td>Two csv files for each project in the INITIAL_DATASET file</td>
      <td>Executes a modified version of the macTool to extract merge attributes. More info in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#execute_mac_toolpy">link</a>.</td>
    </tr>
    <tr>
      <td>assemble_dataset.py</td>
      <td>./data/collected_attributes1.csv, ./data/collected_attributes2.csv, ./data/authors_self_conflicts.csv</td>
      <td>./data/dataset.csv</td>
      <td>Combines the data from all collected data from the previous scripts into a single csv.</td>
    </tr>
    <tr>
      <td>select_projects.py</td>
      <td>./data/LABELLED_DATASET, ./data/number_conflicting_chunks.csv, ./data/dataset.csv</td>
      <td>./data/selected_dataset.csv, ./data/SELECTED_LABELLED_DATASET.csv</td>
      <td>Extracts only the conflicting chunks that satisfy the criteria contained in the script (currently chunks from projects that have at least 1000 conflicting chunks).</td>
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
      <td>Two csv files for each selected project containing all extracted attributes. Two csv files containing all extracted attributes for all projects.</td>
      <td>Splits the dataset into training/validation (80%) and test (20%) parts. Creates the boolean attribute for authors in each selected project. Details can be viewed in this <a href="https://github.com/gems-uff/conflict-resolution-mining/tree/main/scripts#process_projects_datasetpy">link</a></td>
    </tr>
    <tr>
      <td>github_api_data_preprocess.py</td>
      <td>./data/number_conflicting_chunks.csv, ./data/number_chunks__updated_repos.csv, ./data/projects_data_from_github_api.csv</td>
      <td>./data/api_data.csv</td>
      <td>This script joins the data about projects (collected from GitHub API) with the data of the number of chunks per project (extracted from Ghiotto's database) and the data of the new owner/names of the projects, as well the projects not found by the API.</td>
    </tr>
  </tbody>
</table>


### Download dataset:

Execute the script download_dataset_files.py. All data files will be put into the ./data folder.
