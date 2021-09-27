#!/usr/bin/env python
# coding: utf-8

from graphviz import Source

temp = """
digraph mygraph {
  node [shape=box, style=filled, fillcolor=white];
    "Conflicts database" -> "extract_initial_dataset.py" -> "./data/INITIAL_DATASET.csv"
    {"./data/INITIAL_DATASET.csv" "Conflicts database"} -> "concatenation_relabel.py" -> "./data/LABELLED_DATASET.csv"
    "./data/INITIAL_DATASET.csv" -> "clone_projects.py" -> "Repos folder"
    {"./data/INITIAL_DATASET.csv" "Repos folder"} -> "collect_attributes.py" -> "./data/collected_attributes1.csv"
    {"./data/INITIAL_DATASET.csv" "Conflicts database" "Repos folder"} -> "collect_attributes_db.py" -> "./data/collected_attributes2.csv"
    {"./data/INITIAL_DATASET.csv" "Repos folder"} -> "collect_chunk_authors.py" -> "./data/chunk_authors.csv"
    "./data/chunk_authors.csv" -> "extract_author_self_conflict.py" -> "./data/authors_self_conflicts.csv"
    {"./data/INITIAL_DATASET.csv" "Repos folder"} -> "execute_mac_tool.py" -> {"./data/macTool_output.csv" "Two csv files for each analyzed repo"}
    {"./data/macTool_outuput.csv" "Repos folder"} -> "collect_merge_type.py" -> "./data/merge_types_data.csv"
    {"./data/collected_attributes1.csv" "./data/collected_attributes2.csv" "./data/authors_self_conflicts.csv" "./data/merge_types_data.csv" "./data/macTool_output.csv"} -> "assemble_dataset.py" -> "./data/dataset.csv"
    {"./data/LABELLED_DATASET.csv" "./data/number_conflicting_chunks.csv" "./data/dataset.csv"} -> "select_projects.py" -> {"./data/selected_dataset.csv" "./data/SELECTED_LABELLED_DATASET.csv" "./data/projects_intersection.csv"}
    "./data/selected_dataset.csv" -> "transform_boolean_attributes.py" -> "./data/selected_dataset_2.csv"
    {"./data/selected_dataset_2.csv" "./data/chunk_authors.csv"} -> "process_projects_dataset.py" -> {"Two csv files (training/test) for each repo" "./data/dataset-training.csv" "./data/dataset-test.csv"}
    {"./data/number_conflicting_chunks.csv" "./data/number_chunks__updated_repos.csv" "./data/projects_data_from_github_api.csv"} -> "github_api_data_preprocess.py" -> "./data/api_data.csv"
    "extract_initial_dataset.py" [fillcolor=gray45]
    "concatenation_relabel.py" [fillcolor=gray45]
    "clone_projects.py" [fillcolor=gray45]
    "collect_attributes.py" [fillcolor=gray45]
    "collect_attributes_db.py" [fillcolor=gray45]
    "collect_chunk_authors.py" [fillcolor=gray45]
    "extract_author_self_conflict.py" [fillcolor=gray45]
    "execute_mac_tool.py" [fillcolor=gray45]
    "collect_merge_type.py" [fillcolor=gray45]
    "assemble_dataset.py" [fillcolor=gray45]
    "select_projects.py" [fillcolor=gray45]
    "transform_boolean_attributes.py" [fillcolor=gray45]
    "process_projects_dataset.py" [fillcolor=gray45]
    "github_api_data_preprocess.py" [fillcolor=gray45]
    "./data/INITIAL_DATASET.csv" [fillcolor=gray80]
    "./data/LABELLED_DATASET.csv" [fillcolor=gray80]
    "./data/collected_attributes1.csv" [fillcolor=gray80]
    "./data/collected_attributes2.csv" [fillcolor=gray80]
    "./data/chunk_authors.csv" [fillcolor=gray80]
    "./data/authors_self_conflicts.csv" [fillcolor=gray80]
    "./data/macTool_outuput.csv" [fillcolor=gray80]
    "./data/merge_types_data.csv" [fillcolor=gray80]
    "./data/dataset.csv" [fillcolor=gray80]
    "./data/number_conflicting_chunks.csv" [fillcolor=gray80]
    "./data/selected_dataset.csv" [fillcolor=gray80]
    "./data/SELECTED_LABELLED_DATASET.csv" [fillcolor=gray80]
    "./data/projects_intersection.csv" [fillcolor=gray80]
    "./data/selected_dataset_2.csv" [fillcolor=gray80]
    "./data/dataset-training.csv" [fillcolor=gray80]
    "./data/dataset-test.csv" [fillcolor=gray80]
    "./data/number_chunks__updated_repos.csv" [fillcolor=gray80]
    "./data/projects_data_from_github_api.csv" [fillcolor=gray80]
    "./data/api_data.csv" [fillcolor=gray80]
    "Conflicts database" [fillcolor=gray95]
    "Repos folder" [fillcolor=gray95]
    "Two csv files for each analyzed repo" [fillcolor=gray95]
    "Two csv files (training/test) for each repo" [fillcolor=gray95]
}
"""
s = Source(temp, filename="scripts_graph", format="png")
s.view()
