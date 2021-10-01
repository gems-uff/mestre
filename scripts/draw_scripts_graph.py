#!/usr/bin/env python
# coding: utf-8

from graphviz import Source

temp = """
digraph mygraph {
  graph [fontname = "helvetica"];
  node [shape=box, style=filled, fillcolor=white, fontname = "helvetica"];
    "Conflicts database" -> "extract_initial_dataset.py" -> "INITIAL_DATASET.csv"
    {"INITIAL_DATASET.csv" "Conflicts database"} -> "concatenation_relabel.py" -> "LABELLED_DATASET.csv"
    "INITIAL_DATASET.csv" -> "clone_projects.py" -> "Repos folder"
    {"INITIAL_DATASET.csv" "Repos folder"} -> "collect_attributes.py" -> "collected_attributes1.csv"
    {"INITIAL_DATASET.csv" "Conflicts database" "Repos folder"} -> "collect_attributes_db.py" -> "collected_attributes2.csv"
    {"INITIAL_DATASET.csv" "Repos folder"} -> "collect_chunk_authors.py" -> "chunk_authors.csv"
    "chunk_authors.csv" -> "extract_author_self_conflict.py" -> "authors_self_conflicts.csv"
    {"INITIAL_DATASET.csv" "Repos folder"} -> "execute_mac_tool.py" -> {"macTool_output.csv" "Two csv files for each analyzed repo"}
    {"macTool_output.csv" "Repos folder"} -> "collect_merge_type.py" -> "merge_types_data.csv"
    {"collected_attributes1.csv" "collected_attributes2.csv" "authors_self_conflicts.csv" "merge_types_data.csv" "macTool_output.csv"} -> "assemble_dataset.py" -> "dataset.csv"
    {"LABELLED_DATASET.csv" "number_conflicting_chunks.csv" "dataset.csv"} -> "select_projects.py" -> {"selected_dataset.csv" "SELECTED_LABELLED_DATASET.csv" "projects_intersection.csv"}
    "selected_dataset.csv" -> "transform_boolean_attributes.py" -> "selected_dataset_2.csv"
    {"selected_dataset_2.csv" "chunk_authors.csv"} -> "process_projects_dataset.py" -> {"Two csv files (training/test) for each repo" "dataset-training.csv" "dataset-test.csv"}
    {"number_conflicting_chunks.csv" "number_chunks__updated_repos.csv" "projects_data_from_github_api.csv"} -> "github_api_data_preprocess.py" -> "api_data.csv"
    "extract_initial_dataset.py" [fillcolor="#fcfab8"]
    "concatenation_relabel.py" [fillcolor="#fcfab8"]
    "clone_projects.py" [fillcolor="#fcfab8"]
    "collect_attributes.py" [fillcolor="#fcfab8"]
    "collect_attributes_db.py" [fillcolor="#fcfab8"]
    "collect_chunk_authors.py" [fillcolor="#fcfab8"]
    "extract_author_self_conflict.py" [fillcolor="#fcfab8"]
    "execute_mac_tool.py" [fillcolor="#fcfab8"]
    "collect_merge_type.py" [fillcolor="#fcfab8"]
    "assemble_dataset.py" [fillcolor="#fcfab8"]
    "select_projects.py" [fillcolor="#fcfab8"]
    "transform_boolean_attributes.py" [fillcolor="#fcfab8"]
    "process_projects_dataset.py" [fillcolor="#fcfab8"]
    "github_api_data_preprocess.py" [fillcolor="#fcfab8"]
    "INITIAL_DATASET.csv" [fillcolor=gray97]
    "LABELLED_DATASET.csv" [fillcolor=gray97]
    "collected_attributes1.csv" [fillcolor=gray97]
    "collected_attributes2.csv" [fillcolor=gray97]
    "chunk_authors.csv" [fillcolor=gray97]
    "authors_self_conflicts.csv" [fillcolor=gray97]
    "macTool_output.csv" [fillcolor=gray97]
    "merge_types_data.csv" [fillcolor=gray97]
    "dataset.csv" [fillcolor=gray97]
    "number_conflicting_chunks.csv" [fillcolor=gray97]
    "selected_dataset.csv" [fillcolor=gray97]
    "SELECTED_LABELLED_DATASET.csv" [fillcolor=gray97]
    "projects_intersection.csv" [fillcolor=gray97]
    "selected_dataset_2.csv" [fillcolor=gray97]
    "dataset-training.csv" [fillcolor=gray97]
    "dataset-test.csv" [fillcolor=gray97]
    "number_chunks__updated_repos.csv" [fillcolor=gray97]
    "projects_data_from_github_api.csv" [fillcolor=gray97]
    "api_data.csv" [fillcolor=gray97]
    "Conflicts database" [fillcolor=gray97, shape=cylinder]
    "Repos folder" [fillcolor=gray97, shape=folder]
    "Two csv files for each analyzed repo" [fillcolor=gray97, shape=folder]
    "Two csv files (training/test) for each repo" [fillcolor=gray97, shape=folder]
}
"""
s = Source(temp, filename="scripts_graph", format="svg")
s.view()
