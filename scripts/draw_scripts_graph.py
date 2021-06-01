#!/usr/bin/env python
# coding: utf-8

from graphviz import Source

temp = """
digraph mygraph {
  node [shape=box];
    "Conflicts database" -> "extract_initial_dataset.py" -> "./data/INITIAL_DATASET.csv"
    "./data/INITIAL_DATASET.csv" -> "concatenation_relabel.py" -> "./data/LABELLED_DATASET.csv"
    "./data/INITIAL_DATASET.csv" -> "clone_projects.py" -> "Repos folder populated"
    "./data/INITIAL_DATASET.csv" -> "collect_attributes.py" -> "./data/collected_attributes1.csv"
    {"./data/INITIAL_DATASET.csv" "Conflicts database"} -> "collect_attributes_db.py" -> "./data/collected_attributes2.csv"
    "./data/INITIAL_DATASET.csv" -> "collect_chunk_authors.py" -> "./data/chunk_authors.csv"
    "./data/chunk_authors.csv" -> "extract_author_self_conflict.py" -> "./data/authors_self_conflicts.csv"
    "./data/INITIAL_DATASET.csv" -> "execute_mac_tool.py" -> "Two csv files for each project in the INITIAL_DATASET file"
    {"./data/collected_attributes1.csv" "./data/collected_attributes2.csv" "./data/authors_self_conflicts.csv"} -> "assemble_dataset.py" -> "./data/dataset.csv"
    {"./data/LABELLED_DATASET" "./data/number_conflicting_chunks.csv" "./data/dataset.csv"} -> "select_projects.py" -> {"./data/selected_dataset.csv" "./data/SELECTED_LABELLED_DATASET.csv"}
    "./data/selected_dataset.csv" -> "transform_boolean_attributes.py" -> "./data/selected_dataset2.csv"
    {"./data/selected_dataset2.csv" "./data/chunk_authors.csv"} -> "process_projects_dataset.py" -> {"Two csv files for each selected project containing all extracted attributes." "Two csv files containing all extracted attributes for all projects."}
    {"./data/number_conflicting_chunks.csv" "./data/number_chunks__updated_repos.csv" "./data/projects_data_from_github_api.csv"} -> "github_api_data_preprocess.py" -> "./data/api_data.csv"
}
"""
s = Source(temp, filename="scripts_graph", format="png")
s.view()