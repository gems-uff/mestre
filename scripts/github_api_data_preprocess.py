#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import HTML, display
import tabulate

csv1 = pd.read_csv("../data/number_conflicting_chunks.csv")
csv1 = csv1.rename(columns={"chunks": "chunks_correct"})

print(f"../data/number_conflicting_chunks.csv has {len(csv1)} projects:")
print(csv1)

csv2 = pd.read_csv("../data/number_chunks__updated_repos.csv")

print(f"../data/number_chunks__updated_repos.csv has {len(csv1)} projects:")
print(csv2)

df_inner = pd.merge(csv1, csv2, on='project', how='inner')

result = df_inner[["id", "project", "project_new_ownername", "chunks_correct"]]

result = result.rename(columns={"chunks_correct": "chunks"})

result = result.assign(repo_not_found = lambda d: d["project_new_ownername"] == "REPO_NOT_FOUND")

result.project_new_ownername = result.project_new_ownername.map(lambda d: np.nan if d == "REPO_NOT_FOUND" else d)

result.project_new_ownername.str.contains('REPO_NOT_FOUND', regex=False).sum()

api_data = pd.read_csv("../data/projects_data_from_github_api.csv")

api_data2 = api_data.assign(project_actual_ownername = lambda d: d.resourcePath.str[1:])

result2 = result.assign(project_actual_ownername = result["project_new_ownername"].combine_first(result["project"]))

api_and_newnames = pd.merge(result2, api_data2, on='project_actual_ownername', how='left')

api_and_newnames.to_csv("../data/api_data.csv", index=False)

