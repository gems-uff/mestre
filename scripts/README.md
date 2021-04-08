## classifyConcatenation.jar

This program relabels a conflict scenario which was labelled as *Concatenation* in the database into *ConcatenationV1V2* or *ConcatenationV2V1*. It uses the following information: V1, V2, context, and resolution content. For more information, check the project [folder](ClassifyConcatenationType). 

---

## database.py

Provides access to the conflicts database.

---

## extract_initial_dataset.py

Extracts the [file](../data/INITIAL_DATASET.csv) from the conflicts database. 

---

## concatenation_relabel.py

Takes as input a csv [file](../data/INITIAL_DATASET.csv) containing conflicting merge scenarios. It is processed to relabel all scenarios resolved with the *Concatenation* strategy into *ConcatenationV1V2* or *ConcatenationV2V1*. The result is stored into a new csv [file](../data/LABELLED_DATASET.csv).

This script uses the program [classifyConcatenation.jar](classifyConcatenation.jar) to perform the relabelling and the script database.py to query conflicts information from the database.

---

## clone_projects.py

Clones the projects contained in the [dataset](../data/INITIAL_DATASET.csv).

---

## execute_mac_tool.py

Executes the a [modified version](https://github.com/helenocampos/macTool) of the[macTool](https://github.com/catarinacosta/macTool) to extract some merge attributes from the repositories. Output files are put into [../data/macTool_output](../data/macTool_output).

--- 

## collect_attributes.py

Script for collecting the following attributes: number of lines added and removed by left and right (left_lines_added, left_lines_removed, right_lines_added, right_lines_removed); conclusion delay in days, calculated by the difference between the dates of the parent commits (conclusion_delay); number of occurrences of keywords in the commit messages of commits involved in a merge (keyword_fix, keyword_bug, keyword_feature, keyword_improve, keyword_document, keyword_refactor,	keyword_update,	keyword_add, keyword_remove, keyword_use, keyword_delete, keyword_change).

Collected data are put into a [csv file](../data/collected_attributes1.csv).

---

## collect_attributes_db.py

Script for collecting the following attributes: cyclomatic complexity (left and right versions of the conflicting chunk and for the whole conflicted file); chunk size (absolute and relative to the file size); chunk position in the file.

Collected data are put into a [csv file](../data/collected_attributes2.csv).

---

## configs.py

Stores configs such as repos folder path and dataset path.

