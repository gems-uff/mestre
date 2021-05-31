## download_dataset_files.py

Downloads the dataset from an external source and puts it into a folder named data in the repository root.

## classifyConcatenation.jar

This program relabels a conflict scenario which was labelled as *Concatenation* in the database into *ConcatenationV1V2* or *ConcatenationV2V1*. It uses the following information: V1, V2, context, and resolution content. For more information, check the project [folder](ClassifyConcatenationType). 

---

## database.py

Provides access to the conflicts database.

---

## extract_initial_dataset.py

Extracts the file (../data/INITIAL_DATASET.csv) from the conflicts database. 

---

## concatenation_relabel.py

Takes as input a csv file (../data/INITIAL_DATASET.csv) containing conflicting merge scenarios. It is processed to relabel all scenarios resolved with the *Concatenation* strategy into *ConcatenationV1V2* or *ConcatenationV2V1*. The result is stored into a new csv file (../data/LABELLED_DATASET.csv).

This script uses the program [classifyConcatenation.jar](classifyConcatenation.jar) to perform the relabelling and the script database.py to query conflicts information from the database.

---

## clone_projects.py

Clones the projects contained in the dataset file (../data/INITIAL_DATASET.csv).

---

## execute_mac_tool.py

Executes the a [modified version](https://github.com/helenocampos/macTool) of the [macTool](https://github.com/catarinacosta/macTool) to extract some merge attributes from the repositories. Output files are put into ../data/macTool_output.

--- 

## collect_attributes.py

Script for collecting the following attributes: number of lines added and removed by left and right (left_lines_added, left_lines_removed, right_lines_added, right_lines_removed); conclusion delay in days, calculated by the difference between the dates of the parent commits (conclusion_delay); number of occurrences of keywords in the commit messages of commits involved in a merge (keyword_fix, keyword_bug, keyword_feature, keyword_improve, keyword_document, keyword_refactor,	keyword_update,	keyword_add, keyword_remove, keyword_use, keyword_delete, keyword_change).

Collected data are put into a csv file (../data/collected_attributes1.csv).

---

## collect_attributes_db.py

Script for collecting the following attributes: cyclomatic complexity (left and right versions of the conflicting chunk and for the whole conflicted file); chunk size (absolute and relative to the file size); conflicted file size; chunk position in the file.

Collected data are put into a csv file (../data/collected_attributes2.csv).

---

## configs.py

Stores configs such as repos folder path and dataset path.

---

## collect_chunk_authors.py

This script extracts how many lines each author contributed to a conflicting chunk. 

Currently it is able to extract modified lines, renamed/moved code and some cases of deleted lines.

To extract modified lines, it replays the merge operation that caused the conflict and then use git blame on the conflicted files. The conflicting chunk is then located in the git blame file and the author of each involved line is extracted. Note that all involved lines in a conflict chunk are considered. So it might happen that a line was not modified between the merge common ancestor and the merge parents, but it is still included in the conflict. In this case, this line is also considered.

Example:

Project: 3scale/3scale_ws_api_for_java

Steps to reproduce the merge:
```
git clone https://github.com/3scale/3scale_ws_api_for_java
git checkout 4650578dee712b2b08f2ead2bf6a531f82b1e0e9
git merge d623f491daa2f14f06f53338265d28a489138a6b
```

This merge results in a conflict in file src/net/threescale/api/v2/Api2Impl.java.

If we use git blame on such conflicted file, the conflicting chunk is display as follows:

```
00000000 (<not.committed.yet>            2021-05-05 10:43:05 -0300  81) <<<<<<< HEAD
4650578d (<tiago@3scale.net>             2011-01-27 15:15:31 +0100  82)         if (response.getResponseCode() == 200 || response.getResponseCode() == 409) {
c4218480 (<geoffd@professionalhosts.com> 2010-09-29 15:49:38 +0200  83)             return new AuthorizeResponse(response.getResponseText());
eec4974e (<geoffd@professionalhosts.com> 2010-09-29 10:40:05 +0200  84)         } else if (response.getResponseCode() == 403 || response.getResponseCode() == 404) {
dfd42e40 (<geoffd@professionalhosts.com> 2010-09-28 08:08:52 +0200  85)             throw new ApiException(response.getResponseText());
07a08aef (<geoffd@professionalhosts.com> 2010-09-28 09:19:58 +0200  86)         } else {
07a08aef (<geoffd@professionalhosts.com> 2010-09-28 09:19:58 +0200  87)             throw createExceptionForUnexpectedResponse(response);
00000000 (<not.committed.yet>            2021-05-05 10:43:05 -0300  88) =======
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  89)             if (response.getResponseCode() == 200) {
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  90)                 AuthorizeResponse authorizedResponse = new AuthorizeResponse(response.getResponseText());
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  91)                 cache.addAuthorizedResponse(app_key, authorizedResponse);
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  92)                 return authorizedResponse;
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  93)             } else if (response.getResponseCode() == 403 || response.getResponseCode() == 404) {
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  94)                 throw new ApiException(response.getResponseText());
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  95)             } else {
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  96)                 throw createExceptionForUnexpectedResponse(response);
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  97)             }
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  98)         }
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200  99)         else {
b6da57fc (<geoffd@professionalhosts.com> 2010-10-26 15:36:31 +0200 100)             return cached_response;
00000000 (<not.committed.yet>            2021-05-05 10:43:05 -0300 101) >>>>>>> d623f491daa2f14f06f53338265d28a489138a6b
```

From the git blame output, we can observe that for the first parent of the merge, geoffd@professionalhosts.com modified 5 lines and tiago@3scale.net modified 1 line. However, if we check the commit history between the merge common ancestor and the first merge parent (git log 1d2ec0a2016edab9736e163e09d7a994af00ccbf..4650578dee712b2b08f2ead2bf6a531f82b1e0e9), we can only see commits from tiago@3scale.net. geoffd@professionalhosts.com contributed indirectly for the conflict in previous commits, but we also consider him as an author that contributed to this conflict.


When we have a conflict chunk where one of its sides is empty, it means that someone between the merge common ancestor and a merge parent has deleted some lines, while someone between the merge common ancestor and the other mege parent has modified/added lines. In such case, git blame does not help much, because deleted lines are not shown. To extract the authors who deleted lines in a conflict, we use a tool named [difflame](https://github.com/eantoranz/difflame). However, in some cases we cannot correctly locate the conflict chunk in its output.

Some attempts were made to allow the extraction of deleted lines for all cases, but with no success. Some of the attempts were kept in the folder "unused_scripts" for reference in the future.

The output of this script is a csv file that is put into data/chunk_authors.csv.

---

## extract_author_self_conflict.py

Calculates the self conflict percentage metric for each chunk.

Output data is exported to a csv file (../data/authors_self_conflicts.csv).

---

## assemble_dataset.py

Merges the different csv data files with collected attributes into a single csv file.

Output data is exported to a csv file (../data/dataset.csv).

---

## select_projects.py

Select chunks from projects based on a given criteria.

Output data is exported to a csv file (../data/selected_dataset.csv)

---

## transform_boolean_attributes.py

Script for transforming the language constructs from each chunk into a boolean attribute (one column per construct).

Output data is exported to a csv file (../data/selected_dataset2.csv).

---

## process_projects_dataset.py

This script takes the csv files (../data/selected_dataset2.csv) and (../data/chunk_authors.csv) as input.

It splits the dataset into training/validation (80% of the chunks) and test (20% of the chunks) parts. It also creates the boolean attribute author columns, creating one column for each author that participated in a conflicting chunk of that project. For each chunk, a value of 1 is assigned for the author column if it has participated in the conflict and a value of 0 is assigned otherwise.

The outputs of this script are two csv files for each project in the dataset, which are put into (../data/projects). One csv file contains the training dataset (../data/projects/projectowner_projectname-training.csv) and the other contains the test dataset (../data/projects/projectowner__projectname-test.csv) for each project. Two general csv files are also created containing the attributes for all chunks from all projects. One is (../data/dataset-training.csv) and the other is (../data/dataset-test.csv).

---

## github_api_data_preprocess.py

This script takes these CSV files as input:

- ../data/number_conflicting_chunks.csv
- ../data/number_chunks__updated_repos.csv
- ../data/projects_data_from_github_api.csv

<!-- TODO some steps on this script to correct the number of chunks may be removed later, to make the script clearer.-->
This script joins the data about projects (collected from GitHub API) with the data of the number of chunks per project (extracted from Ghiotto's database) and the data of the new *owner/names* of the projects, as well the projects not found by the API.
Finally it saves the output to file [../data/api_data.csv](../data/api_data.csv).
