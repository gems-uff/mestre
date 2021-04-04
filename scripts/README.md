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


