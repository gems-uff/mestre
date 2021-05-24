import pandas as pd
import time

def get_self_conflict_perc(authors_left, authors_right):
    authors_left = eval(authors_left)
    authors_right = eval(authors_right)
    left = set()
    right = set()
    for author in authors_left:
        left.add(author)
    for author in authors_right:
        right.add(author)
    return len(left.intersection(right)) / len(left.union(right))

df = pd.read_csv("../data/chunk_authors.csv")
selected = df[(df['authors_left'] != '{}') & (df['authors_right'] != '{}')]
data = []
current_index = 0
print(f'Starting the process for {len(selected)} chunks...')
for index,row in selected.iterrows():
    current_index +=1
    status = (current_index / len(selected)) * 100
    print(f"{time.ctime()} ### {status:.1f}% of chunks processed. Processing chunk {row['chunk_id']}.")
    perc = get_self_conflict_perc(row['authors_left'], row['authors_right'])
    chunk_id = row['chunk_id']
    data.append([chunk_id, perc])

columns = ['chunk_id', "self_conflict_perc"]
pd.DataFrame(data, columns=columns).to_csv("../data/authors_self_conflicts.csv", index=False)
    
    