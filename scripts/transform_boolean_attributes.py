import pandas as pd
import configs

df = pd.read_csv(f"{configs.DATA_PATH}/selected_dataset.csv")

print('Transforming language constructors into boolean attributes...')
language_constructors = set()
for index, row in df.iterrows():
    row_constructors = row['kind_conflict'].split(',')
    for row_constructor in row_constructors:
        language_constructors.add(row_constructor.strip())

for language_constructor in language_constructors:
    df[language_constructor] = 0

for index, row in df.iterrows():
    row_constructors = row['kind_conflict'].split(',')
    for row_constructor in row_constructors:
        df.loc[index, row_constructor.strip()] = 1
print(f'Finished. Generating file: {configs.DATA_PATH}/selected_dataset_2.csv ')
df.to_csv(f"{configs.DATA_PATH}/selected_dataset_2.csv", index=False)