import pandas as pd
import numpy as np

def get_df_count(dataframe, precedent, consequent):
    df_count = pd.crosstab(dataframe[precedent], dataframe[consequent])
    df_count.loc[:,'Total'] = df_count.sum(axis=1)
    return df_count

def get_df_confidence(df_count, columns):
    data = []
    indexes = []
    for index, row in df_count.iterrows():
        new_row_values = []
        indexes.append(index)
        for column in columns:
            confidence_value = row[column]/row["Total"]
            new_row_values.append(confidence_value)
        data.append(new_row_values)
    df_confidence = pd.DataFrame(data, columns=columns, index=indexes)
    return df_confidence

def get_support(dataframe, consequent):
    consequent_values = list(dataframe[consequent].unique())
    support = {}
    for consequent_value in consequent_values:
        consequent_support = len(dataframe[dataframe[consequent]==consequent_value])/len(dataframe)
        support[consequent_value] = consequent_support
    return support

def get_lift(df_count, columns, df_confidence, support):
    data = []
    indexes = []
    for index, row in df_count.iterrows():
        new_row_values = []
        indexes.append(index)
        for column in columns:
            confidence_value = df_confidence.loc[index][column]
            support_value = support[column]
            if confidence_value > 0:
                lift_value = confidence_value/support_value
            else:
                lift_value = 1
            new_row_values.append(lift_value)
        data.append(new_row_values)

    lift = pd.DataFrame(data, columns=columns, index=indexes)
    return lift

def lift_analysis(dataframe, precedent, consequent):
    df_count = get_df_count(dataframe, precedent, consequent)
    columns = list(df_count.columns)
    columns.remove("Total")
    df_confidence = get_df_confidence(df_count, columns)
    support = get_support(dataframe, consequent)
    lift = get_lift(df_count, columns, df_confidence, support)
    
    return lift, df_confidence, support, df_count

def get_rules_of_interest(min_threshold, lift, attribute, df_confidence, df_count):
    rules_increase = {}
    rules_decrease = {}
    threshold = 0.5
    for index, row in lift.iterrows():
        for column in lift.columns:
            if not pd.isnull(row[column]):
                rule = f'{attribute}={index} => {column}'
                confidence = df_confidence.loc[index][column]
                count = df_count.loc[index][column]
                if row[column] >= 1 + threshold:
                    rules_increase[row[column]] = {'rule': rule, 'confidence': confidence, 'occurrences': count}
                if row[column] <= 1 - threshold:
                    rules_decrease[row[column]] = {'rule': rule, 'confidence': confidence, 'occurrences': count}
    
    data = []
    # print(f"Mined rules with at least {threshold*100:.2f}% increased chance")
    for key in sorted(rules_increase, reverse=True):
        data.append([rules_increase[key]['rule'], key, rules_increase[key]['confidence'], rules_increase[key]['occurrences']])
    df_increase = pd.DataFrame(data, columns=['Rule', "Lift", "Confidence", "Occurrences"])

    data = []
    # print(f"Mined rules with at least {threshold*100:.2f}% decreased chance")
    for key in sorted(rules_decrease):
        data.append([rules_decrease[key]['rule'], key, rules_decrease[key]['confidence'], rules_decrease[key]['occurrences']])
    df_decrease = pd.DataFrame(data, columns=['Rule', "Lift", "Confidence", "Occurrences"])

    return df_increase, df_decrease

# returns a discretized dataframe for numerical (integer) columns
def get_discretized_df(df, columns):
    labels = ["Zero", "Few (<10)", "Tens", "Hundreds", "Thousands", "Tens of thousands", "Hundreds of thousands", "Millions", " >= Tens of millions"]
    discretized = df[['chunk_id', 'developerdecision']].copy()
    for column in columns:
        discretized[column] = pd.cut(df[column],
        bins=[-1, 0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, float('inf')], labels = labels)
        discretized[column] = pd.Categorical(discretized[column], 
                        categories=labels,
                        ordered=True)
    return discretized, labels

def get_log_discretized_value(value):
    if np.isnan(value):
        return int(-2)
    value = round(value)
    if value == 0:
        return int(-1)
    else:
        return int(np.round(np.log2(value)))

# returns a discretized dataframe for numerical (integer) columns
def get_discretized_df_new(df, columns):
    discretized = df[['chunk_id', 'developerdecision']].copy()
    for column in columns:
        discretized[column] = df[column].map(get_log_discretized_value)
    return discretized