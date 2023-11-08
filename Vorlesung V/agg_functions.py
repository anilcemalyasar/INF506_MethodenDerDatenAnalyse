import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_ = pd.read_csv("../datasets/diabetes.csv")
df = df_.copy()

def target_summary_with_num(dataframe, target, num_column):

    print(pd.DataFrame({num_column + ' MEAN': dataframe.groupby(target)[num_column].mean(),
                        num_column + ' MEDIAN': dataframe.groupby(target)[num_column].median(),
                        }))

def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({'TARGET MEAN': dataframe.groupby(target)[cat_col].mean()}))

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and 'Outcome' not in col]

for col in num_cols:
    target_summary_with_num(df, 'Outcome', col)



# cat_cols = [col for col in df.columns if df[col].dtypes == 'O' and 'Outcome' not in col]
#
# for col in cat_cols:
#     target_summary_with_cat(df, 'Outcome', col)
