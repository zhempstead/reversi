import sys

import pandas as pd

results = sys.argv[1]
df = pd.read_csv(results)
df = df.drop(columns='gpt_player')
cols = list(df.columns)
df['count'] = 1
print(df.groupby(cols).count())
