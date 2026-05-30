import os
import pandas as pd
import datetime
p='data/gold/poi_features.parquet'
print('path', p)
print('exists', os.path.exists(p))
if os.path.exists(p):
    df=pd.read_parquet(p)
    print('shape', df.shape)
    print('cols', df.columns.tolist())
    print('mtime', datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat())
else:
    print('file missing')
