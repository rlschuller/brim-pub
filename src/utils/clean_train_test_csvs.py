import pandas as pd

# #TRAIN
# csv_path = "data/processed/pre_cuts/16_16_16_8/train.csv"
# df = pd.read_csv(csv_path)
# df.to_csv("data/processed/pre_cuts/16_16_16_8/train2.csv", mode='w')

# df = df.drop(columns = df.columns[0])
# df = df[df['Class Label'] != 'Class Label']
# df['Class Label'] = df['Class Label'].astype('float') 
# df['File Name'] = df['File Name'].astype('str')

# df.to_csv("data/processed/pre_cuts/16_16_16_8/train.csv", mode='w')

#TEST
csv_path = "data/processed/pre_cuts/16_16_16_8/test.csv"
df = pd.read_csv(csv_path)
df.to_csv("data/processed/pre_cuts/16_16_16_8/test2.csv", mode='w')

df = df.drop(columns = df.columns[0])
df = df[df['Class Label'] != 'Class Label']
df['Class Label'] = df['Class Label'].astype('float') 
df['File Name'] = df['File Name'].astype('str')

df.to_csv("data/processed/pre_cuts/16_16_16_8/test.csv", mode='w')