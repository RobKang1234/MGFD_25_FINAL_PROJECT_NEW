import stats_can
cate_vec_map = {
    'GDP': "v65201210",
    'CPI': "v41690973"
}

df = stats_can.sc.vectors_to_df('v41690973', periods = 6)
df.columns = ['ALL CPI']
df.index.names = ['Date']
print(df)