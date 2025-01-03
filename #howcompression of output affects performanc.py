#how lenght of output affects performance
#run cs.py first

five_sparse_signals=[]
bernoulli_errors=[]
gaussian_errors=[]
bad_m_errors=[]
gaussian_svd_errors=[]
learnt_errors=[]
learnt_svd_errors=[]
errors_varying_output=[bernoulli_errors,gaussian_errors,gaussian_svd_errors,bad_m_errors,learnt_errors,learnt_svd_errors]
for i in [4,8,12,16,28,32,64,128,256]:
  for j, matrix_type in enumerate(['bernoulli', 'gaussian', 'gaussian_svd', 'bad_m']):
    matrix = matrix_constructor(matrix_type,512,i)
    signal=signal_generator(512,5,0.05)
    sol= solve(matrix,matrix @ signal,10)
    errors_varying_output[j].append(error_value(sol,signal))
  learnt_matrix=construct_learnt(i,512)
  learnt_svd_matrix=construct_learnt_svd(i,512)
  errors_varying_output[4].append(error_value(solve(learnt_matrix, learnt_matrix @ signal,10),signal))
  errors_varying_output[5].append(error_value(solve(learnt_svd_matrix, learnt_svd_matrix @ signal,10),signal))

# List of rows, where each row is a list
rows = errors_varying_output
row_names=['bernoulli','gaussian','gaussian_svd','bad_m','learnt','learnt_svd']
# Create a DataFrame from the list of rows
df = pd.DataFrame(rows, columns=['4','8','12','16','28','32','64','128','256'],index=row_names)

# Display the DataFrame
print(df)

for i in range(len(errors_varying_output)) :        
  if i!=3:                                           #skipping bad matrix as its error is huge 
   plt.plot([4,8,16,32,64,128,256],errors_varying_output[i])