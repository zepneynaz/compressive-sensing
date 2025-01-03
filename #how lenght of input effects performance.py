#how lenght of input effects performance
#run cs.py first


five_sparse_signals=[]
bernoulli_errors=[]
gaussian_errors=[]
bad_m_errors=[]
gaussian_svd_errors=[]
errors_5_sparse=[bernoulli_errors,gaussian_errors,gaussian_svd_errors,bad_m_errors]
for i in [128,512,1024,2056,8224]:
  for j, matrix_type in enumerate(['bernoulli', 'gaussian', 'gaussian_svd', 'bad_m']):
    matrix = matrix_constructor(matrix_type,i,32)
    signal=signal_generator(i,5,0.05)
    sol= solve(matrix,matrix @ signal,10)
    errors_5_sparse[j].append(error_value(sol,signal))
 

import pandas as pd



# List of rows, where each row is a list
rows = errors_5_sparse
row_names=['bernoulli','gaussian','gaussian_svd','bad_m']
# Create a DataFrame from the list of rows
df = pd.DataFrame(rows, columns=['128','512','1024','2056','8224'],index=row_names)

# Display the DataFrame
print(df)

 
