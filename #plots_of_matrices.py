#plots_of_matrices
#functions and matrices  as defined in appendix

list_of_matrices=[bernoulli,gaussian,gaussian_svd,learnt,bad_m] #32x128

names_of_matrices=['Bernoulli','Gaussian','Gaussian svd','Learnt','Bad']

for i in range(len(list_of_matrices)):
 
 sol= solve(list_of_matrices[i],list_of_matrices[i] @ x_noisy,10)    #input signal is 5 sparse with noise

 plot(x_noisy,sol,error_vector(x_noisy,sol),names_of_matrices[i],error_value(sol,x))
