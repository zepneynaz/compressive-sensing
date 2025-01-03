import numpy as np                     #import necessary libraries
import matplotlib.pyplot as plt
import cvxpy as cvx
import pandas as pd

def solve(theta,y,j):            # the function to solve for sparse vector s given sensing matrix theta, sensed vector y, and constant j
 lenght=theta.shape[1]
 sol=cvx.Variable(lenght)        
 objective = cvx.Minimize((j*cvx.norm(y-(theta @ sol),2)**2+(cvx.norm(sol,1))))
 prob = cvx.Problem(objective)
 result = prob.solve(verbose=True)
 return sol.value

def error_value(x,sol):                 #function to return percentage error
 return np.linalg.norm(sol-x) /(np.linalg.norm(x))  
   
def error_vector(x,sol):               
  return abs(sol-x)

def plot(x,sol,error_vector,error):
  plt.plot(x,color='red',label='original signal')  
  plt.plot(sol,label='reconstructed signal')
  plt.plot(error_vector, label ='error')
  plt.xlabel('frequency')
  plt.ylabel('amplitude')
  plt.legend(loc='upper right')
  plt.text(0.01,5.8,f"Reconstruction error:  {error:.2f}%")
  plt.ylim(0,7)
  plt.show()

def analyse(theta,j,x):
 y= theta @ x
 sol= solve(theta,y,j)
 error= error_value(x,sol)
 er_vector= error_vector(x,sol)
 plot(x,sol,er_vector,error)
 print(f"Reconstruction error: {error:.6f}")

x = np.concatenate((np.arange(1,6), np.zeros(128-5)))            
x = x[np.random.permutation(128)]                       # original signal without noise
def add_noise(x, sd):
    noise = np.random.normal(0, sd, size=x.shape)       # Generate noise
    return x + noise                                    # Add noise

x_noisy= add_noise(x,0.05) # noisy semi-sparse vector

#different sensing matrices: bernoulli, gaussian, gaussian svd, learnt
bernoulli = np.random.choice([1, 0], size=(32, len(x)))   #noisy error =0.12%   ,noiseless error= 0.02%
gaussian = np.random.normal(0, 0.5, size=(32, len(x)))    #noisy error =0.21%   ,noiseless error= 0.05%
# Perform SVD
U, Sigma, Vt = np.linalg.svd(gaussian)

# Modify the singular values (for example, double the first singular value)
for i in range(len(Sigma)):
 Sigma[0] = 1 +np.random.normal(0,0.001)  # You can modify any singular value like this

# Reconstruct the matrix using the modified singular values
# make sigma a mxn identity
Sigma_modified = np.hstack((np.diag(Sigma), np.zeros((32, 128-32))))

#Convert the vector of singular values to a diagonal matrix
gaussian_svd = np.dot(U, np.dot(Sigma_modified, Vt))         #noisy error =0.21% ,noiseless error= 0.05%

learning_data=[]
for _ in range (128):
 learning_data.append(add_noise(x,0.05))
learnt_ys=[]
for i in range(128):
  learnt_ys.append(bernoulli @ learning_data[i])
  
learnt = np.column_stack(learnt_ys)                       #noisy error = 0.085% ,noiseless error= 0.009%
identity= np.eye(32)
bad_m=np.hstack((identity,np.zeros((32,128-32))))         #noisy error = 2.7%   ,noiseless error= 2.7%




def matrix_constructor(matrix_type,input,output):
  if matrix_type=='bernoulli':
    return np.random.choice([1, 0], size=(output, input))

  elif matrix_type=='gaussian':
    return np.random.normal(0, 0.5, size=(output, input))

  elif matrix_type=='bad_m':
   return np.hstack((np.eye(output),np.zeros((output,input-output))))

  elif matrix_type=='gaussian_svd':
   return svd_modification(np.random.normal(0, 0.5, size=(output, input)))

  else :
   return 'thats not a valid matrix'
  

def signal_generator(size,sparsity,noise):
  x = np.concatenate([np.random.choice(range(2, 5), sparsity), np.zeros(size - sparsity)])

  x = x[np.random.permutation(size)]

  x_noisy=add_noise(x,noise)
  
  return x_noisy




def construct_learnt(output,input):
    learning_data=[]
    for _ in range (input):
     learning_data.append(signal_generator(input,5,0.05))
     learnt_ys=[]
    for i in range(input):
     learnt_ys.append(np.random.choice([1, 0], size=(output, input)) @ learning_data[i])

    return np.column_stack(learnt_ys)

def construct_learnt_svd(output,input):
  
  return svd_modification(construct_learnt(output,input))
