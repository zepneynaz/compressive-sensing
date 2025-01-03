import numpy as np             #import necessary libraries
import scipy.fft
import matplotlib.pyplot as plt
import cvxpy as cp

# Parameters
fs = 1000  # Sampling frequency (Hz)
T = 1.0    # Duration (seconds)
t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector

# Two-tone signal
f1, f2 = 50, 100  # Frequencies of the tones (Hz)
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)            #original signal we are interested in

# Compute FFT of the signal
s = scipy.fft.fft(signal)                     # fourier trasnform of the signal.IDeally two delta peaks, hence a 2 sparse vector

# Compute frequency bins
freqs = scipy.fft.fftfreq(len(signal), d=1/fs)

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Two-Tone Signal in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Plot the FFT (magnitude spectrum)
plt.subplot(2, 1, 2)
plt.stem(freqs[:len(freqs)//2], np.abs(s[:len(s)//2]), basefmt=" ")
plt.title("FFT of the Signal (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.tight_layout()
plt.show()



sensing_matrices=[]
measurements=[]
reconstructed_signals=[]
Ms= ['10','15','20','50']               #different lenghts of measured vector. the nyquist limit is                                     #at 200, so all of these are sub-nyquist

for M in [10,15,20,50]:
 sensing_matrix = np.random.randn(M, len(signal))   #gaussin sensing matrix
 sensing_matrices.append(sensing_matrix)
# Compressed measurements
 y = np.dot(sensing_matrix, signal)                #y = Phi @ x = Phi @ Psi @ s
 measurements.append(y)
F = np.fft.fft(np.eye(len(signal)))                # Fourier transform matrix
F_inv = np.linalg.inv(F)


for i in range(3): 
 s = cp.Variable(len(signal), complex=True)      #sparse matrix we are interested in
 x_reconstructed = cp.real(F_inv @ s)            # x will be the inverse fourier transform of s
 objective = cp.Minimize(
 cp.norm1(s) + cp.norm2(sensing_matrices[i] @ x_reconstructed - measurements[i])**2)
  #equal weight has been given the data fitting and sparsity in this case
 
   
# Define and solve the optimization problem
 problem = cp.Problem(objective)
 problem.solve()

# Recovered sparse vector
 s_recovered = s.value

# Reconstruct the signal
 reconstructed_signal = np.real(np.fft.ifft(s_recovered))
 reconstructed_signals.append(reconstructed_signal)
 plt.plot(t,reconstructed_signal,color='red')
 plt.plot(t,signal,color='blue')
 plt.xlim(0.2,0.3)
 plt.title(f'Compression To {Ms[i]}'),
 plt.tight_layout()
 plt.xlabel('Time (s)')
 plt.ylabel('Amplitude')
 plt.show()



 mse = np.mean((signal - reconstructed_signal) ** 2)
 snr = 10 * np.log10(np.sum(signal ** 2) / np.sum((signal - reconstructed_signal) ** 2))
 print(f"M={Ms[i]}: MSE={mse:.5f}, SNR={snr:.2f} dB")
