import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.fft import fft

drone_present = 0
file_counter = 150
pa_ref = 20e-6 # 20 micropascals reference
prop_freq = 80 # Hz
prop_amp = 0.00001 # Pa
noise_dB = 60
#noise_rms = 0.0001 # Pa - 0.0001 corresponds to 14 dB OASPL
noise_rms = 10**(noise_dB/20)*pa_ref
print("dB Noise = ",noise_dB,"\nRMS Noise = ",noise_rms,"Pascals")
sample_rate = 50000 # Hz
length = 1 # seconds
time = np.arange(0,length,1/sample_rate)
for k in range(file_counter,file_counter+10):
  y = np.random.normal(0, noise_rms, length*sample_rate)
  buzz = drone_present * prop_amp * np.sin(2*np.pi*prop_freq*time)
  buzz4 = drone_present * prop_amp * np.sin(2*np.pi*prop_freq*4*time) # assuming quadcopter harmonics
  y_total = y + buzz + buzz4
  rms = np.sqrt(np.mean(y**2))
  print("true RMS = ",rms)
  dB = 20*np.log10(rms/pa_ref)
  print("total dB = ",dB)
  
  y_short = np.zeros(y_total.shape[0],dtype='f4')
  for i in range(y_total.shape[0]):
     y_short[i] = y_total[i].astype(float)
  output_filename = 'simdata\\test_binary' + str(k) + '.dat'
  binary_file = open(output_filename,'wb')
  binary_file.write(drone_present.to_bytes(2, byteorder='big', signed=False))
  binary_file.write(y_short)
  binary_file.close()

  text_filename = 'simdata\\test_binary' + str(k) + '.csv'
  wtr = csv.writer(open(text_filename,'w'),lineterminator='\n')
  for x in y_short:
     wtr.writerow([x])
#wtr.close()

#with open('test_binary.csv','w') as csvfile:
#   writer = csv.writer(csvfile)
#   writer.writerows(y_total)
#csvfile.close()

  fig, (ax1, ax2) = plt.subplots(2,1,height_ratios=[1, 1.5])
  ax1.plot(time,y_total)
  ax1.set_xlabel("Time (sec)")
  ax1.set_ylabel("Pressure (Pa)")
  ax1.set_title('Transient Acoustic Signal')
  [p,freq] = ax2.psd(y_total,Fs=sample_rate,NFFT=4096)
  duh = np.stack((freq, p), axis = 1)
  ax2.set_title('Acoustic PSD')
  plt.xscale('log')
  ax2.grid(True)
  fig.set_figwidth(8)
  fig.set_figheight(10)

  p_short = np.zeros(p.shape[0],dtype='f4')
  f_short = np.zeros(freq.shape[0],dtype='f4')
  for i in range(p_short.shape[0]):
     p_short[i] = p[i].astype(float)
     f_short[i] = freq[i].astype(float)
  duh = np.stack((f_short, p_short), axis = 1)
  output_filename = 'simdata\\psd_binary' + str(k) + '.dat'
  psd_file = open(output_filename,'wb')
  psd_file.write(drone_present.to_bytes(2, byteorder='big', signed=False))
  psd_file.write(duh)
  psd_file.close()

  y_fft = fft(y_total)
  output_filename = 'simdata\\fft_binary' + str(k) + '.dat'
  fft_file = open(output_filename,'wb')
  fft_file.write(drone_present.to_bytes(2, byteorder='big', signed=False))
  fft_file.write(y_fft)
  fft_file.close()
  real_short = np.zeros(y_fft.shape[0],dtype='f4')
  imag_short = np.zeros(y_fft.shape[0],dtype='f4')
  y_array = np.column_stack([y_fft.real, y_fft.imag])
  for i in range(real_short.shape[0]):
     real_short[i] = y_array[i,0].astype(float)
     imag_short[i] = y_array[i,1].astype(float)
  duh_fft = np.stack((real_short, imag_short), axis = 1)
  output_filename = 'simdata\\fft_binary' + str(k) + '.dat'
  fft_file = open(output_filename,'wb')
  fft_file.write(drone_present.to_bytes(2, byteorder='big', signed=False))
  fft_file.write(duh_fft)
  fft_file.close()
  text_filename = 'simdata\\fft_binary' + str(k) + '.csv'
  wtr = csv.writer(open(text_filename,'w'),lineterminator='\n')
  for x in range(duh_fft.shape[0]):
     wtr.writerow(duh_fft[x,:])
''' For debug purposes  
'''
'''
  text_filename = 'simdata\\psd_binary' + str(k) + '.csv'
  wtr = csv.writer(open(text_filename,'w'),lineterminator='\n')
  for x in range(duh.shape[0]):
     wtr.writerow(duh[x,:])
'''
#plt.show()
