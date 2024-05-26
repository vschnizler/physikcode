import numpy as np
import sympy as sym
from sympy import *
import matplotlib.pyplot as plt

t1 = np.linspace(0, 4.5*60, 10)
t2 = np.linspace(5*60, 7*60, (7-5)*6 + 1)
t3 = np.linspace(7.5*60, 11*60, 8)

t = np.concatenate((t1, t2, t3))

Temp_Alu = [20.6, 20.6, 20.7, 20.6, 20.6, 20.7, 20.6, 20.6, 20.6, 20.6, 23.5, 23.4, 23.4, 24.4, 25.4, 26.4, 27.9, 29.7, 29.8, 29.8, 30.2, 30.7, 31.5, 31.9, 32.3, 32.6, 32.7, 32.8]
Temp_Kupf = [20.9, 21.1, 21.1, 21.1, 21.1, 21.1, 21.1, 21.1, 21.1, 21.1 ,23.8, 28.1, 29.7, 32.1, 33.6, 35, 36.6, 37.7, 38.2, 38.9, 39.5, 40, 40.3, 41.2, 41.5, 41.6, 41.7, 41.7, 41.7, 41.4, 41.4]
Temp_Edel = [20.9, 20.8,20.8, 20.9, 20.9, 20.9, 20.9, 21, 21, 21, 24.5, 26.1, 26.8, 26.9, 31.5, 33.4, 34.3, 35.2, 35.8, 36.4, 36.8, 37.1, 37.3, 37.7, 38.4, 38.5, 39, 39.1, 39.3, 39.3, 39.4]

yerr = 0.5 # Kelvin

m_1, b_1 = np.polyfit(t[0:9], Temp_Kupf[0:9], 1)
m_2, b_2 = np.polyfit(t[0:9], Temp_Alu[0:9], 1)
m_3, b_3 = np.polyfit(t[0:9], Temp_Edel[0:9], 1)

print(t[len(Temp_Edel)-4:])
print(Temp_Edel[len(Temp_Edel)-4:])

m_12, b_12 = np.polyfit(t[len(Temp_Kupf)-8:], Temp_Kupf[len(Temp_Kupf)-8:], 1)
m_22, b_22 = np.polyfit(t[len(Temp_Alu)-6:len(Temp_Alu) + 1], Temp_Alu[len(Temp_Alu)-7:], 1)
m_32, b_32 = np.polyfit(t[len(Temp_Edel)-4:], Temp_Edel[len(Temp_Edel)-4:], 1)

#f = plt.figure()
figure, ax = plt.subplots(nrows = 3, ncols = 1, )
figure.tight_layout()

ax[0].errorbar(t, Temp_Kupf, label='Kupfer', yerr=0.5)
ax[0].plot(t, m_1*t + b_1)
ax[0].set_title('Kupfer')
ax[0].plot(t, m_12*t + b_12)
ax[1].errorbar(t, Temp_Edel, label='Edelmetall', yerr=0.5)
ax[1].plot(t, m_3*t + b_3)
ax[1].set_title('Edelmetall')
ax[1].plot(t, m_32*t + b_32)
ax[2].errorbar(t[0:len(Temp_Alu)], Temp_Alu, label='Aluminium', yerr=0.5)
ax[2].plot(t, m_2*t + b_2)
ax[2].set_title('Aluminum')
ax[2].plot(t, m_22*t + b_22)
#ax.legend(loc='lower right')
plt.show()