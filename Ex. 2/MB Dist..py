import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(0, 1400, 1000)

def mbd(m, k , T, v):
    return 4*np.pi*(m/(2*np.pi*k*T))**(3/2) * v**2 * np.e**((-m*(v**2)) / (2*k*T))


utokg = 1.66054e-27

T = 273.15
k = 1.380649e-23
m_neon = 20.1797 * utokg
m_krypton = 83.798 * utokg
m_xenon = 131.293 * utokg

plt.figure()
plt.ylim(0, 0.008)
plt.xlim(0, 1400)
#plt.plot(v, mbd(m_neon, k, T, v), color='red', label='Neon')
plt.plot(v, mbd(m_krypton, k, T, v), color='green', label='273.15K')
plt.plot(v, mbd(m_krypton, k, T-100, v), color='blue', label='173.15K')
plt.plot(v, mbd(m_krypton, k, T+100, v), color='red', label='373.15K')
#plt.plot(v, mbd(m_xenon, k, T, v), color='green', label='Xenon')
plt.legend(loc='upper right')
plt.xlabel("v[m/s]")
plt.ylabel("p")
plt.title("MB-Verteilung Krypton")
plt.show()
plt.savefig()
