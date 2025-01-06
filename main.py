import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = [0]
lapin = [1]
renard = [2]
step = 0.001
#import real data csv
#date       ,lapin ,renard
#2020-09-10 , 1000 ,2000
real_data = pd.read_csv("D:\school\HETIC\PYTHON\dev back\Lotka-Volterra\populations_lapins_renards.csv")



#show data
print(real_data)
real_data.columns = real_data.columns.str.strip()
print(real_data.columns)


alpha = 2/3
beta = 4/3
gamma = 1
delta = 1

for _ in range(0, 100_000):
    time.append(time[-1] + 0.01)
    lapin.append(lapin[-1]*(alpha - beta*renard[-1])*step + lapin[-1])
    renard.append(renard[-1]*(delta*lapin[-1] - gamma)*step + renard[-1])

lapin = np.array(lapin)
renard = np.array(renard)
lapin *= 1000
renard *= 1000

#plot real data
plt.figure(figsize=(15,6))
plt.plot(real_data["date"], real_data["lapin"],"r*", label='Lapin')
plt.plot(real_data["date"], real_data["renard"],"b*", label='Renard')


plt.plot(time, lapin,"r", label='Lapin')
plt.plot(time, renard,"b", label='Renard')
plt.legend()
plt.show()
