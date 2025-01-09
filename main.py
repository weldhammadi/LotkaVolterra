import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

time = [0]
lapin = [1]
renard = [2]
step = 0.0003
iteration = [1]

# Charger les données réelles
real_data = pd.read_csv("D:\\school\\HETIC\\PYTHON\\dev back\\Lotka-Volterra\\LotkaVolterra\\populations_lapins_renards.csv")
real_data.columns = real_data.columns.str.strip()
real_data['iteration'] = np.arange(1, len(real_data) + 1)  # Ajouter la colonne 'iteration'

# Afficher les données pour vérifier
#print(real_data.head())
#print(real_data.columns)

# Paramètres du modèle Lotka-Volterra
#alpha = 2 / 3
#beta = 4 / 3
#gamma = 1
#delta = 1
# Paramètres du modèle Lotka-Volterra
alpha_values = np.linspace(1/3, 4/3, 4)
beta_values =  np.linspace(1/3, 4/3, 4)
gamma_values = np.linspace(1/3, 4/3, 4)
delta_values = np.linspace(1/3, 4/3, 4)

best_mse = float('inf')
#best_params = (alpha, beta, gamma, delta)

count = 1
#print number of tests
print(f'Number of tests: {len(alpha_values)*len(beta_values)*len(gamma_values)*len(delta_values)}')

for alpha, beta, gamma, delta in product(alpha_values, beta_values, gamma_values, delta_values):
    lapin = [1]
    renard = [2]
    mse = 0
    for _ in range(100_000):
        lapin.append(lapin[-1] * (alpha - beta * renard[-1]) * step + lapin[-1])
        renard.append(renard[-1] * (delta * lapin[-1] - gamma) * step + renard[-1])
        if _ < len(real_data):
            mse_lapin = (lapin[-1] - (real_data["lapin"].iloc[_])/1000) ** 2
            mse_renard = (renard[-1] - (real_data["renard"].iloc[_])/1000) ** 2
            mse += mse_lapin + mse_renard
    mse = mse / (2 * len(real_data))
    print(f'\ntest n°: {count}\n\t loop Best parameters:   alpha={alpha}, beta={beta},\n\t\t\t\t gamma={gamma}, delta={delta}\n')
    print(f'\t loop Mean Squared Error (MSE): {mse}')
    count += 1
    if mse < best_mse:
        best_mse = mse
        best_params = (alpha, beta, gamma, delta)
    

alpha, beta, gamma, delta = best_params
print(f'\n\nBest parameters: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}')
print(f'Best Mean Squared Error (MSE): {best_mse}\n\n')
            
# Initialisation du MSE
mse = 0

# Initialisation des listes pour les résultats du modèle
model_lapin = []
model_renard = []

# Simulation du modèle pendant 100 000 itérations
for _ in range(100_000):
    time.append(time[-1] + 0.01)
    iteration.append(iteration[-1] + 1)
    lapin.append(lapin[-1] * (alpha - beta * renard[-1]) * step + lapin[-1])
    renard.append(renard[-1] * (delta * lapin[-1] - gamma) * step + renard[-1])
    
    # Ajouter les populations prédites à chaque itération
    model_lapin.append(lapin[-1])
    model_renard.append(renard[-1])
    
    # Calculer le MSE à chaque itération pour les premiers N éléments
    if _ < len(real_data):
        mse_lapin = (model_lapin[-1] - (real_data["lapin"].iloc[_])/1000) ** 2
        mse_renard = (model_renard[-1] - (real_data["renard"].iloc[_])/1000) ** 2
        mse += mse_lapin + mse_renard

# Calculer le MSE moyen
mse = mse / (2 * len(real_data))  # Diviser par 2 pour obtenir la moyenne (lapins et renards)
print(f'simulation using result parameters \nMean Squared Error (MSE) : {mse}')

# Convertir les listes en tableaux numpy pour la visualisation
lapin = np.array(lapin)
renard = np.array(renard)
model_lapin = np.array(model_lapin)
model_renard = np.array(model_renard)

# Visualiser les résultats
lapin *= 1000
renard *= 1000
#plt.figure(figsize=(15, 6))
#plt.plot(real_data["date"], real_data["lapin"], "r--", label='Lapin (réel)')
#plt.plot(real_data["date"], real_data["renard"], "b--", label='Renard (réel)')
#
#plt.plot(time, lapin, "r", label='Lapin (modèle)')
#plt.plot(time, renard, "b", label='Renard (modèle)')
#
#plt.legend()
#plt.show()
#