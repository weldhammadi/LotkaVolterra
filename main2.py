import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

def find_best_parameters(step, real_data, alpha_values, beta_values, gamma_values, delta_values):
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
        print(f'\ntest n°: {count}')
        #print(f'\ntest n°: {count}\n\t loop Best parameters:   alpha={alpha}, beta={beta},\n\t\t\t\t gamma={gamma}, delta={delta}\n')
        #print(f'\t loop Mean Squared Error (MSE): {mse}')
        count += 1
        if mse < best_mse:
            best_mse = mse
            best_params = (alpha, beta, gamma, delta)
    return lapin,renard,best_mse,best_params



step = 0.001

# Charger les données réelles
real_data = pd.read_csv("D:\\school\\HETIC\\PYTHON\\dev back\\Lotka-Volterra\\LotkaVolterra\\populations_lapins_renards.csv")
real_data.columns = real_data.columns.str.strip()
real_data['iteration'] = np.arange(1, len(real_data) + 1)  # Ajouter la colonne 'iteration'

# Afficher les données pour vérifier
#print(real_data.head())
#print(real_data.columns)

# Paramètres du modèle Lotka-Volterra
alpha_values = np.linspace(0.1, 2, 10)
beta_values =  np.linspace(0.1, 2, 10)
gamma_values = np.linspace(0.1, 2, 10)
delta_values = np.linspace(0.1, 2, 10)

#lapin, renard, best_mse, best_params = find_best_parameters(step, real_data, alpha_values, beta_values, gamma_values, delta_values)
#alpha, beta, gamma, delta = best_params
#print(f'\n\nBest parameters: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}')
#print(f'Best Mean Squared Error (MSE): {best_mse}\n\n')



#alpha=0.5222222222222223; beta=0.9444444444444444; gamma=1.1555555555555557; delta=1.1555555555555557
#Best Mean Squared Error (MSE): 0.07863417165777078

#alpha = 2 / 3;beta = 4 / 3;delta = 1 ;gamma = 1
#Mean Squared Error (MSE) : 0.12721828857025996

#alpha=0.5222222222222223; beta=0.5222222222222223; gamma=2.0; delta=0.1
#Best Mean Squared Error (MSE): 0.6866123891902626



#alpha = 0.5333333333333333 ; beta = 0.9333333333333333; gamma = 1.1333333333333333 ;delta = 1.3333333333333333     
#Best parameters: alpha=0.5222222222222223, beta=0.9444444444444444, gamma=1.1555555555555557, delta=1.1555555555555557
            
# Paramètres du modèle Lotka-Volterra

alpha=0.48; beta=0.86; gamma=1.2400000000000002; delta=1.2400000000000002

#alpha, beta, gamma, delta = best_params

time = [0]
lapin = [1]
renard = [2]
iteration = [1]
# Initialisation du MSE
mse = 0
msetst = 0

lapintst = [1]
renardtst = [2]
# Simulation du modèle pendant 100 000 itérations
for _ in range(100_000):
    time.append(time[-1] + step)
    iteration.append(iteration[-1] + 1)
    lapin.append(lapin[-1] * (alpha - beta * renard[-1]) * step + lapin[-1])
    renard.append(renard[-1] * (delta * lapin[-1] - gamma) * step + renard[-1])
    
    if iteration[-1] % 100 == 0:
        lapintst.append(lapin[-1] * (alpha - beta * renardtst[-1]) * step + lapin[-1])
        renardtst.append(renard[-1] * (delta * lapintst[-1] - gamma) * step + renard[-1])
    

    # Calculer le MSE à chaque itération pour les premiers N éléments
    if _ < len(real_data):
        mse_lapin = (lapin[-1] - (real_data["lapin"].iloc[_])/1000) ** 2
        mse_renard = (renard[-1] - (real_data["renard"].iloc[_])/1000) ** 2
        mse += mse_lapin + mse_renard


# Calculer le MSE moyen
mse = mse / (2*len(real_data))  # Diviser par 2 pour obtenir la moyenne (lapins et renards)
print(f'simulation using result parameters \nMean Squared Error (MSE) : {mse}')

# Convertir les listes en tableaux numpy pour la visualisation

lapin = np.array(lapin)*1000
#extract a value from lapin each 30 lines at a max of 1000
lapin = lapin[::30]
lapin = lapin[:1000]
renard = np.array(renard)*1000
#extract a value from renard each 30 lines at a max of 1000
renard = renard[::30]
renard = renard[:1000]
iteration = iteration[:1000]
lapintst = np.array(lapintst)*1000
renardtst = np.array(renardtst)*1000
#supprimmer la derniere valeur de lapintst et renardtst
lapintst = lapintst[:-1]
renardtst = renardtst[:-1]

#show lengths of time iteration lapin real_lapin renard real_renard lapintst renardtst
#print(len(time), len(iteration), len(lapin), len(renard), len(real_data["lapin"]), len(real_data["renard"]),)


def save_results(time, iteration, lapin, real_data, renard, ):
    # Ensure all arrays have the same length
    #min_length = min(len(time), len(iteration), len(lapin), len(real_data["lapin"]), len(renard), len(real_data["renard"]),)
    #time = time[:min_length]
    #iteration = iteration[:min_length]
    #lapin = lapin[:min_length]
    #real_lapin = real_data["lapin"][:min_length]
    #renard = renard[:min_length]
    #real_renard = real_data["renard"][:min_length]

    # Save all new data in a new csv file (results.csv)(time, iteration, lapin, real_lapin, renard, real_renard, lapintst, renardtst) if lenght different fil with 0
    # Ensure all arrays have the same length by filling with 0
    max_length = max(len(time), len(iteration), len(lapin), len(real_data["lapin"]), len(renard), len(real_data["renard"]))
    time = np.pad(time, (0, max_length - len(time)), 'constant')
    iteration = np.pad(iteration, (0, max_length - len(iteration)), 'constant')
    lapin = np.pad(lapin, (0, max_length - len(lapin)), 'constant')
    real_lapin = np.pad(real_data["lapin"], (0, max_length - len(real_data["lapin"])), 'constant')
    renard = np.pad(renard, (0, max_length - len(renard)), 'constant')
    real_renard = np.pad(real_data["renard"], (0, max_length - len(real_data["renard"])), 'constant')

    # Save all new data in a new csv file (results.csv)
    data = {'time': time, 'iteration': iteration, 'lapin': lapin, 'real_lapin': real_lapin, 'renard': renard, 'real_renard': real_renard}
    df = pd.DataFrame(data)
    df.to_csv('D:\\school\\HETIC\\PYTHON\\dev back\\Lotka-Volterra\\LotkaVolterra\\resultstest.csv', index=False)
save_results(time, iteration, lapin, real_data, renard)

# Visualize all data in a graph comparing the real and predicted data for both lapin and renard populations over time using the df
plt.figure(figsize=(10, 7))
plt.plot(real_data['iteration'], real_data["lapin"], "r--", label='Lapin (réel)')
plt.plot(real_data['iteration'], real_data["renard"], "b--", label='Renard (réel)')
plt.plot(iteration, lapin, "r", label='Lapin (modèle)')
plt.plot(iteration, renard, "b", label='Renard (modèle)')

plt.xlabel('Itération')
plt.ylabel('Population')
plt.title('Modèle Lotka-Volterra')
plt.legend()
plt.show()
