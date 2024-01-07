import numpy as np
import pandas as pd

def calculate_mape(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Input arrays must have the same length.")

    absolute_percentage_errors = []
    for i in range(len(actual)):
        actual_value = float(actual[i])  # Convert to float
        predicted_value = float(predicted[i]) 
        if actual_value != 0:
            absolute_percentage_error = abs((actual_value - predicted_value) / actual_value) * 100
            absolute_percentage_errors.append(absolute_percentage_error)

    mape = sum(absolute_percentage_errors) / len(actual)
    return mape

def calculate_rmse(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Input arrays must have the same length.")

    actual = np.array(actual, dtype='float64')
    predicted = np.array(predicted, dtype='float64')
    squared_errors = (actual - predicted) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    
    return rmse

# Load data from the NPZ file
data = np.load('data/y_test.npy') 
forecasts='forecasts/mlp_counorm_in168_l7_512_mae_12h.csv' #forecasts/frc_tso_forecasts.csv
forecasts=pd.read_csv(forecasts)
mape_list=[]
Sweden_mape_list=[]
CzechRepublic_mape_list=[]
Latvia_mape_list=[]
Italy_mape_list=[]
Spain_mape_list=[]
France_mape_list=[]
Slovakia_mape_list=[]
Denmark_mape_list=[]
Greece_mape_list=[]
Germany_mape_list=[]
Poland_mape_list=[]
Norway_mape_list=[]
Hungary_mape_list=[]
Montenegro_mape_list=[]
Slovenia_mape_list=[]
Croatia_mape_list=[]
Lithuania_mape_list=[]
Portugal_mape_list=[]
Finland_mape_list=[]
Romania_mape_list=[]
Serbia_mape_list=[]
Austria_mape_list=[]
Belgium_mape_list=[]
Bulgaria_mape_list=[]

rmse_list=[]
Sweden_rmse_list=[]
CzechRepublic_rmse_list=[]
Latvia_rmse_list=[]
Italy_rmse_list=[]
Spain_rmse_list=[]
France_rmse_list=[]
Slovakia_rmse_list=[]
Denmark_rmse_list=[]
Greece_rmse_list=[]
Germany_rmse_list=[]
Poland_rmse_list=[]
Norway_rmse_list=[]
Hungary_rmse_list=[]
Montenegro_rmse_list=[]
Slovenia_rmse_list=[]
Croatia_rmse_list=[]
Lithuania_rmse_list=[]
Portugal_rmse_list=[]
Finland_rmse_list=[]
Romania_rmse_list=[]
Serbia_rmse_list=[]
Austria_rmse_list=[]
Belgium_rmse_list=[]
Bulgaria_rmse_list=[]

for i in range(len(data)): #
    actuals=data[i][3:]
    actuals=actuals[::-1]
    actuals=actuals[:24]
    #actuals=actuals[-24:]
    #print(actuals)
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Sweden'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Sweden_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Sweden_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='CzechRepublic'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      CzechRepublic_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      CzechRepublic_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Latvia'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Latvia_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Latvia_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Italy'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Italy_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Italy_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Spain'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Spain_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Spain_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='France'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      France_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      France_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Slovakia'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Slovakia_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Slovakia_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Denmark'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Denmark_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Denmark_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Greece'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Greece_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Greece_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Germany'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Germany_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Germany_rmse_list.append(rmse)
      rmse_list.append(rmse)
            
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Poland'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Poland_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Poland_rmse_list.append(rmse)
      rmse_list.append(rmse)
            
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Norway'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Norway_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Norway_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Hungary'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Hungary_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Hungary_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Montenegro'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Montenegro_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Montenegro_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Slovenia'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Slovenia_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Slovenia_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Croatia'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Croatia_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Croatia_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Portugal'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Portugal_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Portugal_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Lithuania'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Lithuania_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Lithuania_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Finland'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Finland_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Finland_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Romania'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Romania_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Romania_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Bulgaria'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Bulgaria_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Bulgaria_rmse_list.append(rmse)
      rmse_list.append(rmse)

    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Serbia'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Serbia_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Serbia_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Austria'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Austria_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Austria_rmse_list.append(rmse)
      rmse_list.append(rmse)
      
    current_forecasts=forecasts.iloc[i]
    if(current_forecasts['Country']=='Belgium'):
      current_forecasts = current_forecasts.to_numpy()
      current_forecasts = current_forecasts[2:]
      current_forecasts=current_forecasts[::-1]
      current_forecasts = current_forecasts[:24]
      #print(current_forecasts)
      mape = calculate_mape(actuals, current_forecasts)
      Belgium_mape_list.append(mape)
      mape_list.append(mape)
      rmse = calculate_rmse(actuals, current_forecasts)
      Belgium_rmse_list.append(rmse)
      rmse_list.append(rmse)

average_mape=0

print("\nBelow you can see the MAPE perforance per country:\n")

mape = sum(Sweden_mape_list) / len(Sweden_mape_list)
print("\nSweden: ")
print(mape)
average_mape=average_mape+mape
mape = sum(CzechRepublic_mape_list) / len(CzechRepublic_mape_list)
print("\nCzeckRepublic: ")
print(mape)
average_mape=average_mape+mape
mape = sum(Latvia_mape_list) / len(Latvia_mape_list)
print("\nLatvia: ")
print(mape)
average_mape=average_mape+mape
mape = sum(Italy_mape_list) / len(Italy_mape_list)
print("\nItaly :")
print(mape)
average_mape=average_mape+mape
mape = sum(Spain_mape_list) / len(Spain_mape_list)
print("\nSpain :")
print(mape)
average_mape=average_mape+mape
mape = sum(France_mape_list) / len(France_mape_list)
print("\nFrance :")
print(mape)
average_mape=average_mape+mape
mape = sum(Slovakia_mape_list) / len(Slovakia_mape_list)
print("\nSlovakia :")
print(mape)
average_mape=average_mape+mape
mape = sum(Denmark_mape_list) / len(Denmark_mape_list)
print("\nDenmark :")
print(mape)
average_mape=average_mape+mape
mape = sum(Greece_mape_list) / len(Greece_mape_list)
print("\nGreece :")
print(mape)
average_mape=average_mape+mape
mape = sum(Germany_mape_list) / len(Germany_mape_list)
print("\nGermany :")
print(mape)
average_mape=average_mape+mape
mape = sum(Poland_mape_list) / len(Poland_mape_list)
print("\nPoland :")
print(mape)
average_mape=average_mape+mape
mape = sum(Norway_mape_list) / len(Norway_mape_list)
print("\nNorway :")
print(mape)
average_mape=average_mape+mape
mape = sum(Hungary_mape_list) / len(Hungary_mape_list)
print("\nHungary :")
print(mape)
average_mape=average_mape+mape
mape = sum(Montenegro_mape_list) / len(Montenegro_mape_list)
print("\nMontenegro :")
print(mape)
average_mape=average_mape+mape
mape = sum(Slovenia_mape_list) / len(Slovenia_mape_list)
print("\nSlovenia :")
print(mape)
average_mape=average_mape+mape
mape = sum(Croatia_mape_list) / len(Croatia_mape_list)
print("\nCroatia :")
print(mape)
average_mape=average_mape+mape
mape = sum(Lithuania_mape_list) / len(Lithuania_mape_list)
print("\nLithuania :")
print(mape)
average_mape=average_mape+mape
mape = sum(Portugal_mape_list) / len(Portugal_mape_list)
print("\nPortugal :")
print(mape)
average_mape=average_mape+mape
mape = sum(Finland_mape_list) / len(Finland_mape_list)
print("\nFinland :")
print(mape)
average_mape=average_mape+mape
mape = sum(Romania_mape_list) / len(Romania_mape_list)
print("\nRomania :")
print(mape)
mape = sum(Bulgaria_mape_list) / len(Bulgaria_mape_list)
print("\nBulgaria :")
print(mape)
average_mape=average_mape+mape
average_mape=average_mape+mape
mape = sum(Serbia_mape_list) / len(Serbia_mape_list)
print("\nSerbia :")
print(mape)
average_mape=average_mape+mape
mape = sum(Austria_mape_list) / len(Austria_mape_list)
print("\nAustria :")
print(mape)
average_mape=average_mape+mape
mape = sum(Belgium_mape_list) / len(Belgium_mape_list)
print("\nBelgium :")
print(mape)
average_mape=average_mape+mape

average_mape=average_mape/23

print("\nAverage MAPE:\n")
#print(average_mape)

mape = sum(mape_list) / len(mape_list)
print(mape)

print("\nBelow you can see the RMSE perforance per country:\n")

average_rmse=0

rmse = sum(Sweden_rmse_list) / len(Sweden_rmse_list)
print("\nSweden: ")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(CzechRepublic_rmse_list) / len(CzechRepublic_rmse_list)
print("\nCzeckRepublic: ")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Latvia_rmse_list) / len(Latvia_rmse_list)
print("\nLatvia: ")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Italy_rmse_list) / len(Italy_rmse_list)
print("\nItaly :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Spain_rmse_list) / len(Spain_rmse_list)
print("\nSpain :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(France_rmse_list) / len(France_rmse_list)
print("\nFrance :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Slovakia_rmse_list) / len(Slovakia_rmse_list)
print("\nSlovakia :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Denmark_rmse_list) / len(Denmark_rmse_list)
print("\nDenmark :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Greece_rmse_list) / len(Greece_rmse_list)
print("\nGreece :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Germany_rmse_list) / len(Germany_rmse_list)
print("\nGermany :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Poland_rmse_list) / len(Poland_rmse_list)
print("\nPoland :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Norway_rmse_list) / len(Norway_rmse_list)
print("\nNorway :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Hungary_rmse_list) / len(Hungary_rmse_list)
print("\nHungary :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Montenegro_rmse_list) / len(Montenegro_rmse_list)
print("\nMontenegro :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Slovenia_rmse_list) / len(Slovenia_rmse_list)
print("\nSlovenia :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Croatia_rmse_list) / len(Croatia_rmse_list)
print("\nCroatia :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Lithuania_rmse_list) / len(Lithuania_rmse_list)
print("\nLithuania :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Portugal_rmse_list) / len(Portugal_rmse_list)
print("\nPortugal :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Finland_rmse_list) / len(Finland_rmse_list)
print("\nFinland :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Romania_rmse_list) / len(Romania_rmse_list)
print("\nRomania :")
print(rmse)
rmse = sum(Bulgaria_rmse_list) / len(Bulgaria_rmse_list)
print("\nBulgaria :")
print(rmse)
average_rmse=average_rmse+rmse
average_rmse=average_rmse+rmse
rmse = sum(Serbia_rmse_list) / len(Serbia_rmse_list)
print("\nSerbia :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Austria_rmse_list) / len(Austria_rmse_list)
print("\nAustria :")
print(rmse)
average_rmse=average_rmse+rmse
rmse = sum(Belgium_rmse_list) / len(Belgium_rmse_list)
print("\nBelgium :")
print(rmse)
average_rmse=average_rmse+rmse

average_rmse=average_rmse/23

print("\nAverage rmse:\n")
#print(average_rmse)


rmse = sum(rmse_list) / len(rmse_list)
print(rmse)