import numpy as np
import pandas as pd 

def read_data(filename):
    
    data = pd.read_csv(filename)

    print(data.head(10))
    trainingdata = data.iloc[::5,:] 
    

if __name__ == '__main__':
    read_data('ds-1.txt')
