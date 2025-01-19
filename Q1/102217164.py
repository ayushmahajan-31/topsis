import pandas as pd  
import numpy as np  
import sys  
import os  

def validate_input(file_path, weights, impacts):
    """
    Validate the input file, weights, and impacts.
    Ensure they follow the correct format and criteria.
    """
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError("The input file does not exist. Please provide a valid file path.")

    data = pd.read_csv(file_path)

    
    if data.shape[1] < 3:
        raise ValueError("The input file must have at least 3 columns (Name + Criteria).")

    
    for column in data.columns[1:]:
        if not np.issubdtype(data[column].dtype, np.number):
            raise ValueError("All columns from the 2nd to the last must contain numeric values.")

    
    if len(weights) != len(impacts) or len(weights) != (data.shape[1] - 1):
        raise ValueError("The number of weights, impacts, and numeric columns must be the same.")

   
    if not all(impact in ['+', '-'] for impact in impacts):
        raise ValueError("Impacts must be '+' or '-'.")

def topsis(input_file, weights, impacts, output_file):

    data = pd.read_csv(input_file)
   
    criteria = data.iloc[:, 1:].values  
    names = data.iloc[:, 0]  

    norm_matrix = criteria / np.sqrt((criteria ** 2).sum(axis=0))

    
    weights = np.array(weights)
    weighted_matrix = norm_matrix * weights

    
    pis = [max(weighted_matrix[:, i]) if impacts[i] == '+' else min(weighted_matrix[:, i]) for i in range(len(weights))]
    nis = [min(weighted_matrix[:, i]) if impacts[i] == '+' else max(weighted_matrix[:, i]) for i in range(len(weights))]

   
    sin = np.sqrt(((weighted_matrix - nis) ** 2).sum(axis=1))  
    sip = np.sqrt(((weighted_matrix - pis) ** 2).sum(axis=1))  

    
    scores = sin / (sip + sin)

    
    ranks = scores.argsort()[::-1] + 1  

    
    data['TOPSIS Score'] = scores
    data['Rank'] = ranks

    
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print("Usage: python 102217164.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    
    input_file = sys.argv[1]
    weights = list(map(float, sys.argv[2].split(',')))
    impacts = sys.argv[3].split(',')
    output_file = sys.argv[4]

    try:
        
        validate_input(input_file, weights, impacts)
       
        topsis(input_file, weights, impacts, output_file)
    except Exception as e:
        print(f"Error: {e}")
