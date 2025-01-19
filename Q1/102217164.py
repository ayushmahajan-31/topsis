import pandas as pd  
import numpy as np  
import sys  
import os  

def validate_input(file_path, weights, impacts):
    """
    Validate the input file, weights, and impacts.
    Ensure they follow the correct format and criteria.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError("The input file does not exist. Please provide a valid file path.")

    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Check if there are at least 3 columns
    if data.shape[1] < 3:
        raise ValueError("The input file must have at least 3 columns (Name + Criteria).")

    # Ensure all columns except the first one have numeric values
    for column in data.columns[1:]:
        if not np.issubdtype(data[column].dtype, np.number):
            raise ValueError("All columns from the 2nd to the last must contain numeric values.")

    # Ensure the number of weights, impacts, and numeric columns match
    if len(weights) != len(impacts) or len(weights) != (data.shape[1] - 1):
        raise ValueError("The number of weights, impacts, and numeric columns must be the same.")

    # Check if impacts are valid (only '+' or '-')
    if not all(impact in ['+', '-'] for impact in impacts):
        raise ValueError("Impacts must be '+' or '-'.")

def topsis(input_file, weights, impacts, output_file):
    """
    Perform the TOPSIS analysis and save the results to the output file.
    """
    # Load the data
    data = pd.read_csv(input_file)
    # Extract criteria values and names
    criteria = data.iloc[:, 1:].values  # Exclude the first column (e.g., Fund Names)
    names = data.iloc[:, 0]  # First column (e.g., Fund Names)

    # Step 1: Normalize the criteria matrix
    norm_matrix = criteria / np.sqrt((criteria ** 2).sum(axis=0))

    # Step 2: Multiply each column by its weight
    weights = np.array(weights)
    weighted_matrix = norm_matrix * weights

    # Step 3: Determine the Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
    pis = [max(weighted_matrix[:, i]) if impacts[i] == '+' else min(weighted_matrix[:, i]) for i in range(len(weights))]
    nis = [min(weighted_matrix[:, i]) if impacts[i] == '+' else max(weighted_matrix[:, i]) for i in range(len(weights))]

    # Step 4: Calculate the separation measures
    sip = np.sqrt(((weighted_matrix - pis) ** 2).sum(axis=1))  # Separation from PIS
    sin = np.sqrt(((weighted_matrix - nis) ** 2).sum(axis=1))  # Separation from NIS

    # Step 5: Compute the TOPSIS score
    scores = sin / (sip + sin)

    # Step 6: Rank the alternatives
    ranks = scores.argsort()[::-1] + 1  # Higher score gets a better rank

    # Add results to the original data
    data['TOPSIS Score'] = scores
    data['Rank'] = ranks

    # Save the output to a new file
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python 102217164.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    # Parse command-line arguments
    input_file = sys.argv[1]
    weights = list(map(float, sys.argv[2].split(',')))
    impacts = sys.argv[3].split(',')
    output_file = sys.argv[4]

    try:
        # Validate the inputs
        validate_input(input_file, weights, impacts)
        # Perform the TOPSIS analysis
        topsis(input_file, weights, impacts, output_file)
    except Exception as e:
        print(f"Error: {e}")
