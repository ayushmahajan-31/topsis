import pandas as pd
import numpy as np

class Topsis:
    def __init__(self, df, weights, impacts, distance_metric='euclidean', missing_data_strategy='mean'):
        self.df = self.fill_missing_data(df, strategy=missing_data_strategy)
        self.weights = weights
        self.impacts = impacts
        self.distance_metric = distance_metric

    def fill_missing_data(self, df, strategy='mean'):
        if strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'median':
            return df.fillna(df.median())
        else:
            return df

    def calculate(self):
        numeric_cols = self.df.select_dtypes(include=['number'])
        norm_df = numeric_cols.apply(lambda x: x / np.sqrt((x**2).sum()), axis=0)
        norm_df = norm_df * self.weights

        ideal_solution = norm_df.max()
        negative_ideal_solution = norm_df.min()

        for i, impact in enumerate(self.impacts):
            if impact == '-':
                ideal_solution[i], negative_ideal_solution[i] = negative_ideal_solution[i], ideal_solution[i]

        def calculate_distance(df, solution, metric):
            if metric == 'euclidean':
                return np.sqrt(((df - solution) ** 2).sum(axis=1))
            elif metric == 'manhattan':
                return np.abs(df - solution).sum(axis=1)
            else:
                raise ValueError(f"Unknown distance metric: {metric}")

        distance_ideal = calculate_distance(norm_df, ideal_solution, self.distance_metric)
        distance_negative_ideal = calculate_distance(norm_df, negative_ideal_solution, self.distance_metric)

        scores = distance_negative_ideal / (distance_ideal + distance_negative_ideal)

        result_df = pd.DataFrame({
            'Alternative': self.df.index,
            **numeric_cols.to_dict(orient='list'),
            'Topsis Score': scores
        })

        result_df['Rank'] = result_df['Topsis Score'].rank(ascending=False, method='max')
        result_df = result_df.sort_values(by='Rank', ascending=True)
        return result_df[['Alternative', 'Topsis Score', 'Rank']]
