import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
n = 200
rng = np.random.default_rng(0)
df = pd.DataFrame({
    'customer_id': [f"C{1000+i}" for i in range(n)],
    'churn': rng.integers(0,2,size=n),
    'monthly_charges': np.round(rng.uniform(15,120,size=n),2),
    'tenure_months': rng.integers(0,60,size=n),
})
df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)
df.to_csv('data/churn_sample.csv', index=False)
print('Wrote data/churn_sample.csv with', len(df), 'rows')
