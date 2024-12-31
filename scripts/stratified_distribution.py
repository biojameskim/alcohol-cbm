import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Example data setup (adjust sizes to match your actual data)
np.random.seed(0) # For reproducibility
n_samples = 654
N_SPLITS = 5
y = np.load('data/training_data/aligned/y_aligned_control_moderate.npy')
site_data = np.load('data/training_data/aligned/site_location_control_moderate.npy')
simple_site_data = np.argmax(site_data, axis=1)

# Create composite stratification key
stratification_key = [str(a) + '_' + str(b) for a, b in zip(y, simple_site_data)]

# Initialize StratifiedKFold
outer_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Plot distribution for each fold
plt.figure(figsize=(15, 8))
overall_distribution = Counter(stratification_key)

for fold_idx, (train_index, test_index) in enumerate(outer_kf.split(np.zeros(n_samples), stratification_key)):
    train_keys = [stratification_key[i] for i in train_index]
    test_keys = [stratification_key[i] for i in test_index]
    
    # Count distributions in the train and test splits
    train_distribution = Counter(train_keys)
    test_distribution = Counter(test_keys)
    
    # Plot for each fold
    plt.subplot(N_SPLITS, 2, fold_idx * 2 + 1)
    plt.bar(*zip(*sorted(train_distribution.items())), color='blue', alpha=0.6)
    plt.title(f"Fold {fold_idx + 1} Training Distribution")
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    
    plt.subplot(N_SPLITS, 2, fold_idx * 2 + 2)
    plt.bar(*zip(*sorted(test_distribution.items())), color='orange', alpha=0.6)
    plt.title(f"Fold {fold_idx + 1} Testing Distribution")
    plt.ylabel('Counts')
    plt.xticks(rotation=45)

# Add overall distribution as a reference on the last plot
plt.figure(figsize=(8, 4))
plt.bar(*zip(*sorted(overall_distribution.items())), color='green', alpha=0.6)
plt.title("Overall Distribution")
plt.ylabel('Counts')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()