import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

def regress_out_demos(dataset, covariate_df, control_only=False):
    n_features = dataset.shape[1] # Number of features

    # Rename columns in covariate dataframe (African-American/Black to African_American_Black)
    # print("column names before:", covariate_df.columns)
    covariate_df.columns = covariate_df.columns.str.replace('-', '_').str.replace('/', '_')
    print("column names after:", covariate_df.columns)

    # Loop over each feature to regress out the demographic effects
    for i in range(n_features):
        feature = dataset[:, i]
        # Combine covariate data with the current feature to create the table
        tbl_df = covariate_df.copy()
        tbl_df['feature'] = feature
        
        # Fit a linear mixed effects model
        if control_only:
            columns = ['visit_age', 'sex', 'African_American_Black', 'Asian', 'Other', 'ses_parent_yoe', 'A', 'B', 'C', 'D', 'scanner_model']
            model_formula = 'feature ~ ' + ' + '.join(columns)
        else:
            # columns = ['moderate', 'visit_age', 'sex', 'African_American_Black', 'Asian', 'Other', 'ses_parent_yoe', 'A', 'B', 'C', 'D', 'scanner_model']
            columns = ['visit_age', 'sex', 'African_American_Black', 'Asian', 'Other', 'ses_parent_yoe', 'A', 'B', 'C', 'D', 'scanner_model']  # try simpler model to diagnose
            model_formula = 'feature ~ ' + ' + '.join(columns)
        
        try:
            model = MixedLM.from_formula(model_formula, tbl_df, groups=tbl_df['subject'])
            lme_results = model.fit(method='powell')  # use a different solver if default fails
            print(f"Model summary for feature {i}:\n", lme_results.summary())
            
            # Adjust the feature to regress out the effect of the covariates
            # Extract the fixed effects coefficients 
            estimates = lme_results.fe_params[1:] # Ignore the intercept and Group Variances

            # Debugging information
            print(f"Feature index {i}")
            print(f"Estimates length: {len(estimates)}")
            print(f"Columns shape: {tbl_df[columns].shape}")

            # Calculate the predicted values from covariates
            predicted_values = tbl_df[columns].dot(estimates)
            adjusted_feature = feature - predicted_values
            dataset[:, i] = adjusted_feature

        except np.linalg.LinAlgError as e:
            print(f"Error in feature {i}: {str(e)}")
            continue

    return dataset

def calculate_vif(covariate_df):
    features = columns = ['moderate', 'visit_age', 'sex', 'African_American_Black', 'Asian', 'Other', 'ses_parent_yoe', 'A', 'B', 'C', 'D', 'TrioTim'] + ['MR750']
    covariate_df.columns = covariate_df.columns.str.replace('-', '_').str.replace('/', '_')
    df = covariate_df

    # Adding a constant of 1 for intercept variance control
    X = df[features].assign(Intercept=1)
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    # Printing Table of VIF, ignoring intercept in practical use
    print(vif_data)
    print(covariate_df.describe())

if __name__ == '__main__':
    CONTROL_ONLY = False

    aligned_SC_data = np.load('data/training_data/aligned/X_SC_control_moderate.npy')
    aligned_FC_data = np.load('data/training_data/aligned/X_FC_control_moderate.npy')
    aligned_FCgsr_data = np.load('data/training_data/aligned/X_FCgsr_control_moderate.npy')

    if CONTROL_ONLY:
        demos_df = pd.read_csv('data/data_with_subject_ids/aligned/control_demos_with_subjects.csv')
    else:
        demos_df = pd.read_csv('data/data_with_subject_ids/aligned/control_moderate_demos_with_subjects.csv')

    # print(demos_df['MR750'].sum())
    # print(demos_df['Prisma_Fit'].sum())
    # calculate_vif(demos_df)  # Including MR750
    # print(demos_df['African-American/Black'].sum())
    # print(demos_df['Asian'].sum())
    # print(demos_df['Other'].sum())
    # print(demos_df['A'].sum())
    # print(demos_df['B'].sum())
    # print(demos_df['C'].sum())
    # print(demos_df['D'].sum())

    SC_demos_regressed = regress_out_demos(aligned_SC_data, demos_df, control_only=CONTROL_ONLY)
    np.save('data/training_data/demos_regressed/X_SC_control_moderate_demos_regressed.npy', SC_demos_regressed)

    FC_demos_regressed = regress_out_demos(aligned_FC_data, demos_df, control_only=CONTROL_ONLY)
    np.save('data/training_data/demos_regressed/X_FC_control_moderate_demos_regressed.npy', FC_demos_regressed)
    
    FCgsr_demos_regressed = regress_out_demos(aligned_FCgsr_data, demos_df, control_only=CONTROL_ONLY)
    np.save('data/training_data/demos_regressed/X_FCgsr_control_moderate_demos_regressed.npy', FCgsr_demos_regressed)

    print("Shape of SC data after regressing out demos:", SC_demos_regressed.shape)
    print("Shape of FC data after regressing out demos:", FC_demos_regressed.shape)
    print("Shape of FCgsr data after regressing out demos:", FCgsr_demos_regressed.shape)

    print("Regressed out demographic effects and saved the new datasets")