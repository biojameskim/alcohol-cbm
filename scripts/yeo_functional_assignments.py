import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from edges_heatmap import upper_tri_to_matrix
from sig_coefs import get_sig_indices
from get_haufe_coefs import get_haufe_coefs

def get_network_indices(yeo_assignment_file):
  """
  [get_network_indices] reads the Yeo7 assignment file and returns the indices where the regions in the SC and FC matrices belong to each network
  """
  df = pd.read_csv(yeo_assignment_file)
  df.loc[df['Yeo7_Index'] > 7, 'Yeo7_Index'] = df.loc[df['Yeo7_Index'] > 7, 'Yeo7_Index'] - 7 # Adjust indices for 9-network parcellation

  df_SC = df[df['Used for tractography subcortical'] == 'Y']
  df_SC.reset_index(drop=True, inplace=True)

  df_FC = df[df['Used for rs-fMRI gm signal'] == 'Y']
  df_FC.reset_index(drop=True, inplace=True)

  # Dictionaries to store the results
  SC_network_indices = {}
  FC_network_indices = {}

  for network in [1,2,3,4,5,6,7,8]:
    # Find indices in df_SC and df_FC where 'Yeo7_Index' matches the network value
    SC_network = np.where(df_SC['Yeo7_Index'] == network)
    FC_network = np.where(df_FC['Yeo7_Index'] == network)
    
    # Save the results into dictionaries
    SC_network_indices[network] = SC_network
    FC_network_indices[network] = FC_network
  
  FC_network_indices[9] = np.where(df_FC['Yeo7_Index'] == 9) # Only FC/FCgsr has Cerebellum regions

  return SC_network_indices, FC_network_indices

def get_network_influences(SC_region_vector, FC_region_vector, FCgsr_region_vector, SC_pos_region_vector, SC_neg_region_vector, FC_pos_region_vector, FC_neg_region_vector, FCgsr_pos_region_vector, FCgsr_neg_region_vector, SC_network_indices, FC_network_indices):
  """
  [get_network_influences] calculates the mean influence of each network in the SC and FC matrices for positive and negative values separately.
  """
  SC_network_influences = {}
  FC_network_influences = {}
  FCgsr_network_influences = {}

  SC_pos_network_influences = {}
  SC_neg_network_influences = {}
  FC_pos_network_influences = {}

  FC_neg_network_influences = {}
  FCgsr_pos_network_influences = {}
  FCgsr_neg_network_influences = {}

  for network in [1,2,3,4,5,6,7,8]:
    # Get the indices for the network
    SC_indices = SC_network_indices[network][0]
    FC_indices = FC_network_indices[network][0]

    # Calculate the mean influence for the network
    SC_network_influences[network] = np.round(np.mean(SC_region_vector[SC_indices]), 3)
    FC_network_influences[network] = np.round(np.mean(FC_region_vector[FC_indices]), 3)
    FCgsr_network_influences[network] = np.round(np.mean(FCgsr_region_vector[FC_indices]), 3)

    SC_pos_network_influences[network] = np.round(np.mean(SC_pos_region_vector[SC_indices]), 3)
    SC_neg_network_influences[network] = np.round(np.mean(SC_neg_region_vector[SC_indices]), 3)
    FC_pos_network_influences[network] = np.round(np.mean(FC_pos_region_vector[FC_indices]), 3)

    FC_neg_network_influences[network] = np.round(np.mean(FC_neg_region_vector[FC_indices]), 3)
    FCgsr_pos_network_influences[network] = np.round(np.mean(FCgsr_pos_region_vector[FC_indices]), 3)
    FCgsr_neg_network_influences[network] = np.round(np.mean(FCgsr_neg_region_vector[FC_indices]), 3)
  
  # Only FC/FCgsr has Cerebellum regions
  FC_network_influences[9] = np.round(np.mean(FC_region_vector[FC_network_indices[9][0]]), 3)
  FCgsr_network_influences[9] = np.round(np.mean(FCgsr_region_vector[FC_network_indices[9][0]]), 3)

  FC_pos_network_influences[9] = np.round(np.mean(FC_pos_region_vector[FC_network_indices[9][0]]), 3)
  FC_neg_network_influences[9] = np.round(np.mean(FC_neg_region_vector[FC_network_indices[9][0]]), 3)

  FCgsr_pos_network_influences[9] = np.round(np.mean(FCgsr_pos_region_vector[FC_network_indices[9][0]]), 3)
  FCgsr_neg_network_influences[9] = np.round(np.mean(FCgsr_neg_region_vector[FC_network_indices[9][0]]), 3)

  network_influences = {
    "SC_network_influences": SC_network_influences,
    "FC_network_influences": FC_network_influences,
    "FCgsr_network_influences": FCgsr_network_influences,
    "SC_pos_network_influences": SC_pos_network_influences,
    "SC_neg_network_influences": SC_neg_network_influences,
    "FC_pos_network_influences": FC_pos_network_influences,
    "FC_neg_network_influences": FC_neg_network_influences,
    "FCgsr_pos_network_influences": FCgsr_pos_network_influences,
    "FCgsr_neg_network_influences": FCgsr_neg_network_influences
  }

  return network_influences

def yeo_network_barplot(network_influences, file_name, sex):
  networks = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex', 'Cerebellum']
  positive_coeffs_SC = [network_influences['SC_pos_network_influences'][1], network_influences['SC_pos_network_influences'][2], network_influences['SC_pos_network_influences'][3], network_influences['SC_pos_network_influences'][4], network_influences['SC_pos_network_influences'][5], network_influences['SC_pos_network_influences'][6], network_influences['SC_pos_network_influences'][7], network_influences['SC_pos_network_influences'][8], 0]
  negative_coeffs_SC = [network_influences['SC_neg_network_influences'][1], network_influences['SC_neg_network_influences'][2], network_influences['SC_neg_network_influences'][3], network_influences['SC_neg_network_influences'][4], network_influences['SC_neg_network_influences'][5], network_influences['SC_neg_network_influences'][6], network_influences['SC_neg_network_influences'][7], network_influences['SC_neg_network_influences'][8], 0]
  positive_coeffs_FC = [network_influences['FC_pos_network_influences'][1], network_influences['FC_pos_network_influences'][2], network_influences['FC_pos_network_influences'][3], network_influences['FC_pos_network_influences'][4], network_influences['FC_pos_network_influences'][5], network_influences['FC_pos_network_influences'][6], network_influences['FC_pos_network_influences'][7], network_influences['FC_pos_network_influences'][8], network_influences['FC_pos_network_influences'][9]]
  negative_coeffs_FC = [network_influences['FC_neg_network_influences'][1], network_influences['FC_neg_network_influences'][2], network_influences['FC_neg_network_influences'][3], network_influences['FC_neg_network_influences'][4], network_influences['FC_neg_network_influences'][5], network_influences['FC_neg_network_influences'][6], network_influences['FC_neg_network_influences'][7], network_influences['FC_neg_network_influences'][8], network_influences['FC_neg_network_influences'][9]]

  print(positive_coeffs_SC)
  print(negative_coeffs_SC)
  print(positive_coeffs_FC)
  print(negative_coeffs_FC)

  # Number of networks
  n = len(networks)
  # Position of the bars on the x-axis
  ind = np.arange(n)
  # Width of a bar
  width = 0.35

  fig, ax = plt.subplots(figsize=(10, 7))

  # Plotting the bars
  bars1 = ax.bar(ind - width/2, positive_coeffs_SC, width, label='Positive SC', color='b')
  bars2 = ax.bar(ind + width/2, positive_coeffs_FC, width, label='Positive FC', color='g')
  bars3 = ax.bar(ind - width/2, negative_coeffs_SC, width, label='Negative SC', color='r')
  bars4 = ax.bar(ind + width/2, negative_coeffs_FC, width, label='Negative FC', color='orange')

  # Adding a horizontal line at y=0
  ax.axhline(0, color='black', linewidth=0.8)

  # Adding labels
  ax.set_xlabel('Networks')
  ax.set_ylabel('Mean Coefficients')
  ax.set_title('Yeo Network Mean Influences')
  ax.set_xticks(ind)
  ax.set_xticklabels(networks)
  ax.legend()

  # Set y-axis limits
  ax.set_ylim(-8.9, 8.9)
  
  # plt.show()
  plt.savefig(f'figures/yeo_network_barplots/yeo_network_influences_barplot_{file_name}{sex}.png')

if __name__ == "__main__":
  # 9 networks in the Yeo7 parcellation where Subcortex=8 and Cerebellum=9
  # Note THAT SC DOES NOT HAVE CEREBELLAR REGIONS (NETWORK 9)

  CONTROL_ONLY = False
  MALE = True
  FEMALE = False

  if CONTROL_ONLY:
    file_name = 'control'
    control_str = "Baseline subjects with cahalan=='control' only"
  else:
    file_name = 'control_moderate'
    control_str = "Baseline subjects with cahalan=='control' or cahalan=='moderate'"  
  
  if MALE:
    sex = '_male'
    sex_str = "Male subjects only"
  elif FEMALE:
    sex = '_female'
    sex_str = "Female subjects only"
  else:
    sex = ''
    sex_str = "All subjects"

  SC_reject, FC_reject, FCgsr_reject = get_sig_indices(control_only=CONTROL_ONLY, male=MALE, female=FEMALE, p_value_threshold=0.05)

  avg_SC_haufe_coefs = get_haufe_coefs(matrix_type='SC', file_name=file_name, sex=sex)
  avg_FC_haufe_coefs = get_haufe_coefs(matrix_type='FC', file_name=file_name, sex=sex)
  avg_FCgsr_haufe_coefs = get_haufe_coefs(matrix_type='FCgsr', file_name=file_name, sex=sex)

  # Find indices where the coefficients are not significant (failed to reject --> False)
  SC_false_indices = np.where(SC_reject == False)
  FC_false_indices = np.where(FC_reject == False)
  FCgsr_false_indices = np.where(FCgsr_reject == False)
  # Set non-significant coefficients to 0
  avg_SC_haufe_coefs[SC_false_indices] = 0
  avg_FC_haufe_coefs[FC_false_indices] = 0
  avg_FCgsr_haufe_coefs[FCgsr_false_indices] = 0

  # Reconstruct square matrices from upper triangular matrices
  SC_coef_matrix = upper_tri_to_matrix(avg_SC_haufe_coefs, 90)
  FC_coef_matrix = upper_tri_to_matrix(avg_FC_haufe_coefs, 109)
  FCgsr_coef_matrix = upper_tri_to_matrix(avg_FCgsr_haufe_coefs, 109)

  # Sum each row of the region x region matrix to get a vector of region influences
  SC_region_vector = np.sum(SC_coef_matrix, axis=1)
  FC_region_vector = np.sum(FC_coef_matrix, axis=1)
  FCgsr_region_vector = np.sum(FCgsr_coef_matrix, axis=1) 

  # Sum each row of the region x region matrix to get separate vectors for positive and negative region influences
  SC_pos_region_vector = np.sum(np.where(SC_coef_matrix > 0, SC_coef_matrix, 0), axis=1)
  SC_neg_region_vector = np.sum(np.where(SC_coef_matrix < 0, SC_coef_matrix, 0), axis=1)
  FC_pos_region_vector = np.sum(np.where(FC_coef_matrix > 0, FC_coef_matrix, 0), axis=1)

  FC_neg_region_vector = np.sum(np.where(FC_coef_matrix < 0, FC_coef_matrix, 0), axis=1)
  FCgsr_pos_region_vector = np.sum(np.where(FCgsr_coef_matrix > 0, FCgsr_coef_matrix, 0), axis=1)
  FCgsr_neg_region_vector = np.sum(np.where(FCgsr_coef_matrix < 0, FCgsr_coef_matrix, 0), axis=1)

  # Get network indices
  SC_network_indices, FC_network_indices = get_network_indices('data/csv/tzo116plus_yeo7xhemi.csv')

  # Get network influences for positive and negative values separately
  network_influences = get_network_influences(SC_region_vector=SC_region_vector, FC_region_vector=FC_region_vector, FCgsr_region_vector=FCgsr_region_vector, SC_pos_region_vector=SC_pos_region_vector, SC_neg_region_vector=SC_neg_region_vector, FC_pos_region_vector=FC_pos_region_vector, FC_neg_region_vector=FC_neg_region_vector, FCgsr_pos_region_vector=FCgsr_pos_region_vector, FCgsr_neg_region_vector=FCgsr_neg_region_vector, SC_network_indices=SC_network_indices, FC_network_indices=FC_network_indices)

  # Create report to print out the network influences
  report_lines = [
    "Network Influences\n",

    f"{control_str}",
    f"{sex_str}\n",

    "Networks: {1: 'Visual (Vis)', 2: 'Somatomotor (SomMot)', 3: 'Dorsal Attention (DorsAttn)', 4: 'Ventral Attention (SalVentAttn)', 5: 'Limbic (Limbic)', 6: 'Frontoparietal Control (Cont)', 7: 'Default Networks (Default)', 8: 'Subcortex', 9: 'Cerebellum'}",
    "Combined Influences:",
    f"SC: {network_influences['SC_network_influences']}",
    f"FC: {network_influences['FC_network_influences']}",
    f"FCgsr: {network_influences['FCgsr_network_influences']}\n",
    "Positive Influences:",
    f"SC: {network_influences['SC_pos_network_influences']}",
    f"FC: {network_influences['FC_pos_network_influences']}",
    f"FCgsr: {network_influences['FCgsr_pos_network_influences']}\n",
    "Negative Influences:",
    f"SC: {network_influences['SC_neg_network_influences']}",
    f"FC: {network_influences['FC_neg_network_influences']}",
    f"FCgsr: {network_influences['FCgsr_neg_network_influences']}\n",
    "\n",
    "\n",
    "Different view:\n",
    "SC, SC (Pos), SC (Neg), FC, FC (Pos), FC (Neg), FCgsr, FCgsr (Pos), FCgsr (Neg)",
    f"Network 1: {network_influences['SC_network_influences'][1]}, {network_influences['SC_pos_network_influences'][1]}, {network_influences['SC_neg_network_influences'][1]}, {network_influences['FC_network_influences'][1]}, {network_influences['FC_pos_network_influences'][1]}, {network_influences['FC_neg_network_influences'][1]}, {network_influences['FCgsr_network_influences'][1]}, {network_influences['FCgsr_pos_network_influences'][1]}, {network_influences['FCgsr_neg_network_influences'][1]}",
    f"Network 2: {network_influences['SC_network_influences'][2]}, {network_influences['SC_pos_network_influences'][2]}, {network_influences['SC_neg_network_influences'][2]}, {network_influences['FC_network_influences'][2]}, {network_influences['FC_pos_network_influences'][2]}, {network_influences['FC_neg_network_influences'][2]}, {network_influences['FCgsr_network_influences'][2]}, {network_influences['FCgsr_pos_network_influences'][2]}, {network_influences['FCgsr_neg_network_influences'][2]}",
    f"Network 3: {network_influences['SC_network_influences'][3]}, {network_influences['SC_pos_network_influences'][3]}, {network_influences['SC_neg_network_influences'][3]}, {network_influences['FC_network_influences'][3]}, {network_influences['FC_pos_network_influences'][3]}, {network_influences['FC_neg_network_influences'][3]}, {network_influences['FCgsr_network_influences'][3]}, {network_influences['FCgsr_pos_network_influences'][3]}, {network_influences['FCgsr_neg_network_influences'][3]}",
    f"Network 4: {network_influences['SC_network_influences'][4]}, {network_influences['SC_pos_network_influences'][4]}, {network_influences['SC_neg_network_influences'][4]}, {network_influences['FC_network_influences'][4]}, {network_influences['FC_pos_network_influences'][4]}, {network_influences['FC_neg_network_influences'][4]}, {network_influences['FCgsr_network_influences'][4]}, {network_influences['FCgsr_pos_network_influences'][4]}, {network_influences['FCgsr_neg_network_influences'][4]}",
    f"Network 5: {network_influences['SC_network_influences'][5]}, {network_influences['SC_pos_network_influences'][5]}, {network_influences['SC_neg_network_influences'][5]}, {network_influences['FC_network_influences'][5]}, {network_influences['FC_pos_network_influences'][5]}, {network_influences['FC_neg_network_influences'][5]}, {network_influences['FCgsr_network_influences'][5]}, {network_influences['FCgsr_pos_network_influences'][5]}, {network_influences['FCgsr_neg_network_influences'][5]}",
    f"Network 6: {network_influences['SC_network_influences'][6]}, {network_influences['SC_pos_network_influences'][6]}, {network_influences['SC_neg_network_influences'][6]}, {network_influences['FC_network_influences'][6]}, {network_influences['FC_pos_network_influences'][6]}, {network_influences['FC_neg_network_influences'][6]}, {network_influences['FCgsr_network_influences'][6]}, {network_influences['FCgsr_pos_network_influences'][6]}, {network_influences['FCgsr_neg_network_influences'][6]}",
    f"Network 7: {network_influences['SC_network_influences'][7]}, {network_influences['SC_pos_network_influences'][7]}, {network_influences['SC_neg_network_influences'][7]}, {network_influences['FC_network_influences'][7]}, {network_influences['FC_pos_network_influences'][7]}, {network_influences['FC_neg_network_influences'][7]}, {network_influences['FCgsr_network_influences'][7]}, {network_influences['FCgsr_pos_network_influences'][7]}, {network_influences['FCgsr_neg_network_influences'][7]}",
    f"Network 8: {network_influences['SC_network_influences'][8]}, {network_influences['SC_pos_network_influences'][8]}, {network_influences['SC_neg_network_influences'][8]}, {network_influences['FC_network_influences'][8]}, {network_influences['FC_pos_network_influences'][8]}, {network_influences['FC_neg_network_influences'][8]}, {network_influences['FCgsr_network_influences'][8]}, {network_influences['FCgsr_pos_network_influences'][8]}, {network_influences['FCgsr_neg_network_influences'][8]}",
    f"Network 9: N/A, N/A, N/A, {network_influences['FC_network_influences'][9]}, {network_influences['FC_pos_network_influences'][9]}, {network_influences['FC_neg_network_influences'][9]}, {network_influences['FCgsr_pos_network_influences'][9]}, {network_influences['FCgsr_pos_network_influences'][9]}, {network_influences['FCgsr_neg_network_influences'][9]}\n",

    "Networks: 1-7 = Yeo7 Networks, 8 = Subcortex, 9 = Cerebellum",
    "Network 1: Visual (Vis)",
    "Network 2: Somatomotor (SomMot)",
    "Network 3: Dorsal Attention (DorsAttn)",
    "Network 4: Ventral Attention (SalVentAttn)",
    "Network 5: Limbic (Limbic)",
    "Network 6: Frontoparietal Control (Cont)",
    "Network 7: Default Networks (Default)",
    "Network 8: Subcortex",
    "Network 9: Cerebellum"
  ]

  with open(f'results/reports/yeo_network_influences/posneg_yeo_network_influences_{file_name}{sex}.txt', 'w') as report_file:
    report_file.write("\n".join(report_lines))

  # Create barplot
  yeo_network_barplot(network_influences, file_name, sex)