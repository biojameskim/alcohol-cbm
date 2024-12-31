import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

def get_sig_indices(control_only, male, female, p_value_threshold=0.05):
  """
  [get_sig_indices] returns an array where the index of significant {SC, FC, FCgsr} coefficients are marked with True.
  """
  if control_only:
    file_name = 'control'
  else:
    file_name = 'control_moderate'
  
  if male:
    sex = '_male'
  elif female:
    sex = '_female'
  else:
    sex = ''

  # Use coefs before haufe transformation
  SC_coefs = np.load(f'results/SC/logreg/{file_name}/logreg_SC_{file_name}{sex}_coefficients.npy')
  FC_coefs = np.load(f'results/FC/logreg/{file_name}/logreg_FC_{file_name}{sex}_coefficients.npy')
  FCgsr_coefs = np.load(f'results/FCgsr/logreg/{file_name}/logreg_FCgsr_{file_name}{sex}_coefficients.npy')

  # Perform t-test to determine significant coefficients
  SC_pvalues = ttest_1samp(SC_coefs, 0, axis=0).pvalue
  FC_pvalues = ttest_1samp(FC_coefs, 0, axis=0).pvalue
  FCgsr_pvalues = ttest_1samp(FCgsr_coefs, 0, axis=0).pvalue

  # Print number of sig coefs before BH correction
  print(f"SC: {np.sum(SC_pvalues < p_value_threshold)} significant coefficients (before BH)")
  print(f"FC: {np.sum(FC_pvalues < p_value_threshold)} significant coefficients (before BH)")
  print(f"FCgsr: {np.sum(FCgsr_pvalues < p_value_threshold)} significant coefficients (before BH)")
  
  # Adjust p-values for multiple comparisons using the Benjamini-Hochberg procedure
  SC_reject, _, _, _ = multipletests(SC_pvalues, alpha=p_value_threshold, method='fdr_bh')
  FC_reject, _, _, _ = multipletests(FC_pvalues, alpha=p_value_threshold, method='fdr_bh')
  FCgsr_reject, _, _, _ = multipletests(FCgsr_pvalues, alpha=p_value_threshold, method='fdr_bh')

  # These are indicator arrays where the index of significant coefficients are marked with True
  return SC_reject, FC_reject, FCgsr_reject

if __name__ == '__main__':
  CONTROL_ONLY = False
  MALE = False
  FEMALE = False

  SC_reject, FC_reject, FCgsr_reject = get_sig_indices(control_only=CONTROL_ONLY, male=MALE, female=FEMALE, p_value_threshold=0.05)
  print(SC_reject.shape, FC_reject.shape, FCgsr_reject.shape)

  # Print number of sig coefs after BH correction
  print(f"SC: {np.sum(SC_reject)} significant coefficients (after BH)")
  print(f"FC: {np.sum(FC_reject)} significant coefficients (after BH)")
  print(f"FCgsr: {np.sum(FCgsr_reject)} significant coefficients (after BH)")