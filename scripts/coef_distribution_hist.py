import numpy as np
import matplotlib.pyplot as plt

def plot_coefficients(matrix_type, save_fig=False):
  """
  [plot_coefficients] plot [matrix_type]'s coefficients as histograms to check the distribution of the coefficients
  """
  coefs = np.load(f'results/{matrix_type}/logreg/logreg_{matrix_type}_coefficients.npy')

  plt.figure(figsize=(10, 8))

  if matrix_type == 'demos':
    plt.xlim(-0.5, 0.5)
  else:
    plt.xlim(-0.2, 0.2)

  plt.hist(coefs.flatten(), bins=50, alpha=0.5, label=f'{matrix_type}')
  plt.xlabel('Coefficient Value')
  plt.ylabel('Frequency')
  plt.title(f'Histogram of {matrix_type} Coefficients')
  plt.legend()
  if save_fig:
    plt.savefig(f'figures/coef_distribution_hist/{matrix_type}_coefs_histogram.png')
  else:
    plt.show()

if __name__ == "__main__":
  plot_coefficients('SC', save_fig=True)
  plot_coefficients('FC', save_fig=True)
  plot_coefficients('FCgsr', save_fig=True)
  plot_coefficients('demos', save_fig=True)