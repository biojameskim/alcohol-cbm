import numpy as np
import pandas as pd
from edges_heatmap import upper_tri_to_matrix
from brainmontage import create_montage_figure
from get_haufe_coefs import get_haufe_coefs
from sig_coefs import get_sig_indices

def get_coefs_matrices(control_only, male, female):
    """
    [get_coefs_matrices] loads the Haufe coefficients for SC, FC, and FCgsr matrices, calculates the average coefficients, and converts the upper triangular coefficients to a square matrix
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

    # Load the Haufe coefficients
    SC_coefficients = get_haufe_coefs(matrix_type='SC', file_name=file_name, sex=sex)
    FC_coefficients = get_haufe_coefs(matrix_type='FC', file_name=file_name, sex=sex)
    FCgsr_coefficients = get_haufe_coefs(matrix_type='FCgsr', file_name=file_name, sex=sex)

    # Find indices where the coefficients are not significant (failed to reject --> False)
    SC_reject, FC_reject, FCgsr_reject = get_sig_indices(control_only=control_only, male=male, female=female, p_value_threshold=0.05)
    SC_false_indices = np.where(SC_reject == False)
    FC_false_indices = np.where(FC_reject == False)
    FCgsr_false_indices = np.where(FCgsr_reject == False)
    # Set non-significant coefficients to 0
    SC_coefficients[SC_false_indices] = 0
    FC_coefficients[FC_false_indices] = 0
    FCgsr_coefficients[FCgsr_false_indices] = 0

    # Convert the upper triangular coefficients to a square matrix
    SC_coefs_matrix = upper_tri_to_matrix(SC_coefficients, 90)
    FC_coefs_matrix = upper_tri_to_matrix(FC_coefficients, 109)
    FCgsr_coefs_matrix = upper_tri_to_matrix(FCgsr_coefficients, 109)

    return SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix

def get_aal_roi_vals(SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix):
    """
    [get_aal_roi_vals] calculates the average positive and negative coefficients for each AAL region and returns a dictionary of 1x116 aal vectors for {SC, FC, FCgsr} x {pos, neg} 
    """
    # Initialize with nan because any ROIs with "nan" will not be displayed
    SC_pos_coefs = np.full((116,), np.nan) 
    SC_neg_coefs = np.full((116,), np.nan)
    FC_pos_coefs = np.full((116,), np.nan) 
    FC_neg_coefs = np.full((116,), np.nan)
    FCgsr_pos_coefs = np.full((116,), np.nan) 
    FCgsr_neg_coefs = np.full((116,), np.nan)

    # SC - Cerebellar regions and Vermis set to nan (regions 91-116)
    for roi_idx in range(SC_coefs_matrix.shape[0]):
        SC_region = SC_coefs_matrix[roi_idx]
        pos_avg = np.sum(SC_region[SC_region > 0]) / len(SC_region)
        neg_avg = np.sum(SC_region[SC_region < 0]) / len(SC_region)
        SC_pos_coefs[roi_idx] = pos_avg
        SC_neg_coefs[roi_idx] = neg_avg

    # FC and FCgsr - Pallidum_L and Pallidum_R and Vermis set to nan (regions 75-76, 109-116)
    # Pallidum_L is at index 74, Pallidum_R is at index 75 so we need to skip over those (leave as nan)
    # FC matrix size is 109x109 but the last 3 regions are Vermis regions that we want to skip (so only index up to 105)
    # We skip Vermis regions because the AAL atlas has 8 Vermis regions but our data only has 3

    for idx in range(106): # 0 to 105 (We skip over the last 3 regions). Set range to 109 if you want to include the Vermis regions
        roi_idx = idx
        FC_region = FC_coefs_matrix[roi_idx]
        FCgsr_region = FCgsr_coefs_matrix[roi_idx]

        FC_pos_avg = np.sum(FC_region[FC_region > 0]) / len(FC_region)
        FC_neg_avg = np.sum(FC_region[FC_region < 0]) / len(FC_region)
        FCgsr_pos_avg = np.sum(FCgsr_region[FCgsr_region > 0]) / len(FCgsr_region)
        FCgsr_neg_avg = np.sum(FCgsr_region[FCgsr_region < 0]) / len(FCgsr_region)
        
        # Shift the indices by 2 after skipping over Pallidum_L and Pallidum_R
        if roi_idx >= 74:
            roi_idx = idx + 2

        FC_pos_coefs[roi_idx] = FC_pos_avg
        FC_neg_coefs[roi_idx] = FC_neg_avg
        FCgsr_pos_coefs[roi_idx] = FCgsr_pos_avg
        FCgsr_neg_coefs[roi_idx] = FCgsr_neg_avg


    aal_roi_vals = {
        "SC_pos_coefs": SC_pos_coefs,
        "SC_neg_coefs": SC_neg_coefs,
        "FC_pos_coefs": FC_pos_coefs,
        "FC_neg_coefs": FC_neg_coefs,
        "FCgsr_pos_coefs": FCgsr_pos_coefs,
        "FCgsr_neg_coefs": FCgsr_neg_coefs
    }

    return aal_roi_vals

def visualize_brain_regions(matrix_type, aal_roi_vals, positive, control_only, male, female):
    """
    [visualize_brain_regions] creates and saves a montage figure of the AAL regions with the average positive or negative coefficients for a given matrix type
    """
    if positive:
        sign = 'positive'
    else:
        sign = 'negative'

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

    if not control_only and not male and not female:
        clim=[-0.055,0.0965]
    else:
        clim=None

    # Create the montage figure
    create_montage_figure(aal_roi_vals,roilutfile="data/aal116_brainmontage/AAL116_LUT.tsv",lhannotfile="data/aal116_brainmontage/fsaverage.lh.AAL116.label.gii",rhannotfile="data/aal116_brainmontage/fsaverage.rh.AAL116.label.gii",annotsurfacename="fsaverage",subcorticalvolume="data/aal116_brainmontage/AAL116_subcortex.nii.gz",colormap="coolwarm",slice_dict={'axial':[23,33,43,53]},mosaic_dict={'axial':[-1,1]},add_colorbar=True,clim=clim,outputimagefile=f"figures/brain_regions/TEST_{matrix_type}_brain_regions_{sign}_{file_name}{sex}.png")

if __name__ == "__main__":
    CONTROL_ONLY = False
    MALE = False
    FEMALE = False

    if CONTROL_ONLY:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    if MALE:
        sex = '_male'
    elif FEMALE:
        sex = '_female'
    else:
        sex = ''

    SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix = get_coefs_matrices(control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    aal_roi_vals = get_aal_roi_vals(SC_coefs_matrix=SC_coefs_matrix, FC_coefs_matrix=FC_coefs_matrix, FCgsr_coefs_matrix=FCgsr_coefs_matrix)

    pd.DataFrame(aal_roi_vals).to_csv(f'data/regions_info/aal_roi_vals/{file_name}{sex}_aal_roi_vals.csv', index=False)

    visualize_brain_regions(matrix_type='SC', aal_roi_vals=aal_roi_vals["SC_pos_coefs"], positive=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    visualize_brain_regions(matrix_type='SC', aal_roi_vals=aal_roi_vals["SC_neg_coefs"], positive=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    visualize_brain_regions(matrix_type='FC', aal_roi_vals=aal_roi_vals["FC_pos_coefs"], positive=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    visualize_brain_regions(matrix_type='FC', aal_roi_vals=aal_roi_vals["FC_neg_coefs"], positive=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    visualize_brain_regions(matrix_type='FCgsr', aal_roi_vals=aal_roi_vals["FCgsr_pos_coefs"], positive=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    visualize_brain_regions(matrix_type='FCgsr', aal_roi_vals=aal_roi_vals["FCgsr_neg_coefs"], positive=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    
    
    

