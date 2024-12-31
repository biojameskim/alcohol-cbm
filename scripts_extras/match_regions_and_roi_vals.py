import pandas as pd

CONTROL_ONLY = False
MALE = False
FEMALE = True

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

SC_regions_info = pd.read_csv('data/regions_info/SC_regions_info.csv')
FC_regions_info = pd.read_csv('data/regions_info/FC_regions_info.csv')

aal_roi_vals = pd.read_csv(f'data/regions_info/aal_roi_vals/{file_name}{sex}_aal_roi_vals.csv')

# Concatenate the coefficients to the regions
SC_regions_info = pd.concat([SC_regions_info, aal_roi_vals[['SC_pos_coefs', 'SC_neg_coefs']]], axis=1)
FC_regions_info = pd.concat([FC_regions_info, aal_roi_vals[['FC_pos_coefs', 'FC_neg_coefs', 'FCgsr_pos_coefs', 'FCgsr_neg_coefs']]], axis=1)

SC_regions_info.to_csv(f'data/regions_info/aal_roi_vals/SC_regions_info_roi_vals_{file_name}{sex}.csv', index=False)
FC_regions_info.to_csv(f'data/regions_info/aal_roi_vals/FC_regions_info_roi_vals_{file_name}{sex}.csv', index=False)