import sys
sys.path.append('../src/')
import weighting 
import numpy as np
import pandas as pd

if __name__ == "__main__":
    (m1, q, z, a1, a2, cos_tilt1, cos_tilt2, 
    pdraw, ndraw) = weighting.extract_selection_samples(
        '../../GW_2025/search_sensitivity/endo3_bbhpop-LIGO-T2100113-v12.hdf5',
        nsamp=None, desired_pop_wt=None)

    df = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'a1': a1, 
                    'a2':a2, 'cos_tilt_1': cos_tilt1, 'cos_tilt_2': cos_tilt2, 
                    'pdraw_m1sqz': pdraw, 'ndraw': ndraw}) #m1 is source frame

    df['dm1sz_dm1ddl'] = weighting.dm1sz_dm1ddl(df['z'])
    df.to_hdf('./selection_samples.h5', 'samples')


