import os
import numpy as np
from nibabel.testing import data_path
example_filename = os.path.join(data_path, 'example4d.nii.gz')


import nibabel as nib
img = nib.load(example_filename)



# -- end code --