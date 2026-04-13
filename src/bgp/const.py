import numpy as np


SYNTHSEG_CODEMAP = {
    0:  "background",
    2:  "left_cerebral_white_matter", 
    3:  "left_cerebral_cortex",
    4:  "left_lateral_ventricle",
    5:  "left_inferior_lateral_ventricle",
    7:  "left_cerebellum_white_matter",
    8:  "left_cerebellum_cortex",
    10: "left_thalamus",
    11: "left_caudate",
    12: "left_putamen", 
    13: "left_pallidum", 
    14: "third_ventricle", 
    15: "fourth_ventricle", 
    16: "brain_stem", 
    17: "left_hippocampus",
    18: "left_amygdala",
    24: "csf", 
    26: "left_accumbens_area", 
    28: "left_ventral_dc", 
    41: "right_cerebral_white_matter",
    42: "right_cerebral_cortex",
    43: "right_lateral_ventricle",
    44: "right_inferior_lateral_ventricle",
    46: "right_cerebellum_white_matter", 
    47: "right_cerebellum_cortex", 
    49: "right_thalamus", 
    50: "right_caudate", 
    51: "right_putamen", 
    52: "right_pallidum", 
    53: "right_hippocampus",
    54: "right_amygdala",
    58: "right_accumbens_area", 
    60: "right_ventral_dc"
}


COARSE_REGIONS = [
    "accumbens_area",
    "amygdala",
    "brain_stem",
    "caudate",
    "cerebellum_cortex",
    "cerebellum_white_matter",
    "cerebral_cortex",
    "cerebral_white_matter",
    "csf",
    "fourth_ventricle",
    "hippocampus",
    "inferior_lateral_ventricle",
    "lateral_ventricle",
    "pallidum",
    "putamen",
    "thalamus",
    "third_ventricle",
    "ventral_dc",
]

CONDITIONING_VARIABLES = [
    "age", 
    "sex",
    # "diagnosis",
    "cerebral_cortex", 
    # "hippocampus", 
    # "amygdala",
    "cerebral_white_matter",
    "lateral_ventricle", 
]

# chosen resolution
RESOLUTION = 1.3                             

# shape of the MNI152 (1mm^3) template
# INPUT_SHAPE_1mm = (182, 218, 182)   
# for ANTS 0-8
INPUT_SHAPE_1mm = (135, 177, 138)   
# for MNI 4.5-8.5
# INPUT_SHAPE_1mm = (197, 233, 189)   

# resampling the MNI152 to (1.5mm^3)
# INPUT_SHAPE_1p5mm = (122, 146, 122) 
# resampling the template to (1.3mm^3) for ANTS 0-8
# INPUT_SHAPE_1p5mm = (104, 136, 106)   
# INPUT_SHAPE_1p5mm = (104, 136, 106) 
# resampling the template to (1.3mm^3) for MNI 4.5-8.5
# INPUT_SHAPE_1p5mm = (152, 179, 146)   

# Adjusting the dimensions to be divisible by 8 (2^3 where 3 are the downsampling layers of the AE)
# INPUT_SHAPE_AE = (120, 144, 120)  
# INPUT_SHAPE_AE = (112, 144, 112)
INPUT_SHAPE_AE = (120, 144, 120)    

# Latent shape of the autoencoder 
LATENT_SHAPE_AE = (3, 15, 18, 15) 

# Adjusting the latent space (with constant padding) to be divisible by 4 (2^2 where 2 are the downsampling layers of U-Net)
LATENT_SHAPE_DM = (3, 16, 20, 16)   

# Affine matrix for ANTS 0-8 resampled to 1.3mm^3
MNI152_1P5MM_AFFINE = np.array([         
    [ -1.3, 0,    0,    65   ],
    [ 0,    1.3,  0,    -105 ],
    [ 0,    0,    1.3,  -56  ],
    [ 0,    0,     0,   1    ]
])

AGE_MIN, AGE_MAX, AGE_DELTA = 1, 7, 6
SEX_MIN, SEX_MAX, SEX_DELTA = 1, 2, 1