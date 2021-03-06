"""
ame2016.py

The AME2016 atomic mass evaluation [1,2]_

See https://www-nds.iaea.org/amdc/ame2016/mass16.txt

References
----------
.. [1]  "The Ame2016 atomic mass evaluation (I)"
   by W.J.Huang, G.Audi, M.Wang, F.G.Kondev, S.Naimi and X.Xu.
   Chinese Physics C41 030002, March 2017.

.. [2] "The Ame2016 atomic mass evaluation (II)" 
   by M.Wang, G.Audi, F.G.Kondev, W.J.Huang, S.Naimi and X.Xu.
   Chinese Physics C41 030003, March 2017.

"""

# the atomic isotope dictionary
# the values are a tuple with format
# (atomic mass, uncertainty, spin multiplicity)
#
_masses = {
    "X":    (0, 0, 1),  # dummy atom
    
    "H":    (1007825.03224e-6, 0.00009e-6, 2),
    "H-1":  (1007825.03224e-6, 0.00009e-6, 2),
    "D":    (2014101.77811e-6, 0.00012e-6, 3),
    "H-2":  (2014101.77811e-6, 0.00012e-6, 3),
    "T":    (3016049.28199e-6, 0.00023e-6, 2),
    "H-3":  (3016049.28199e-6, 0.00023e-6, 2),
    
    "He-3": (3016029.32265e-6, 0.00022e-6, 2),
    "He":   (4002603.25413e-6, 0.00006e-6, 1),
    "He-4": (4002603.25413e-6, 0.00006e-6, 1),
    
    "Li-6": (6015122.88742e-6, 0.00155e-6, 3),
    "Li":   (7016003.43666e-6, 0.00454e-6, 4),
    "Li-7": (7016003.43666e-6, 0.00454e-6, 4),
    
    "Be":   (9012183.066e-6, 0.082e-6, 4),
    "Be-9": (9012183.066e-6, 0.082e-6, 4),
    
    "B-10": (10012936.862e-6, 0.016e-6, 7),
    "B":    (11009305.166e-6, 0.013e-6, 4),
    "B-11": (11009305.166e-6, 0.013e-6, 4),
    
    "C":    (12.0, 0.0, 1), # exact
    "C-12": (12.0, 0.0, 1), # exact
    "C-13": (13003354.83521e-6, 0.00023e-6, 2),
    "C-14": (14003241.98843e-6, 0.00403e-6, 1),
    
    "N":    (14003074.00446e-6, 0.00021e-6, 3),
    "N-14": (14003074.00446e-6, 0.00021e-6, 3),
    "N-15": (15000108.89894e-6, 0.00065e-6, 2),
    
    "O":    (15994914.61960e-6, 0.00017e-6, 1),
    "O-16": (15994914.61960e-6, 0.00017e-6, 1),
    "O-17": (16999131.75664e-6, 0.00070e-6, 6),
    "O-18": (17999159.61284e-6, 0.00076e-6, 1),
    
    "F":    (18998403.16288e-6, 0.00093e-6, 2),
    "F-19": (18998403.16288e-6, 0.00093e-6, 2),
    
    "Ne":   (19992440.17619e-6, 0.00168e-6, 1),
    "Ne-20":(19992440.17619e-6, 0.00168e-6, 1),
    "Ne-21":(20993846.685e-6, 0.041e-6, 4),
    "Ne-22":(21991385.109e-6, 0.018e-6, 1),
    
    "Na":   (22989769.28199e-6, 0.00194e-6, 4),
    "Na-23":(22989769.28199e-6, 0.00194e-6, 4),
    
    "Mg":   (23985041.697e-6, 0.014e-6, 1),
    "Mg-24":(23985041.697e-6, 0.014e-6, 1),
    "Mg-25":(24985836.964e-6, 0.050e-6, 6),
    "Mg-26":(25982592.971e-6, 0.032e-6, 1),
    
    "Al":   (26981538.408e-6, 0.050e-6, 6), 
    "Al-27":(26981538.408e-6, 0.050e-6, 6), 
    
    "Si":   (27976926.53499e-6, 0.00052e-6, 1),
    "Si-28":(27976926.53499e-6, 0.00052e-6, 1),
    "Si-29":(28976494.66525e-6, 0.00060e-6, 2),
    "Si-30":(29973770.136e-6,   0.023e-6, 1),
    
    "P":    (30973761.99863e-6, 0.00072e-6, 1),
    "P-31": (30973761.99863e-6, 0.00072e-6, 1),
    
    "S":    (31972071.17443e-6, 0.00141e-6, 1),
    "S-32": (31972071.17443e-6, 0.00141e-6, 1),
    "S-33": (32971458.90985e-6, 0.00145e-6, 4),
    "S-34": (33967867.012e-6, 0.047e-6, 1),
    "S-36": (35967080.699e-6, 0.201e-6, 1),
    
    "Cl":   (34968852.694e-6, 0.038e-6, 4),
    "Cl-35":(34968852.694e-6, 0.038e-6, 4),
    "Cl-37":(36965902.584e-6, 0.055e-6, 4),
    
    "Ar-36":(35967545.105e-6, 0.028e-6, 1),
    "Ar-38":(37962732.104e-6, 0.209e-6, 1),
    "Ar":   (39962383.12378e-6, 0.00240e-6, 1),
    "Ar-40":(39962383.12378e-6, 0.00240e-6, 1),
    }