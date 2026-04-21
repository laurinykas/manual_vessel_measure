"""
pipeline/ - Esamo retinalinio vaizdo apdorojimo pipeline moduliai.

Pernešti iš pagrindinio projekto be esminių pakeitimų.
Naudojami od_detector.py ir exclusion_zones.py viduje.
"""

from .masking import createMask, propCoef, normingMask, norming, create_fundus_mask
from .preprocessing import preprocessing1, preprocessing2, preprocessing3
from .vessel_extraction import bwe1, thinning
from .optic_disc import detect_optic_disc