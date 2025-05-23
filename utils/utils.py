import numpy as np

period_boundaries = [
    -500,  # Ancient Literature starts (before 500 CE)
    500,   # Medieval Literature starts
    1450,  # Renaissance starts
    1660,  # Enlightenment/Neoclassical starts
    1790,  # Romantic Period starts
    1850,  # Victorian/Realist starts
    1900,  # Modernist starts
    1945,  # Post-War/Mid-Century starts
    1970,  # Postmodern starts
    2000,  # Contemporary starts
    float('inf')  # End boundary
]
        
def get_period_index(year):
    for i in range(len(period_boundaries) - 1):
        if period_boundaries[i] <= year < period_boundaries[i + 1]:
            return period_boundaries[i]
    
    return len(period_boundaries) - 2  # Should never happen with infinity boundary