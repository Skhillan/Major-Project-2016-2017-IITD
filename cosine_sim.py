import numpy as np

def cosine_sim(vec1, vec2):
    if(len(vec1)==len(vec2)):
        dot_value = np.dot(vec1, vec2)
        vec1_mod = np.sqrt((vec1 * vec1).sum())
        vec2_mod = np.sqrt((vec2 * vec2).sum())
        cos_angle = dot_value / (vec1_mod * vec2_mod)  # cosine of angle between x and y
        return cos_angle
    else:
        print "Vector dimensions don't match"
        return