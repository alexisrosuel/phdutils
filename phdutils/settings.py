try:
    del M, B, c
except:
    pass

def get_c():
    return 0.8

def get_alpha():
    return 0.75

def get_beta():
    return 0.75

def get_B(N):
    B_candidat = int(N**get_alpha())
    if B_candidat % 2 == 1:
        return B_candidat + 1
    return B_candidat

def get_M(N):
    return int(get_c()*get_B(N))
