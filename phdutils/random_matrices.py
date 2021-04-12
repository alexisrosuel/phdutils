import scipy.stats
import numpy as np

from .entries import complex_gaussian



def equivalent_deterministe(c, test_type):
    """
    dans le cas du test FNT:
    équivalent deterministe de ||S_cor(\nu) - I_M||_F^2
    ie Tr((S_cor(\nu) - I_M)(S_cor(\nu) - I_M)^*)
    = M * [(1+c) -2 +1] = M*c
    """
    if test_type=='logdet':
        return -((c-1)/c*np.log(1-c)-1)
    elif test_type=='frobenius':
        return c


def marcenko_pastur_pdf(x,c):
    lambda_plus = (1+np.sqrt(c))**2
    lambda_moins = (1-np.sqrt(c))**2

    result = np.zeros(x.shape[0])

    valid_index = np.logical_and(x>lambda_moins,x<lambda_plus)
    x_valid = x[valid_index]
    result[valid_index] = np.sqrt((x_valid-lambda_moins)*(lambda_plus-x_valid)) / (2*np.pi*x_valid*c)

    return result


def marcenko_pastur_cdf(x, c):
    """
    from https://mathoverflow.net/questions/247448/cumulative-integral-of-the-marchenko-pastur-density-for-wishart-eigenvalues
    works only for c<1
    """

    lambda_plus = (1+np.sqrt(c))**2
    lambda_moins = (1-np.sqrt(c))**2

    valid_index = np.logical_and(x>lambda_moins,x<lambda_plus)
    x_valid = x[valid_index]

    ### Computation of the three terms, note that arcsin is taken [0,pi] instead of [-pi/2,pi/2]
    t1 = np.sqrt((x_valid-lambda_moins)*(lambda_plus-x_valid))

    temp_2 = (2*x_valid - lambda_plus - lambda_moins)/(lambda_plus-lambda_moins)
    t2 = np.arcsin(temp_2) + np.pi/2

    temp_3 = ((lambda_plus+lambda_moins)*x_valid - 2*lambda_plus*lambda_moins)/(x_valid*(lambda_plus-lambda_moins))
    t3 = 1/np.sqrt(lambda_plus*lambda_moins)*(np.arcsin(temp_3)+np.pi/2)

    integral = t1 + (lambda_plus+lambda_moins)/2 * t2 - lambda_plus*lambda_moins * t3
    ###

    res = np.zeros(x.shape[0])
    res[valid_index] = 1/(2*np.pi*c)*integral
    res[x>=lambda_plus] = 1

    return res





def phi(gamma, c):
    """
    Maps true eigenvalue to observed eigenvalue in a spike model
    """
    result = (1+gamma)*(gamma+c)/gamma
    indices_in_support = (gamma<np.sqrt(c)) and (gamma>-np.sqrt(c))
    result[indices_in_support] = (1+np.sqrt(c))**2
    return result




def t_mp(z, c):
    """
    Stieltjes transform of the MP distribution
    """
    if c==0:
        return 1/(z-1)

    lambda_plus, lambda_moins = (1+np.sqrt(c))**2, (1-np.sqrt(c))**2,
    produit = (z-lambda_plus)*(z-lambda_moins)
    angle = np.angle(produit)
    if angle <0:
        angle = 2*np.pi+angle
    sqrt_produit = np.sqrt(np.absolute(produit))*np.exp(1j*angle/2)
    numerateur = -(z+(c-1)) + sqrt_produit
    denominateur = 2*z*c
    return numerateur / denominateur


def t_mp_tilde(z,c):
    return -1/(z*(1+c*t_mp(z, c)))
