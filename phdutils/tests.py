def compute_T_LSS(B, test_type, Y):
    C_hats = compute_C_hats(B=B, Y=Y)
    M = Y.shape[0]

    if test_type=='logdet':
        Tns = [np.product(np.linalg.slogdet(C_hat)) for C_hat in C_hats]
        Tns = -np.real(Tns)

    elif test_type=='frobenius':
        Tns = [np.linalg.norm(C_hat-np.identity(M))**2 for C_hat in C_hats]

    else:
        raise ValueError('test_type not recognised: %s' % test_type)

    #renormalisation
    Tns = np.array(Tns)/M

    indice_max = np.argmax(np.abs(Tns - equivalent_deterministe(c=M/(B+1), test_type=test_type)))
    return Tns[indice_max]
