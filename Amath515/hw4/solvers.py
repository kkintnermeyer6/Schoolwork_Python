# this file contains collection of solver we learned in the class
from numpy.linalg import norm
from numpy.linalg import solve
import numpy as np

# =============================================================================
# TODO Complete the following optimization algorithms
#	* simple interior point method
#   * Chambolle-Pock method
#	* accelerated gradient descent
#	* accelerated proximal gradient descent
# =============================================================================

# interior point method
# -----------------------------------------------------------------------------
def optimizeWithIP(x0, A, b, C, d, mu=1.0, rate=0.1, tol=1e-6, max_iter=1000):
    """
    Optimize with interior point method
    for quadratic over box problem
        min_x 1/2||Ax - b||^2 s.t. Cx <= d

    input
    -----
    x0 : array_like
        Starting point for the solver
    A : array_like
        Problem input
    b : array_like
        Problem input
    C : array_like
        Problem input
    d : array_like
        Problem input
    mu : float, optional
        initial relax parameter mu
    rate : float, optional
        shrinkage rate of mu
    tol : float, optional
        KKT tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.

    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Primal objective function value convergence history
    err_his : array_like
        Norm of KKT system convergence history
    exit_flag : int
        0, norm of KKT below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # check all the input
    n = x0.size
    m = A.shape[0]
    k = C.shape[0]
    #
    assert A.shape[1]==n, 'A: number of column is wrong.'
    assert C.shape[1]==n, 'C: number of column is wrong.'
    assert b.size==m, 'b: size of b must be m.'
    assert d.size==k, 'd: size of d must be k.'
    #
    # setup the iterations
    x = x0.copy()
    r = b - A.dot(x)
    s = d - C.dot(x)
    v = mu/s
    z = np.hstack((x, v))
    # id of primal and dual variable, easy access
    id_x = slice(n)
    id_v = slice(n, n+k)
    # initialize KKT system and its Jacobian
    F = np.zeros(n+k)
    # temp = -A.T.dot(b2-A.dot(x2))+C.T.dot(v)
    F[id_x] = np.matmul(-A.T,b-np.matmul(A,x))+np.matmul(C.T,v)
    # temp = v.T.dot(d2-C.dot(x2)) - mu*np.ones((k,1))
    F[id_v] = v*(d-np.matmul(C,x))-mu*np.ones(k)
    # *************************************************************************
    # TODO: Fill in F -- the optimality conditions for primal and dual variables. 
    # 
    # Notice that in the lecture we defined F to be a function of three arguments:
    # F(s, z, x). There are two notation changes here comparing to the class.
    # First, in this code x is x (as in class), but z = v (dual variable is called v, not z).
    # 
    # Second, we notice that we can substitute s = d - Cx everywhere, 
    # so the Lagrangian, F, and dF are now functions of just two arguments -- x and v: 
    # L(x, v) = 1/2*||b - Ax||^2 - v^T(d - Cx)
    # with two constraints:
    # v >= 0 and d - Cx >= 0. 
    # Differentiating the Lagrangian once wrt x we get F[id_x]. The second part, F[id_v], comes from 
    # relaxing the slackness conditions: (d-Cx)_i*v_i = mu, written in a vector form.
    #
    # All of these are just differences in notation, overall we do exactly the same as Sasha did in class.
    #
    # *************************************************************************
    dF = np.zeros((n+k, n+k))
    dF[id_x,id_x] = np.matmul(A.T,A)
    dF[id_x,id_v] = C.T
    # dF[id_v,id_x] = -v.T.dot(C)
    dF[id_v,id_x] = -np.matmul(np.diag(v),C)
    # dF[id_v,id_v] = np.diag(d-C.dot(x2))
    dF[id_v,id_v] = np.diag(s)
    # *************************************************************************
    # TODO: Fill in dF
    # Hint: dF is a block-derivative of F(x, v) with respect to x and v:
    # dF = [d(F[id_x])/dx, d(F[id_x])/dv; d(F[id_v])/dx, d(F[id_v])/dv]
    # *************************************************************************
    #
    # record the primal objective and error measure
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    #
    iter_count = 0
    err = np.linalg.norm(F)
    while err >= tol:
        # update direction
        dz = np.linalg.solve(dF, -F)
        dx = dz[id_x]
        dv = dz[id_v]
        # safe guard on step size alpha
        alpha = 1.0
        # s+ has to be positive
        # *********************************************************************
        # TODO: Adjust alpha such that d - C(x + alpha dx) > 0
        # (In Sahsa's notation from class we want to make sure here that s > 0).
        # *********************************************************************
        temp = d-np.matmul(C,x)
        temp2 = np.matmul(C,dx)
        ind = np.where(-temp2 < 0.0)[0]
        alpha = min(alpha, 0.99*np.min(temp[ind]/temp2[ind]))
        # v+ has to be positive
        # Adjust alpha such that v + alpha dv > 0 (done for you)
        ind = np.where(dv < 0.0)[0]
        alpha = min(alpha, 0.99*np.min(-v[ind]/dv[ind]))
        #
        # update variable
        x += alpha*dx
        v += alpha*dv
        #
        r = b - A.dot(x)
        s = d - C.dot(x)
        z = np.hstack((x, v))
        #
        # update mu
        mu = rate*np.mean(v*s)
        #
        # update F and dF
        # *********************************************************************
        # TODO: update F and dF
        # Hint: Do not need to update all parts of dF
        # F[id_x] = -A.T.dot(b2-A.dot(x2))+C.T.dot(v)
        F[id_x] = np.matmul(-A.T,b-np.matmul(A,x))+np.matmul(C.T,v)
        # F[id_v] = (d-C.dot(x2)).dot(v) - mu*np.ones((k,1))
        F[id_v] = v*(d-np.matmul(C,x))-mu*np.ones(k)
        
        # dF[id_v,id_x] = -v.T.dot(C)
        dF[id_v,id_x] = -np.matmul(np.diag(v),C)
        # dF[id_v,id_v] = np.diag(d-C.dot(x2))
        dF[id_v,id_v] = np.diag(s)
        # *********************************************************************
        #
        obj = 0.5*np.sum(r**2)
        err = np.linalg.norm(F)
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        iter_count += 1
        if iter_count >= max_iter:
            print('Interior point method reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0

# chambolle-pock method
# -----------------------------------------------------------------------------
def optimizeWithCP(x0, A, b, c, h, k, prox_ch, prox_k,
    tol=1e-6, max_iter=1000):
    """
    Optimize with Chambolle-Pock Algorithm
        min_x <c, x> + h(b - Ax) + k(x)
    where h and k are closed and convex.

    input
    -----
    x0 : array_like
        Starting point for the solver
    A : array_like
        Problem data input
    b : array_like
        Problem data input
    c : array_like
        Problem data input
    h : function
        objective function
    k : function
        objective function (regularizer)
    prox_ch : function
        Input x and return prox of conjugate of h
    prox_k : function
        Input x and return prox of k
    tol : float, optional
        Optimiality tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.

    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        optimality condition convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # check all the input
    n = x0.size
    m = A.shape[0]
    #
    assert A.shape[1]==n, 'A: number of column is wrong.'
    assert b.size==m, 'b: size of b must be same with num of rows of A.'
    assert c.size==n, 'c: size of c must be same with x.'
    #
    # setup the iterations
    x = x0.copy()
    v = np.ones(m)
    rx = b - A.dot(x)
    rv = c - A.T.dot(v)
    #
    alpha = 1.0/np.linalg.norm(A, 2)
    # record the primal objective and error measure
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    #
    err = tol + 1.0
    iter_count = 0
    #
    while err >= tol:
        # In the lectures Sasha derived the following iteration for Chambolle-Pock:
        # x' = prox_{γk}(x - γA^Tz)
        # z' = prox_{γh*}(z + γA(x - 2x'))
        # As in IP method, we rename z to v here (historical reasons, don't ask), 
        # and using the "linear shift" lemma from lectures we get the following form of CP iterationx
        # x' = prox_{γk}(x + γA^Tv - γc)
        # v' = prox_{γh*}(v + γA(x-2x') + γb)
        # If we denote the residuals for x and v respectively as rx and rv (see above), we'll see that the form
        # for the iterations can be simplified even further:
        # x' = prox_{γk}(x - γ*rv)
        # v' = prox_{γh*}(v + γ(2rx' - rx))
        #
        # update x
        # *********************************************************************
        # TODO: update x and rx
        x_new = prox_k(x-alpha*(rv),alpha)
        rx_new = b - np.matmul(A,x_new)
        # *********************************************************************
        # update v
        # *********************************************************************
        # TODO: update x and rx
        v_new = prox_ch(v+alpha*(2*rx_new-rx),alpha)
        rv_new = c-np.matmul(A.T,v_new)
        # *********************************************************************
        # update convergence result
        obj = c.dot(x_new) + h(rx_new) + k(x_new)
        err = np.linalg.norm(x - x_new)/alpha
        #
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        # update variables
        np.copyto(x, x_new)
        np.copyto(v, v_new)
        np.copyto(rx, rx_new)
        np.copyto(rv, rv_new)
        #
        iter_count += 1
        if iter_count >= max_iter:
            print('Chambolle-Pock reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0

# =============================================================================
# From previous homeworks:
#   * gradient descent
#   * Newton's method
#   * proximal gradient desecnt
#   * accelerated gradient descent
#   * accelerated proximal gradient descent
# =============================================================================

# Proximal gradient descent
# -----------------------------------------------------------------------------
def optimizeWithPGD(x0, func_f, func_g, grad_f, prox_g, beta_f, tol=1e-6, max_iter=1000):
    """
    Optimize with Proximal Gradient Descent Method
        min_x f(x) + g(x)
    where f is beta smooth and g is proxiable.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func_f : function
        Input x and return the function value of f
    func_g : function
        Input x and return the function value of g
    grad_f : function
        Input x and return the gradient of f
    prox_g : function
        Input x and a constant float number and return the prox solution
    beta_f : float
        beta smoothness constant for f
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    g = grad_f(x)
    #
    step_size = 1.0/beta_f
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        # proximal gradient descent step
        x_new = prox_g(x - step_size*g, step_size)
        #
        # update information
        obj = func_f(x_new) + func_g(x_new)
        err = norm(x - x_new)/step_size
        #
        np.copyto(x, x_new)
        g = grad_f(x)
        #
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0

# Accelerated gradient descent
# -----------------------------------------------------------------------------
def optimizeWithAGD(x0, func, grad, beta, tol=1e-6, max_iter=1000):
    """
    Optimize with Accelerated Gradient Descent Method
        min_x f(x)
    where f is beta smooth.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func : function
        Input x and return the function value
    grad : function
        Input x and return the gradient
    beta : float
        beta smoothness constant for the function
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    y = x0.copy()
    g = grad(y)
    t = 1.0
    #
    step_size = 1.0/beta
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter+1)
    err_his = np.zeros(max_iter+1)
    #
    obj_his[0] = func(x)
    err_his[0] = norm(g)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        # proximal gradient descent step
        x_new = y - step_size*g
        t_new = 0.5*(1.0 + np.sqrt(1.0 + 4.0*t**2))
        y_new = x_new + (t - 1.0)/t_new*(x_new - x)
        #
        # update information
        np.copyto(x, x_new)
        np.copyto(y, y_new)
        t = t_new
        g = grad(y)
        #
        obj = func(x_new)
        err = norm(g)
        #
        obj_his[iter_count + 1] = obj
        err_his[iter_count + 1] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0

# Accelerated proximal gradient descent
# -----------------------------------------------------------------------------
def optimizeWithAPGD(x0, func_f, func_g, grad_f, prox_g, beta_f, tol=1e-6, max_iter=1000):
    """
    Optimize with Accelerated Proximal Gradient Descent Method
        min_x f(x) + g(x)
    where f is beta smooth and g is proxiable.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func_f : function
        Input x and return the function value of f
    func_g : function
        Input x and return the function value of g
    grad_f : function
        Input x and return the gradient of f
    prox_g : function
        Input x and a constant float number and return the prox solution
    beta_f : float
        beta smoothness constant for f
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    y = x0.copy()
    g = grad_f(y)
    t = 1.0
    #
    step_size = 1.0/beta_f
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        # proximal gradient descent step
        x_new = prox_g(y - step_size*g, step_size)
        t_new = 0.5*(1.0 + np.sqrt(1.0 + 4.0*t**2))
        y_new = x_new + (t - 1.0)/t_new*(x_new - x)
        #
        # update information
        obj = func_f(x_new) + func_g(x_new)
        err = norm(x - x_new)
        #
        np.copyto(x, x_new)
        np.copyto(y, y_new)
        t = t_new
        g = grad_f(y)
        #
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0

# Gradient descent
# -----------------------------------------------------------------------------
def optimizeWithGD(x0, func, grad, beta, tol=1e-6, max_iter=1000):
    """
    Optimize with Gradient Descent
    	min_x f(x)
    where f is beta smooth.

    input
    -----
    x0 : array_like
        Starting point for the solver.
    func : function
        Input x and return the function value.
    grad : function
        Input x and return the gradient.
    beta : float
        beta smoothness constant
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = np.copy(x0)
    g = grad(x)
    step_size = 1.0/beta
    #
    obj = func(x)
    err = norm(g)
    #
    obj_his = np.zeros(max_iter + 1)
    err_his = np.zeros(max_iter + 1)
    #
    obj_his[0] = obj
    err_his[0] = err
    
    # start iterations
    iter_count = 0
    while err >= tol:
        # gradient descent step
        x -= step_size*g
        #
        # update function and gradient
        g = grad(x)
        #
        obj = func(x)
        err = norm(g)
        #
        iter_count += 1
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        if iter_count >= max_iter:
            print('Gradient descent reach maximum number of iteration.')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0

# Newton's Method
# -----------------------------------------------------------------------------
def optimizeWithNT(x0, func, grad, hess, tol=1e-6, max_iter=100):
    """
    Optimize with Newton's Method
    
    input
    -----
    x0 : array_like
        Starting point for the solver.
    func : function
        Input x and return the function value.
    grad : function
        Input x and return the gradient.
    hess : function
        Input x and return the Hessian matrix.
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial step
    x = np.copy(x0)
    g = grad(x)
    H = hess(x)
    #
    obj = func(x)
    err = norm(g)
    #
    obj_his = np.zeros(max_iter + 1)
    err_his = np.zeros(max_iter + 1)
    #
    obj_his[0] = obj
    err_his[0] = err
    
    # start iteration
    iter_count = 0
    while err >= tol:
        # Newton's step
        x -= solve(H, g)
        #
        # update function, gradient and Hessian
        g = grad(x)
        H = hess(x)
        #
        obj = func(x)
        err = norm(g)
        #
        iter_count += 1
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        if iter_count >= max_iter:
            print('Gradient descent reach maximum number of iteration.')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0