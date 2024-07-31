# this file contains collections of proxes we learned in the class
import numpy as np
from scipy.optimize import bisect


def sillyFunction(t,k,z,O):
	a = z-t*O
	O2 = O.T
	ans = -k-(t*len(z))+O2.dot(z)
	for i in range(0,len(z)):
		if z[i][0] - t > 1:
			ans = ans + 1 - z[i][0] +t
		elif z[i][0] - t < 0:
			ans = ans - z[i][0] + t


	return ans

# =============================================================================
# TODO Complete the following prox for simplex
# =============================================================================

# Prox of capped simplex
# -----------------------------------------------------------------------------
def prox_csimplex(z, k):
	"""
	Prox of capped simplex
		argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

	input
	-----
	z : arraylike
		reference point
	k : int
		positive number between 0 and z.size, denote simplex cap

	output
	------
	x : arraylike
		projection of z onto the k-capped simplex
	"""
	# safe guard for k
	assert 0<=k<=z.size, 'k: k must be between 0 and dimension of the input.'

	# TODO do the computation here
	# Hint: 1. construct the scalar dual object and use `bisect` to solve it.
	#		2. obtain primal variable from optimal dual solution and return it.
	#
	O = np.ones((len(z),1))
	O2 = O.T
	z2 = z.reshape(len(z),1)

    
	f = lambda x: sillyFunction(x,k,z2,O)
    
	root = bisect(f,-10000,10000)
    
    
	proj = z2-root*O
	for i in range(0,len(z)):
		if proj[i][0] > 1:
			proj[i][0] = 1
		elif proj[i][0] < 0:
			proj[i][0] = 0
    


	proj = proj.reshape(len(proj))
    
    
	return proj

