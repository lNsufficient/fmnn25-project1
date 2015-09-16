import scipy
import numpy
class BSpline(object):
	def __init__(self,grid,dvalues):
		self.grid = grid #these are the u values, such that grid[i] = u_i
		self.dvalues = dvalues #these are the corresponding d values, so that dvalues[i] = d_i, 
		
	def __call__(self):
		return dvalues	

	def findHotInterval(self, t):
		return (self.grid > t).argmax()-1

        def basisFunction(u, k, j):
            if k == 0:
                return lambda x: ((x>=u[j-1])-(x>=u[j]))
            else: 
                return lambda x: (x-u[j-1])/(u[j+k-1] - u[j-1])*basisFunction(u, k-1, j)(x) + (u[j+k] - x)/(u[i+k]-u[i])*basisFunction(u, k-1, j+1)(x)

	def findS(self, i, t):
		#intressanta d: dvalues[i-2:i+1]
                second1 = theD(d[i-2],d[i-1], i-2, i+1, t)
                second2 = theD(d[i-1],d[i], i-1, i+2, t)
                second3 = theD(d[i], d[i+1], i, i+3, t)
                last1 = theD(second1, second2, i-1, i+1, t)
                last2 = theD(second2, second3, i, i+2, t)
		return theD(last1, last2, i, i +1, t)	

	def theD(self, d1, d2, i_l, i_r, t):
		return (alpha(i_l, i_r, t)*d1 + (1-alpha(i_l, i_r, t))*d2)

	def alpha(self, i_l, i_r, t):
            return (self.grid[i_r] - t)/(self.grid[i_r] - self.grid[i_l])
	
	def plot(self):
		points = 1000
		j = 0
		xy = scipy.zeros(2, points)
		for t in linspace(grid[0], grid[-1],points):
			i = findHotInterval(t)
			xy[:,j] = findS(i, t)
			j+=1
		plot(x, y)
		

