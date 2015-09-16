class BSpline(object):
	def __init__(self,grid,dvalues):
		self.grid = grid #these are the u values, such that grid[i] = u_i
		self.dvalues = dvalues #these are the corresponding d values, so that dvalues[i] = d_i, 
		
	def __call__(self):
		return dvalues	

	def findHotInterval(self, t):
		return (self.grid > t).argmax()-1
	
	def findS(self, i, t):
		#intressanta d: dvalues[i-2:i+1]
				

		return 	
	def theD(self, i1, i2):
		

	def alpha(self, i_l, i_r, t):
		return (u[i_r] - t)/(u[i_r] - u[i_l])
	
	def plot(self):
		points = 1000
		j = 0
		xy = scipy.zeros(2, points)
		for t in linspace(grid[0], grid[-1],points):
			i = findHotInterval(t)
			xy[:,j] = findS(i, t)
			j+=1
		plot(x, y)
		

