class BSpline(object):
	def __init__(self,grid,dvalues):
		self.grid = grid
	def __call__(self):
		return dvalues	
	def plot(self):
