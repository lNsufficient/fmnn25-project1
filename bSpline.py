import scipy
import scipy.linalg
import numpy
import matplotlib.pyplot as plt
class BSpline(object):
    def __init__(self,dvalues,grid=None):
        #Grid should be same length as dvalues
        if(grid is None):
            grid = numpy.linspace(0,1,numpy.size(dvalues,1)-2)
            self.grid = numpy.insert(grid,[0,0,-1,-1],[grid[0],grid[0],grid[-1],grid[-1]])
        else:
            self.grid = grid #these are the u values, such that grid[i] = u_i
        self.d = dvalues #these are the corresponding d values, so that dvalues[i] = d_i, 

    def __call__(self):
        self.plot()

    def findHotInterval(self, t):
        return (self.grid > t).argmax()-1
    
    @classmethod
    def basisFunction(cls,u,j):
        return lambda x: cls.N(u,3,j,x)
    @classmethod
    def N(cls, u, k, j, x):
        L = numpy.size(u) - 1 #If we ever run into problem, the first thing to do is uncomment this line, because it really can mess up things.
        if k == 0:
            return (x>=u[j-1] and x<u[j])
        else: 
            #return lambda x: cls.alpha(u, u[(j+k-1)%L], u[(j-1)%L])(x)*BSpline.basisFunction(u, (k-1), j%L)(x) + cls.alpha(u, u[j%L], u[(j+k)%L])(x)*BSpline.basisFunction(u, k-1, (j+1)%L)(x)
            return cls.alpha(u, (j+k-1), (j-1),x)*cls.N(u, (k-1), j, x) + cls.alpha(u, j, (j+k),x)*cls.N(u, k-1, (j+1),x)
            
            #return lambda x: (x-u[j-1])/(u[j+k-1] - u[j-1])*cls.basisFunction(u,k-1,j)(x) + (u[j+k]-x)/(u[j+k]-u[j])*cls.basisFunction(u,k-1,j+1)(x)


    def findS(self, t):
        i = self.findHotInterval(t)
        #intressanta d: dvalues[i-2:i+1]
        second1 = self.theD(self.d[:,i-2],self.d[:,i-1], i-2, i+1, t)
        second2 = self.theD(self.d[:,i-1],self.d[:,i], i-1, i+2, t)
        second3 = self.theD(self.d[:,i], self.d[:,i+1], i, i+3, t)
        last1 = self.theD(second1, second2, i-1, i+1, t)
        last2 = self.theD(second2, second3, i, i+2, t)
        return self.theD(last1, last2, i, i +1, t)

    def theD(self, d1, d2, i_l, i_r, t):
        alpha = self.alpha(self.grid, i_l, i_r,t)
        return (alpha*d1 + (1-alpha)*d2)

    @classmethod
    def alpha(cls, u, i_l, i_r, x):
        #print(i_r)
        #print(i_l)
        #if (self.grid[i_r] == self.grid[i_l]):
            #if ((self.grid[i_r] - t) == 0):
                #this case should be taken care of by the self.grid[i_r]!=x inside lambda x: that is returned. 
            #    pass
            #else:
                #serious problems
            #    pass
        if(i_r >= len(u) or i_l >= len(u)):
            return 0
        if (u[i_r]==u[i_l]):
            return 0
        return (u[i_r] - x)/(u[i_r] - u[i_l])

    def plot(self):
        points = 1000
        xy = numpy.zeros((2, points-1))
        for j, t in enumerate(numpy.linspace(self.grid[0], self.grid[-1],points)[0:-1]):
            xy[:,j] = self.findS(t)
        plt.plot(xy[0,:],xy[1,:])
        plt.plot(self.d[0,:],self.d[1,:])
        plt.scatter(self.d[0,:],self.d[1,:])

        plt.show()
    
    @classmethod
    def interpolation(cls, grid, xy):
        # Skapa xi
        tmpgrid = grid
        #grid = grid[1:-1]
        xi  = numpy.array((grid[:-2]+grid[1:-1]+grid[2:])/3)
        grid = tmpgrid
        L = numpy.size(xi)
        # utvärdera splinevärdena på alla xchi (men det kanske inte behövs med alla xchi)?
        #print(xi)
        #print(xi[1])
        #print(BSpline.basisFunction(grid, 0, 2)(xi[1]))
        #print(BSpline.basisFunction(grid, 3, 2)(xi[0])) 
        N = numpy.array([[BSpline.basisFunction(grid, j+1)(xi[i]) for i in range(0,L)] for j in range(0,L)]).T
        # skapa ekvationssystemet
        print(numpy.shape(N))
        numpy.set_printoptions(precision=3)
        print(N)
        print(xy)
        dx = scipy.linalg.solve(N, xy[:,0])  #detta bör bytas mot solve_banded när vi fattar hur saker funkar. 	
        dy = scipy.linalg.solve(N, xy[:,1])  #detta också.
        d = numpy.array([dx,dy])
        return [xi,d]
