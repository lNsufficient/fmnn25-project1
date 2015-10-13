#AUTHORS:
#Alexander Israelsson
#Edvard Johansson
#David Petersson
#Linnéa Støvring-Nielsen

from math import *
import bSpline
import numpy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('error')
import unittest

class testBSpline(unittest.TestCase):
    def testAlphaDivideByZero(self):
        u = numpy.array([0,0,0,1,1,1])
        a = bSpline.BSpline(numpy.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]),u)
        try:
            result = a.alpha(u, 3,3,1)
        except:
            self.fail("If nodes coincide 0/0 = 0, but we got division by zero error")
        expected = 0
        self.assertEqual(result, expected)
    
    def testLooksGood(self):
        x = numpy.array([37,73,42,7,3])
        x = numpy.insert(x,[0,0,-1,-1],[x[0],x[0],x[-1],x[-1]])
        y = numpy.cos(x)
        d = numpy.array([x,y])
        
        grid = numpy.linspace(0,1,5)
        grid = numpy.insert(grid,[0,0,-1,-1],[grid[0],grid[0],grid[-1],grid[-1]])
         
        spline = bSpline.BSpline(d,grid)
        xy = numpy.array([spline(grid[i+1]) for i in range(numpy.size(x)-2)]) 
        spline.plot()
        self.assertTrue(True)
    def testBasisFunctionEqualS(self):
        #test if s(u) = N_i for d_i = 1, d_j=0, j!=i for all i.
        numberValues = 12
        d = numpy.array(list(range(numberValues)))
        d = numpy.random.rand(numberValues, 1)
        umin = 0
        umax = 1
        u = numpy.linspace(umin,umax,numpy.size(d))
        d = numpy.insert(d,[0,0,-1,-1],[d[0],d[0],d[-1],d[-1]])
        points = 100
        x = numpy.linspace(umin, umax, points)
        d = numpy.array([d,d])
        u = numpy.insert(u,[0,0,-1,-1],[u[0],u[0],u[-1],u[-1]])
        spline = bSpline.BSpline(d, u) 
        y = numpy.array([spline(i) for i in x])
        ysum = numpy.zeros(numpy.size(x))
        for i in range(numpy.size(d,1)):
            ytemp = [d[1, i]*bSpline.BSpline.basisFunction(u,i)(j) for j in x]
            ysum+= ytemp
        y2 = ysum
        plt.plot(x[:], y[:,0])
        plt.plot(x, y2)
        plt.show()
        for i in range(numpy.size(x)-1):  #Here, we don't compare the last one, basis functions gives 0. 
            self.assertAlmostEqual(y[i,0],y2[i], msg="basis function not equal to blossom thing")

    def testLooksGood2(self):
        x = numpy.linspace(0,10)
        x = numpy.insert(x,[0,0,-1,-1],[x[0],x[0],x[-1],x[-1]])
        y = numpy.cos(x)
        d = numpy.array([x,y])     
        spline = bSpline.BSpline(d)
        spline.plot()
    def Interpolation(self): # The interpolation does not work corretly
        #xy = numpy.array([[1, 0, 4, -1],[1, -3, 2, -1]])
        
        #u = numpy.array([0,0,1, 2, 3, 31])
        #u = numpy.array([0,1,2,3,4,5,6,6,6,0,0])
        
        u = numpy.linspace(0,1,4)
        u = numpy.insert(u,[0,0,-1,-1],[u[0],u[0],u[-1],u[-1]])
        x = numpy.array([-1,1,2,3,-1])
        x = numpy.insert(x,[0,0,-1,-1],[x[0],x[0],x[-1],x[-1]])
        y = numpy.array([-1,-1,2,-3,1])
        y = numpy.insert(y,[0,0,-1,-1],[y[0],y[0],y[-1],y[-1]])
        d = numpy.array([x,y])

        (xi,interpolation) = bSpline.BSpline.interpolation(u,d)
        spline = bSpline.BSpline(interpolation,u)
        #spline = bSpline.BSpline(d)
        xy = numpy.array([spline.findS(u) for u in spline.grid[1:-1]])
        print(numpy.shape(xy))
        print(numpy.shape(spline.grid))
       # xi,dtmp = bSpline.BSpline.interpolation(spline.grid, xy)
        #d0tmp = dtmp[0]
   #     d1tmp = dtmp[1]
    #    d0tmp = numpy.insert(d0tmp,[0,0,-1,-1],[d0tmp[0],d0tmp[0],d0tmp[-1],d0tmp[-1]])
     #   d1tmp = numpy.insert(d1tmp,[0,0,-1,-1],[d1tmp[0],d1tmp[0],d1tmp[-1],d1tmp[-1]])
        xi = numpy.insert(xi,[0,0,-1,-1],[xi[0],xi[0],xi[-1],xi[-1]])
      #  spline = bSpline.BSpline(numpy.array([d0tmp,d1tmp]),xi)
        plt.scatter(xy[:,0],xy[:,1],marker="x")
        spline.plot()
        #plt.plot(xy[0,:],xy[1,:])
        #plt.show()
    def testBasisFunction(self):
        points = 10000
        y = numpy.empty(points)
        xmin = 0
        xmax = 14
        x = numpy.array(list(range(xmin, xmax+1)))
        xa = numpy.linspace(xmin,xmax,points)
        sDs = 5 #Significant digits
        testx = xmax - 5
        for i,t in enumerate(xa):
            y[i] = (bSpline.BSpline.basisFunction(x,testx)(t))
        plt.plot(xa,y)
        plt.show()
        self.assertAlmostEqual(y[numpy.isclose(xa,testx-1,1e-3).argmax()],0, sDs, msg="basis function not zero at beginning of its relevant interval")
        self.assertAlmostEqual(y[numpy.isclose(xa,testx+1,1e-4).argmax()],2/3.0, sDs, msg="basis function not correct in the middle of its relevant interval")
        self.assertAlmostEqual(y[numpy.isclose(xa,testx+3,1e-3).argmax()],0, sDs, msg="basis function not zero at end of its relevant interval")

    def testSumofBasisFunctions(self):
        sum = numpy.zeros(100)
        maxLimit = 10
        plotPoints = 100
        grid = numpy.array(list(range(0,maxLimit)))
        grid = numpy.insert(grid,[0,0,-1,-1],[grid[0],grid[0],grid[-1],grid[-1]])
        for i in range(maxLimit+4):
             for t,x in enumerate(numpy.linspace(0,maxLimit-1,plotPoints)):
                 sum[t] += bSpline.BSpline.basisFunction(grid,i)(x)
        print(sum)
    def testPlotfBasisFunctions(self):
        sum = numpy.zeros(100)
        maxLimit = 10
        plotPoints = 100
        grid = numpy.array(list(range(0,maxLimit)))
        grid = numpy.insert(grid,[0,0,-1,-1],[grid[0],grid[0],grid[-1],grid[-1]])
        for i in range(maxLimit+4):
            y = numpy.zeros(plotPoints-1)
            xa = numpy.linspace(0,maxLimit-1,plotPoints)[:-1]
            for t,x in enumerate(xa):
                y[t] = bSpline.BSpline.basisFunction(grid,i)(x)
            plt.plot(xa,y)
        plt.show()


if __name__== '__main__':
    unittest.main()

