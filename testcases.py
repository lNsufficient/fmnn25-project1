import unittest
from math import *
import bSpline
import numpy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('error')


class testBSpline(unittest.TestCase):
    def AlphaDivideByZero(self):
        u = numpy.array([0,0,0,1,1,1])
        a = bSpline.BSpline(numpy.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]),u)
        try:
            result = a.alpha(u, 3,3,1)
        except:
            self.fail("If nodes coincide 0/0 = 0, but we got division by zero error")
        expected = 0
        self.assertEqual(result, expected)
    def testDimensionOfDValues(self):
        pass
    def LooksGood(self):
#xtmp = numpy.linspace(0,2*pi,5)
        x = numpy.array([37,73,42,7,3])
        x = numpy.insert(x,[0,0,-1,-1],[x[0],x[0],x[-1],x[-1]])
        y = numpy.cos(x)
        d = numpy.array([x,y])
        
        grid = numpy.linspace(0,1,5)
        grid = numpy.insert(grid,[0,0,-1,-1],[grid[0],grid[0],grid[-1],grid[-1]])
        
        spline = bSpline.BSpline(d,grid)
        spline.plot()
        self.assertTrue(True)
    def testBasisFunctionEqualS(self):
        #test if s(u) = N_i for d_i = 1, d_j=0, j!=i for all i.
        
        spline = bSpline.BSpline() 
        numberValues = 12
        for i in range(gridPoints):
            spline.d = numpy.zeroes
        pass

    def LooksGood2(self):
        x = numpy.linspace(0,10)
        x = numpy.insert(x,[0,0,-1,-1],[x[0],x[0],x[-1],x[-1]])
        y = numpy.cos(x)
        d = numpy.array([x,y])     
        spline = bSpline.BSpline(d)
        spline.plot()
    def testInterpolation(self):
        #xy = numpy.array([[1, 0, 4, -1],[1, -3, 2, -1]])
        
        #u = numpy.array([0,0,1, 2, 3, 31])
        #u = numpy.array([0,1,2,3,4,5,6,6,6,0,0])
        
        #u = numpy.linspace(0,1,4)
        #u = numpy.insert(u,[0,0,-1,-1],[u[0],u[0],u[-1],u[-1]])
        x = numpy.array([37,73,42,7,3])
        x = numpy.insert(x,[0,0,-1,-1],[x[0],x[0],x[-1],x[-1]])
        y = numpy.cos(x)
        d = numpy.array([x,y])

        spline = bSpline.BSpline(d)
        xy = numpy.array([spline.findS(u) for u in spline.grid[1:-1]])
        print(numpy.shape(xy))
        print(numpy.shape(spline.grid))
        xi,dtmp = bSpline.BSpline.interpolation(spline.grid, xy)
        d0tmp = dtmp[0]
        d1tmp = dtmp[1]
        d0tmp = numpy.insert(d0tmp,[0,0,-1,-1],[d0tmp[0],d0tmp[0],d0tmp[-1],d0tmp[-1]])
        d1tmp = numpy.insert(d1tmp,[0,0,-1,-1],[d1tmp[0],d1tmp[0],d1tmp[-1],d1tmp[-1]])
        xi = numpy.insert(xi,[0,0,-1,-1],[xi[0],xi[0],xi[-1],xi[-1]])
        spline = bSpline.BSpline(numpy.array([d0tmp,d1tmp]),xi)
        plt.scatter(xy[:,0],xy[:,1],marker="x")
        spline()
        #plt.plot(xy[0,:],xy[1,:])
        #plt.show()
    def BasisFunction(self):
        points = 10000
        y = numpy.empty(points)
        xa = numpy.linspace(0,14,points)
        for i,x in enumerate(xa):
            #y[i] = (bSpline.BSpline.basisFunction(numpy.array([1,1,1,2,3,4,5,6,7,7,7]),1)(x))
            y[i] = (bSpline.BSpline.basisFunction(numpy.array([0,1,2,3,4,5,6,7,8,9,10,11,12]),11)(x))
        plt.plot(xa,y)
        plt.show()
        self.assertAlmostEqual(y[numpy.isclose(xa,2,1e-3).argmax()],0, msg="basis function not zero at beginning of its relevant interval")
        self.assertAlmostEqual(y[numpy.isclose(xa,4,1e-4).argmax()],2/3.0,places=4, msg="basis function not correct in the middle of its relevant interval")
        self.assertAlmostEqual(y[numpy.isclose(xa,6,1e-3).argmax()],0, msg="basis function not zero at end of its relevant interval")

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
        #print(sum)


if __name__== '__main__':
    unittest.main()

