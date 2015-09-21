import unittest
from math import *
import bSpline
import numpy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('error')


class testBSpline(unittest.TestCase):
    def testAlphaDivideByZero(self):
        u = numpy.array([0,0,0,1,1,1])
        a = bSpline.BSpline(u, numpy.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]))
        try:
            result = a.alpha(u, 3,3)(1)
        except:
            self.fail("If nodes coincide 0/0 = 0, but we got division by zero error")
        expected = 0
        self.assertEqual(result, expected)
    def LooksGood(self):
#xtmp = numpy.linspace(0,2*pi,5)
        xtmp = numpy.array([37,73,42,7,3])
        x = numpy.append(xtmp[0],xtmp[0])
        x = numpy.append(x,xtmp)
        x = numpy.append(x,x[-1])
        x = numpy.append(x,x[-1])
        y = numpy.cos(x)
        d = numpy.array([x,y])

        gridtmp = numpy.linspace(0,1,5)
        grid = numpy.append(gridtmp[0],gridtmp[0])
        grid = numpy.append(grid,gridtmp)
        grid = numpy.append(grid,grid[-1])
        grid = numpy.append(grid,grid[-1])
        spline = bSpline.BSpline(grid, d)
        spline.plot()
        self.assertTrue(True)
    def testInterpolation(self):
        xy = numpy.array([[1, 0, 4, -1],[1, -3, 2, -1]])
        #u = numpy.array([0,0,1, 2, 3, 31])
        u = numpy.array([0,1,2,3,4,5,6,7])
        xi,dtmp = bSpline.BSpline.interpolation(u, xy)
        d0tmp = dtmp[0]
        d1tmp = dtmp[1]
        d = numpy.append(d0tmp[0],d0tmp[0])
        d = numpy.append(d,d0tmp)
        d = numpy.append(d,d[-1])
        d0 = numpy.append(d,d[-1])
        d = numpy.append(d0tmp[1],d0tmp[1])
        d = numpy.append(d,d1tmp)
        d = numpy.append(d,d[-1])
        d1 = numpy.append(d,d[-1])
        gridtmp = xi
        grid = numpy.append(gridtmp[0],gridtmp[0])
        grid = numpy.append(grid,gridtmp)
        grid = numpy.append(grid,grid[-1])
        grid = numpy.append(grid,grid[-1])    
        spline = bSpline.BSpline(grid, numpy.array([d0,d1]))
        spline()
        #plt.plot(xy[0,:],xy[1,:])
        #plt.show()
    def testBasisFunction(self):
        points = 10000
        y = numpy.empty(points)
        xa = numpy.linspace(0,7,points)
        for i,x in enumerate(xa):
            y[i] = (bSpline.BSpline.basisFunction(numpy.array([0,1,2,3,4,5,6,7,7]),3,3)(x))
        plt.plot(xa,y)
        plt.show()
        self.assertAlmostEqual(y[numpy.isclose(xa,2,1e-3).argmax()],0, msg="basis function not zero at beginning of its relevant interval")
        self.assertAlmostEqual(y[numpy.isclose(xa,4,1e-4).argmax()],2/3.0,places=4, msg="basis function not correct in the middle of its relevant interval")
        self.assertAlmostEqual(y[numpy.isclose(xa,6,1e-3).argmax()],0, msg="basis function not zero at end of its relevant interval")



if __name__== '__main__':
    unittest.main()

