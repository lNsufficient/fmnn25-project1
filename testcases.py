import unittest
from math import *
import bSpline
import numpy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('error')


class testBSpline(unittest.TestCase):
    def testAlphaDivideByZero(self):
        a = bSpline.BSpline(numpy.array([0,0,0,1,1,1]),numpy.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]))
        try:
            result = a.alpha(3,3)(1)
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
        print(grid)
        spline = bSpline.BSpline(grid, d)
        spline.plot()
        self.assertTrue(True)
    def testInterpolation(self):
        xy = numpy.array([[1, 0, 4, -1],[1, -3, 2, -1]])
        u = numpy.array([0,0,0, 1, 1, 1])
        d = bSpline.BSpline.interpolation(u, xy)
        spline = bSpline.BSpline(u, d)
        spline()
        plt.plot(xy[0,:],xy[1,:])
        plt.show()


if __name__== '__main__':
    unittest.main()

