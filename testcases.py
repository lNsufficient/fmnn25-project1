from math import *
import bSpline
import numpy 
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
