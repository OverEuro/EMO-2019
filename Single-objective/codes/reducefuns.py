# -*- coding: utf-8 -*-
"""
SuSTech OPAL Lab

A test function sets for optimization from Yi-jun Yang (Total 30 functions)

All functions are scalable and with closed form for fast evaluation

Using example:"

import testfuns as tfs
f = tfs.Ackley(dim) # you need to pre-define the dimension of function i.e. "dim". Then
you can use "f.do_evaluate" to compute a single-value result according solution x.
x: a ndarray with size [dim]. Of course, you can get other attributes by f.bounds, etc."

"""
import math
import random
import numpy as np
np.seterr(all = 'ignore')

random.seed(5)
pop = list(np.arange(0,1000,1))
ri = np.array(random.sample(pop,10))

def lzip(*args):
    """
    returns zipped result as a list.
    """
    return list(zip(*args))

def generateID(x,best,dim,ri,bounds):
    tx = x.copy()
    x = np.array(best)
    if np.any(x < bounds[0,0]):
        lid = x < bounds[0,0]
        x[lid] = 2 * bounds[0,0] - x[lid]
    if np.any(x > bounds[0,1]):
        uid = x > bounds[0,1]
        x[uid] = 2 * bounds[0,1] - x[uid]
    x[ri] = tx[ri]
    return x

class Ackley(object): # 01
    def __init__(self, dim):
#        super(Ackley, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [30] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 22.26946404462
        self.classifiers = ['complicated', 'oscillatory', 'unimodal', 'noisy']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        a = 20
        b = 0.2
        c = 2 * math.pi
        return (-a * math.exp(-b * np.sqrt(1.0 / self.dim * np.sum(x ** 2))) -
                math.exp(1.0 / self.dim * np.sum(np.cos(c * x))) + a + math.exp(1))
        
class Alpine01(object): # 02
    def __init__(self, dim):
#        super(Alpine01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-6] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 8.71520568065 * self.dim
        self.classifiers = ['nonsmooth']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))
    
class ArithmeticGeometricMean(object): # 03
    def __init__(self, dim):
#        super(ArithmeticGeometricMean, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = (10 * (self.dim - 1.0) / self.dim) ** 2
        self.classifiers = ['bound_min', 'boring', 'multi_min']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return (np.mean(x) - np.prod(x) ** (1.0 / self.dim)) ** 2
    
class Csendes(object): # 04
    def __init__(self, dim):
#        super(Csendes, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-0.5] * self.dim, [1] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([1] * self.dim))
        self.classifiers = ['unimodal']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum((x ** 6) * (2 + np.sin(1 / (x + np.finfo(float).eps))))
    
class Deb01(object): # 05
    def __init__(self, dim):
#        super(Deb01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-1] * self.dim, [1] * self.dim))
        self.min_loc = [0.3] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['oscillatory', 'multi_min']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return -(1.0 / self.dim) * np.sum(np.sin(5 * math.pi * x) ** 6)
    
class Deb02(object): # 06
    def __init__(self, dim):
#        super(Deb02, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [1] * self.dim))
        self.min_loc = [0.0796993926887] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['oscillatory', 'multi_min']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return -(1.0 / self.dim) * np.sum(np.sin(5 * math.pi * (x ** 0.75 - 0.05)) ** 6)
    
class DeflectedCorrugatedSpring(object): # 07
    def __init__(self, dim):
#        super(DeflectedCorrugatedSpring, self).__init__(dim)
        self.dim = dim
        self.alpha = 5.0
        self.K = 5.0
        self.bounds = np.array(lzip([0] * self.dim, [1.5 * self.alpha] * self.dim))
        self.min_loc = [self.alpha] * self.dim
        self.fmin = self.do_evaluate(np.asarray(self.min_loc))
        self.fmax = self.do_evaluate(np.zeros(self.dim))
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return -np.cos(self.K * np.sqrt(np.sum((x - self.alpha) ** 2))) + 0.1 * np.sum((x - self.alpha) ** 2)
    
class DropWave(object): # 08
    def __init__(self, dim):
#        super(DropWave, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-2] * self.dim, [5.12] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        norm_x = sum(x ** 2)
        return -(1 + np.cos(12 * np.sqrt(norm_x))) / (0.5 * norm_x + 2)
    
class Easom(object): # 09
    def __init__(self, dim):
#        super(Easom, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-100] * self.dim, [20] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 22.3504010789
        self.classifiers = ['unimodal', 'boring']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        a = 20
        b = 0.2
        c = 2 * math.pi
        n = self.dim
        return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(c * x)) / n) + a + np.exp(1)
    
class Exponential(object): # 10
    def __init__(self, dim):
#        super(Exponential, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-0.7] * self.dim, [0.2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = self.do_evaluate(np.asarray([-0.7] * self.dim))
        self.classifiers = ['unimodal']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return -np.exp(-0.5 * np.sum(x ** 2))
    
class Perm01(object): # 11
    def __init__(self, dim):
#        assert dim > 1
#        super(Perm01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim))
        self.min_loc = [1] * self.dim
        self.fmin = 0
#        self.fmax = self.do_evaluate([self.dim + 1] * self.dim)
        self.classifiers = ['unscaled']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(
            np.sum([(j ** k + 0.5) * ((x[j - 1] / j) ** k - 1) for j in range(1, self.dim + 1)]) ** 2
            for k in range(1, self.dim + 1)
        )
        
class Perm02(object): # 12
    def __init__(self, dim):
#        assert dim > 1
#        super(Perm02, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim))
        self.min_loc = 1 / np.arange(1, self.dim + 1)
        self.fmin = 0
#        self.fmax = self.do_evaluate([self.dim + 1] * self.dim)
        self.classifiers = ['unscaled']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(
            np.sum([(j + 10) * (x[j - 1]**k - (1.0 / j)**k) for j in range(1, self.dim + 1)]) ** 2
            for k in range(1, self.dim + 1)
        )
    
class Plateau(object): # 13
    def __init__(self, dim):
#        super(Plateau, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-2.34] * self.dim, [5.12] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 30
#        self.fmax = self.do_evaluate([5.12] * self.dim)
        self.classifiers = ['discrete', 'unimodal']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return 30 + np.sum(np.floor(np.abs(x)))
    
class RippleSmall(object): # 14
    def __init__(self, dim):
#        assert dim == 2
#        super(RippleSmall, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [1] * self.dim))
        self.min_loc = [0.1] * self.dim
        self.fmin = -2.2
        self.fmax = 0
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(-np.exp(-2 * np.log(2) * ((x - 0.1) / 0.8) ** 2) * (np.sin(5 * math.pi * x) ** 6 + 0.1 * np.cos(500 * math.pi * x) ** 2))
    
class RippleBig(object): # 15
    def __init__(self, dim):
#        assert dim == 2
#        super(RippleBig, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [1] * self.dim))
        self.min_loc = [0.1] * self.dim
        self.fmin = -2
        self.fmax = 0
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(-np.exp(-2 * np.log(2) * ((x - 0.1) / 0.8) ** 2) * (np.sin(5 * math.pi * x) ** 6))
    
class RosenbrockLog(object): # 16
    def __init__(self, dim):
#        assert dim == 11
#        super(RosenbrockLog, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-30] * self.dim, [30] * self.dim))
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 10.09400460102

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.log(1 + np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    
class Salomon(object): # 17
    def __init__(self, dim):
#        super(Salomon, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-100] * self.dim, [50] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-100] * self.dim))
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return 1 - np.cos(2 * math.pi * np.sqrt(np.sum(x ** 2))) + 0.1 * np.sqrt(np.sum(x ** 2))
    
class Sargan(object): # 18
    def __init__(self, dim):
#        assert dim > 1
#        super(Sargan, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-2] * self.dim, [4] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([4] * self.dim))
        self.classifiers = ['unimodal']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        x0 = x[:-1]
        x1 = np.roll(x, -1)[:-1]
        return np.sum(self.dim * (x ** 2 + 0.4 * np.sum(x0 * x1)))
    
class Schwefel20(object): # 19
    def __init__(self, dim):
#        super(Schwefel20, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-60] * self.dim, [100] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([100] * self.dim))
        self.classifiers = ['unimodal', 'nonsmooth']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(np.abs(x))
    
class Schwefel22(object): # 20
    def __init__(self, dim):
#        super(Schwefel22, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([10] * self.dim))
        self.classifiers = ['unimodal', 'nonsmooth']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    
class Schwefel26(object): # 21
    def __init__(self, dim):
#        assert dim == 2
#        super(Schwefel26, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-500] * self.dim, [500] * self.dim))
        self.min_loc = [420.968746] * self.dim
        self.fmin = 0
        self.fmax = 1675.92130876
        self.classifiers = ['oscillatory', 'multimin']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return -np.sum(x * np.sin(np.sqrt(np.abs(x)))) + 418.9829 * self.dim
    
class SineEnvelope(object): # 22
    def __init__(self, dim):
#        assert dim > 1
#        super(SineEnvelope, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-20] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.dim - 1
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        x_sq = x[0:-1] ** 2 + x[1:] ** 2
        return sum((np.sin(np.sqrt(x_sq)) ** 2 - 0.5) / (1 + 0.001 * x_sq) ** 2 + 0.5)
    
class Step(object): # 23
    def __init__(self, dim):
#        super(Step, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [5] * self.dim))
        self.min_loc = [0.5] * self.dim
        self.fmin = self.do_evaluate(np.asarray([0] * self.dim))
        self.fmax = self.do_evaluate(np.asarray([5] * self.dim))
        self.classifiers = ['discrete', 'unimodal']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum((np.floor(x) + 0.5) ** 2)
    
class StyblinskiTang(object): # 24
    def __init__(self, dim):
#        super(StyblinskiTang, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [5] * self.dim))
        self.min_loc = [-2.903534018185960] * self.dim
        self.fmin = -39.16616570377142 * self.dim
        self.fmax = self.do_evaluate(np.asarray([5] * self.dim))

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2
    
class SumPowers(object): # 25
    def __init__(self, dim):
#        super(SumPowers, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-1] * self.dim, [0.5] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-1] * self.dim))
        self.classifiers = ['unimodal','boring']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum([np.abs(x) ** (i + 1) for i in range(1, self.dim + 1)])
    
class Weierstrass(object): # 26
    def __init__(self, dim):
#        super(Weierstrass, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-0.5] * self.dim, [0.2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = self.do_evaluate(np.asarray(self.min_loc))
        self.fmax = self.do_evaluate(np.asarray([-0.5] * self.dim))
        self.classifiers = ['complicated']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        a, b, kmax = 0.5, 3, 20
        ak = a ** (np.arange(0, kmax + 1))
        bk = b ** (np.arange(0, kmax + 1))
        return np.sum([np.sum(ak * np.cos(2 * math.pi * bk * (xx + 0.5))) - self.dim * np.sum(ak * np.cos(math.pi * bk)) for xx in x])
    
class XinSheYang02(object): # 27
    def __init__(self, dim):
#        assert dim == 2
#        super(XinSheYang02, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-math.pi] * self.dim, [2 * math.pi] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 88.8266046808
        self.classifiers = ['nonsmooth', 'unscaled']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x ** 2)))
    
class XinSheYang03(object): # 28
    def __init__(self, dim):
#        assert dim == 2
#        super(XinSheYang03, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [20] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 1
        self.classifiers = ['boring', 'unimodal']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        beta, m = 15, 5
        return np.exp(-np.sum((x / beta) ** (2 * m))) - 2 * np.exp(-np.sum(x ** 2)) * np.prod(np.cos(x) ** 2)
    
class YaoLiu(object): # 29
    def __init__(self, dim):
#        super(YaoLiu, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5.12] * self.dim, [2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-4.52299366685] * self.dim))
        self.classifiers = ['oscillatory','as same as Rastrigin']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x) + 10)
    
class ZeroSum(object): # 30
    def __init__(self, dim):
#        super(ZeroSum, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-8] * self.dim, [6] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 1
        self.fmax = self.do_evaluate(np.asarray([-8] * self.dim))
        self.classifiers = ['nonsmooth', 'multi_min']

    def do_evaluate(self, x):
        x = generateID(x,self.min_loc,self.dim,ri,self.bounds)
        return 1 + (10000 * np.abs(np.sum(x))) ** 0.5