import numpy as np
import pylab as pl


#ZDT3
class Individual():
    def __init__(self,x):
        self.x=x
        self.NumX=len(x)

        f1=float(x[0])
        h=float(1+x[1])
        f2=h*(1-(f1/h)**2-(f1/h)*np.sin(10*np.pi*f1))
        self.f=[f1,f2]    #multiobjective function
def Initial(N,dim):
    #initialize the population and the weight vector lambda list
    P=[]
    Lamb=[]
    for i in range(N):
        temp=[]
        x = np.random.rand(dim)
        P.append(Individual(x))
        temp.append(float(i)/(N))
        temp.append(1.0-float(i)/(N))
        Lamb.append(temp)

    return P,Lamb


#cal x dominated y or not
def Dominate(x,y,min=True):
    if min:

        for i in range(len(x.f)):
            if x.f[i]>y.f[i]:
                return False


        return True
    else:
        for i in range(len(x.f)):
            if x.f[i]<y.f[i]:
                return False
        return True
def Tchebycheff(x,lamb,z):
    #Tchebycheff approach operator

    temp=[]
    for i in range(len(x.f)):
        temp.append(np.abs(x.f[i]-z[i])*lamb[i])
    return np.max(temp)
def Neighbor(Lamb,T):
    #Lambda list,numbers of neighbors is T
    B=[]
    for i in range(len(Lamb)):
        temp=[]
        for j in range(len(Lamb)):
            distance=np.sqrt((Lamb[i][0]-Lamb[j][0])**2+(Lamb[i][1]-Lamb[j][1])**2)
            temp.append(distance)
        l=np.argsort(temp)
        B.append(l[:T])
    return B
def BestValue(P):

    best=[]
    for i in range(len(P[0].f)):
        best.append(P[0].f[i])
    for i in range(1,len(P)):
        for j in range(len(P[i].f)):
            if P[i].f[j]<best[j]:
                best[j]=P[i].f[j]

    return best

def mutation_cross(i, p, T, B, dim, F, CR):
    # Mutation
#    m, n = np.shape(XTemp)
    XMutationTmp = []
    r1 = 0
    r2 = 0
    r3 = 0
    while r1 == r2 or r1 == r3 or r2 == r3:
        r1 = np.random.randint(T)
        r2 = np.random.randint(T)
        r3 = np.random.randint(T)
    
    XMutationTmp = p[B[i][r1]].x + F * (p[B[i][r2]].x - p[B[i][r3]].x)

    # Cross-over
    XCorssOverTmp = np.ones(dim)
    for j in range(dim):
        r = np.random.rand()
        if (r <= CR):
            XCorssOverTmp[j] = XMutationTmp[j]
        else:
            XCorssOverTmp[j] = p[B[i][r1]].x[j]
        # Bounding check
        if XCorssOverTmp[j] > 1:
            XCorssOverTmp[j] = 2 * 1 - XCorssOverTmp[j]
        if XCorssOverTmp[j] < 0:
            XCorssOverTmp[j] = 2 * 0 - XCorssOverTmp[j]
    
    return Individual(p[B[i][r1]].x),Individual(XCorssOverTmp)


#the main algorithm
#N:population numbers
#T:the number of neighborhood of each weight vector
F = 0.5
CR = 0.9
N = 200
T = 200
dim = 100000
p,Lamb=Initial(N,dim)
B=Neighbor(Lamb,T)
z=BestValue(p)
EP=[]
t=0
while(t<30):
    t+=1
    print('PF number:',len(EP))
    for i in range(N):
        y1,y2 = mutation_cross(i, p, T, B, dim, F, CR)
        if Dominate(y1,y2):
            y=y1
        else:
            y=y2
        for j in range(len(z)):
            if y.f[j] < z[j]:
                z[j] = y.f[j]
        for j in range(len(B[i])):
            Ta = Tchebycheff(p[B[i][j]], Lamb[B[i][j]], z)
            Tb = Tchebycheff(y, Lamb[B[i][j]], z)
            if Tb < Ta:
                p[B[i][j]] = y
        if EP == []:
            EP.append(y)
        else:
            dominateY = False
            rmlist=[]
            for j in range(len(EP)):
                if Dominate(y, EP[j]):
                    rmlist.append(EP[j])
                elif Dominate(EP[j], y):
                    dominateY = True

            if dominateY == False:
                EP.append(y)
                for j in range(len(rmlist)):
                    EP.remove(rmlist[j])
                    
np.save("dim100000.npy",EP)
x = []
y = []
for i in range(len(EP)):
    x.append(EP[i].f[0])
    y.append(EP[i].f[1])

pl.plot(x, y, '*', label='Nondominated Solutions')
pl.grid(True)
pl.title('ZDT3 Test Function: Dim=100000')
pl.xlabel('f1')
pl.ylabel('f2')
pl.legend()
pl.savefig('100000',dpi=600,quality=95)
pl.show()