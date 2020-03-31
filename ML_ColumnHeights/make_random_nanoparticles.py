from ase.build import bulk
from ase import Atoms
import numpy as np

class Random_Nanoparticle(object):
    
    directions=np.array(
        [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],
        [0,-1,0],[0,0,-1],[1,1,1],[-1,1,1],
        [1,-1,1],[1,1,-1],[-1,-1,1],[-1,1,-1],
        [1,-1,-1],[-1,-1,-1]])
    directions=(directions.T/np.linalg.norm(directions,axis=1)).T
    
    def __init__(self,lc,N):
        
        self.sites=self.get_sites(lc,N)
        self.bonds=self.get_bonds(lc/np.sqrt(2))
    
    def get_sites(self, lc, N):
        atoms = bulk('Au', 'fcc', a=lc, cubic=True)
        atoms*=(N,)*3
        atoms.center()
        
        self.center = np.diag(atoms.get_cell())/2
        return atoms.get_positions()
        
    def get_bonds(self, bond_length):

        bonds = []
        for i, s in enumerate(self.sites):
            distances = np.linalg.norm(self.sites - s, axis=1)
            indices = np.where(distances <= bond_length * 1.05)[0]
            bonds.append(indices)
        
        return bonds
            
    def create_seed(self,lengths100,lengths111):
        
        self.active=np.ones(len(self.sites),dtype=bool)
        
        lengths = np.hstack((lengths100,lengths111))
        
        for length,direction in zip(lengths,self.directions):
            r0=self.center+length*direction
            
            for i,site in enumerate(self.sites):
                if self.active[i]:
                    self.active[i]=np.sign(np.dot(direction,site-r0))==-1        
    
        self.active_bonds=np.array([self.active[b] for b in self.bonds])
        self.available_sites=np.where([any(ab)&(not a) for ab,a in zip(self.active_bonds,self.active)])[0]
        
    def build(self, N, T0, T1=None):
        
        if T1 is None:
            T1=T0
        
        for i in range(N):
            T=T0+(T1-T0)*i/N
            
            coordination=self.get_coordination(self.available_sites)
            
            p=np.zeros_like(coordination,dtype=np.float)
            
            p[coordination>2]=np.exp(coordination[coordination>2]/T)
            p=p/float(np.sum(p))
            p[p<0]=0
            
            n=np.random.choice(len(p), p=p)
            
            k=self.available_sites[n]
            
            self.available_sites = np.delete(self.available_sites, n)
            
            self.expand(k)
            
    def expand(self, k):
        
        self.active[k] = True
        
        new_avail = self.bonds[k][self.active[self.bonds[k]]==0]

        self.available_sites = np.array(list(set(np.append(self.available_sites, new_avail))))
        
        if len(new_avail)>0:
            to_update = np.array([np.where(self.bonds[x]==k)[0] for x in new_avail]).T[0]
            for i,j in enumerate(to_update):
                self.active_bonds[new_avail][i][j]=True
        
    def get_coordination(self, sites):
        return np.array([sum(self.active_bonds[site]) for site in sites])
    
    def get_cluster(self, element='Au'):
        return Atoms([element]*len(self.sites[self.active]),self.sites[self.active])


def get_random_atomic_model(element,random_size,cell_size):

    random_nanoparticle=Random_Nanoparticle(4,22)
    radius=5+np.random.rand()*random_size

    lengths100=np.random.uniform(radius,radius+.2*radius,6)
    lengths111=np.random.uniform(radius-.2*radius,radius,8)

    random_nanoparticle.create_seed(lengths100,lengths111)
    random_nanoparticle.build(int(np.sum(random_nanoparticle.active)/4.),10,2)
    random_nanoparticle.build(int(np.sum(random_nanoparticle.active)/4.),2,2)

    random_nanoparticle=random_nanoparticle.get_cluster(element=element)

    random_nanoparticle.rotate(v='y',a=45)

    omega=np.random.random()*360
    alpha=np.random.random()*0.0

    random_nanoparticle.rotate(v='z',a=omega,center='COP')
    random_nanoparticle.rotate(v='y',a=alpha,center='COP')
    random_nanoparticle.rotate(v='z',a=-omega,center='COP')

    random_nanoparticle.center(vacuum=0)
    
    size=np.diag(random_nanoparticle.get_cell())

    random_nanoparticle.set_cell((cell_size,)*3)
    random_nanoparticle.center()

    tx=(cell_size-size[0]-5)*(np.random.rand()-.5)
    ty=(cell_size-size[1]-5)*(np.random.rand()-.5)

    random_nanoparticle.translate((tx,ty,0))
    
    return random_nanoparticle
    
    

