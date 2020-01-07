import numpy as np
import matplotlib.pyplot as plt
import ase
from ase.cluster import wulff_construction
from ase.visualize import view
import pyqstem
from pyqstem.imaging import CTF
from pyqstem import PyQSTEM
from data_augmentation import local_normalize,random_flip,random_contrast,random_brightness,random_blur,random_gamma,random_flip
from make_label import make_label


surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [6, 2, 4]   
#surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1),(2,1,0),(2,1,1),(2,2,1),(3,1,0),(3,1,1),(3,2,0),(3,2,1),(3,2,2),(3,3,1),(3,3,2)]
#esurf = [6, 2, 4,1,2,8,1,7,1,5,5,3,5]
lc = 4.08


n_min_atoms=100
n_max_atoms=2000

random_alpha=[[0,0],[45,0],[45,45]]

qstem=PyQSTEM('TEM')

img_size=256
random_sampling = np.random.uniform(low=0.250,high=0.265,size=1)[0]
cell_size=random_sampling*img_size

n_data=8000



training_images_folder_path='data/training/images/image_{}.npy'
training_labels_folder_path='data/training/labels/label_{}.npy'

for n in range(n_data):
    random_size = np.random.randint(low=n_min_atoms,high=n_max_atoms)  # Number of atoms

    atomic_model= wulff_construction('Au', surfaces, esurf,
                           random_size, 'fcc',
                           rounding='above', latticeconstant=lc)


    random_rotation=np.random.randint(len(random_alpha),size=1)[0]

    alpha_x=random_alpha[random_rotation][0]
    alpha_z=random_alpha[random_rotation][1]

    atomic_model.rotate(v='x',a=alpha_x,center='COP')
    atomic_model.rotate(v='z',a=alpha_z,center='COP')

    atomic_model.set_cell((cell_size,)*3)

    atomic_model.center()
    
    lx=np.random.uniform(-cell_size/6,cell_size/6)
    ly=np.random.uniform(-cell_size/6,cell_size/6)

    atomic_model.translate([lx,ly,0])
 
    wave_size=(int(atomic_model.get_cell()[0,0]/random_sampling),int(atomic_model.get_cell()[1,1]/random_sampling))
    qstem.set_atoms(atomic_model)
    qstem.build_wave('plane',300,wave_size)
    qstem.build_potential(int(atomic_model.get_cell()[2,2]*2))
    qstem.run()
    wave=qstem.get_wave()
    wave.array=wave.array.astype(np.complex64)


    Cs = np.random.uniform(low=-20*1e4,high=20*1e4,size=1)[0]

    defocus=np.random.uniform(low=-250,high=-150,size=1)[0]

    focal_spread=np.random.uniform(low=20,high=40,size=1)[0]

    aberrations={'a22' : np.random.uniform(low=0,high=50,size=1)[0],
                 'phi22' :  np.random.uniform(low=0,high=2*np.pi,size=1)[0]}
 
    ctf=CTF(defocus=defocus,Cs=Cs,focal_spread=focal_spread,aberrations=aberrations)

    blur= np.random.uniform(low=0,high=2,size=1)[0]

    dose=np.random.uniform(low=1e2,high=1e6,size=1)[0]

    c1=np.random.uniform(low=0.95,high=1,size=1)[0]

    c2=np.random.uniform(low=0,high=1e-2,size=1)[0]

    c3=np.random.uniform(low=5e-1,high=6e-1,size=1)[0]

    c4=c1=np.random.uniform(low=2,high=3,size=1)[0]

    MTF_param=[c1,c2,c3,c4]

    image=wave.apply_ctf(ctf).detect(resample=random_sampling,blur=blur,dose=dose,MTF_param=MTF_param)

    label=make_label(atomic_model,random_sampling,(img_size,)*2,width=2)
   
    image=local_normalize(image,12/random_sampling,12/random_sampling)

    image=random_brightness(image,-0.1,0.1)

    image=random_contrast(image,0.9,1.1)

    image=random_gamma(image,0.9,1.1)

    image=random_blur(image,0.9,1.1)

    image,label=random_flip(image,label,0,1)

    image=image.reshape((1,)+image.shape+(1,))
    
    label=label.reshape((1,)+label.shape+(1,))

    np.save(training_images_folder_path.format(n),image)
    
    np.save(training_labels_folder_path.format(n),label)
    
    print('data ' +str(n)+' are saved')


