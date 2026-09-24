import numpy as np
import scipy.sparse as sp

DEFAULT_PEIERLS_SIGN = -1.0

def site_index(ix, iy, ny):
    return int(ix)*int(ny)+int(iy)

def build_periodic_x_hamiltonian(prep, periodic_x=True, peierls_sign=DEFAULT_PEIERLS_SIGN):
    x_nm=np.asarray(prep.x_nm,float); y_nm=np.asarray(prep.y_nm,float)
    U=np.asarray(prep.U_for_transport_meV,float)
    ny,nx=U.shape
    if len(x_nm)!=nx or len(y_nm)!=ny:
        raise ValueError(f'Grid/potential mismatch: U={U.shape}, x={len(x_nm)}, y={len(y_nm)}')
    if periodic_x and nx<3:
        raise ValueError('Periodic x requires nx >= 3.')
    t=float(prep.t_meV); alpha=float(prep.alpha)
    rows=[]; cols=[]; data=[]
    for ix in range(nx):
        for iy in range(ny):
            p=site_index(ix,iy,ny)
            rows.append(p); cols.append(p); data.append(4*t+U[iy,ix])
    # open y, real hopping
    for ix in range(nx):
        for iy in range(ny-1):
            p=site_index(ix,iy,ny); q=site_index(ix,iy+1,ny)
            rows += [p,q]; cols += [q,p]; data += [-t,-t]
    # periodic x, A=(-By,0,0): phase depends only on y, constant along x
    for iy in range(ny):
        theta = -float(peierls_sign)*2*np.pi*alpha*iy
        hop = -t*np.exp(1j*theta)
        for ix in range(nx-1):
            p=site_index(ix,iy,ny); q=site_index(ix+1,iy,ny)
            rows += [p,q]; cols += [q,p]; data += [hop,np.conjugate(hop)]
        if periodic_x:
            p=site_index(nx-1,iy,ny); q=site_index(0,iy,ny)
            rows += [p,q]; cols += [q,p]; data += [hop,np.conjugate(hop)]
    H=sp.coo_matrix((np.asarray(data,complex),(rows,cols)),shape=(nx*ny,nx*ny)).tocsr()
    x_site_nm=np.repeat(x_nm,ny)
    y_site_nm=np.tile(y_nm,nx)
    return H,x_site_nm,y_site_nm

def mean_y_from_probabilities(probabilities,y_site_nm):
    p=np.asarray(probabilities,float); y=np.asarray(y_site_nm,float)
    return float(y@p) if p.ndim==1 else y@p

def variance_y_from_probabilities(probabilities,y_site_nm):
    p=np.asarray(probabilities,float); y=np.asarray(y_site_nm,float)
    m=mean_y_from_probabilities(p,y)
    m2=float((y*y)@p) if p.ndim==1 else (y*y)@p
    return np.maximum(m2-m*m,0.0)

def hermiticity_error(H):
    d=H-H.getH()
    return 0.0 if d.nnz==0 else float(np.max(np.abs(d.data)))

def periodic_x_potential_mismatch(prep):
    U=np.asarray(prep.U_for_transport_meV,float)
    diff=U[:,-1]-U[:,0]
    return {'max_abs_meV':float(np.max(np.abs(diff))), 'rms_meV':float(np.sqrt(np.mean(diff**2))), 'mean_meV':float(np.mean(diff)), 'difference_meV':diff}

def plaquette_phase(prep,iy=0,peierls_sign=DEFAULT_PEIERLS_SIGN):
    alpha=float(prep.alpha)
    th0=-float(peierls_sign)*2*np.pi*alpha*iy
    th1=-float(peierls_sign)*2*np.pi*alpha*(iy+1)
    return float(np.angle(np.exp(1j*(th0-th1))))

def expected_plaquette_phase(prep,peierls_sign=DEFAULT_PEIERLS_SIGN):
    return float(np.angle(np.exp(1j*float(peierls_sign)*2*np.pi*float(prep.alpha))))

def analytic_zero_field_spectrum(nx,ny,t,onsite_offset=0.0):
    vals=[]
    for m in range(nx):
        kx=2*np.pi*m/nx
        for n in range(1,ny+1):
            ky=n*np.pi/(ny+1)
            vals.append(4*t+onsite_offset-2*t*np.cos(kx)-2*t*np.cos(ky))
    return np.sort(np.asarray(vals,float))

def benchmark_zero_field_periodic_x(nx=8,ny=7,t=1.0):
    class Dummy: pass
    prep=Dummy(); prep.x_nm=np.arange(nx,dtype=float); prep.y_nm=np.arange(ny,dtype=float)
    prep.U_for_transport_meV=np.zeros((ny,nx)); prep.t_meV=float(t); prep.alpha=0.0
    H,_,_=build_periodic_x_hamiltonian(prep,True)
    num=np.linalg.eigvalsh(H.toarray()); exact=analytic_zero_field_spectrum(nx,ny,t)
    return {'max_abs_error':float(np.max(np.abs(num-exact))), 'numerical':num, 'exact':exact}
