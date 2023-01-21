import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale,resize,iradon
import scipy
import scipy.fftpack
from scipy.interpolate import griddata
from time import time

def ramp_filter(fft2,rt):
    Y,X = np.meshgrid(np.arange(fft2.shape[0]),np.arange(fft2.shape[1]))
    Y = Y - fft2.shape[0]//2
    X = X - fft2.shape[1]//2
    R = np.hypot(Y,X) # R function
    R = (R/R.max())**(1/rt) # Scale to 1 max

    plt.plot(R[:,fft2.shape[1]//2])
    plt.title('Ramp Filter')
    plt.show()

    fft2 = fft2*R
    return fft2

# Imma try making a dummy radon FFT function using grid space..!
def radon_FFT(I,ang_space):
    # Spatial hyper parameters
    M,N = I.shape
    K = 300 # Radius resolution
    m = math.ceil(M/2)
    n = math.ceil(N/2)
    #rho_space = np.arange(-math.ceil(math.sqrt(m**2 + n**2)),math.ceil(math.sqrt(m**2 + n**2))+1)
    rho_space = np.linspace(-math.ceil(math.sqrt(m**2 + n**2)),math.ceil(math.sqrt(m**2 + n**2)),num = K,endpoint=True)
    # What we can try here is....
    # Assign each grid of (x,y) (after flattening) to radius for each angle
    # Then use binning function (assignment) and take cummulative sum
    x = np.arange(N) - N//2
    y = np.arange(M) - M//2

    Y,X = np.meshgrid(y,x)
    Y = Y.flatten()  # The y vector 
    X = X.flatten()  # The x vector
    
    proj = np.array([-X*np.sin(ang)+Y*np.cos(ang) for ang in np.deg2rad(ang_space)]) # I forgot the formula so fuck that
    # Dimension is N_ang by M*N
    # Now we should have pair for X,Y -> Radius per angle
    
    # Next step: We should find a function that bins to correct discrete radius (axis = 1) per angle (summing shoudl happen there too)
    binned = np.array([np.digitize(p,rho_space) for p in proj])
    # Nang by MN we now want Nang by rhospace
    sinogram = np.array([[I.flatten()[binned[k,:] == j].sum() for j in np.arange(len(rho_space))] for k in range(len(ang_space))]).T
    sinogram=scipy.ndimage.gaussian_filter1d(sinogram,sigma = 1.7, axis=0)
    return sinogram,rho_space


def radon_FFT_manual(I,ang_space):
    # Evaluate for th1
    M,N = I.shape
    m = math.ceil(M/2)
    n = math.ceil(N/2)
    rho_space = np.arange(-math.ceil(math.sqrt(m**2 + n**2)),math.ceil(math.sqrt(m**2 + n**2))+1)
    sinogram = np.zeros((rho_space.shape[0],ang_space.shape[0]))

    for j,ang in enumerate(ang_space):
        # Version 2
        # Case 1 : 0 < theta <= 45
        th = np.deg2rad(ang)
        if th > 0 and th <= np.pi/4:
            for i,r in enumerate(rho_space):
                a = -math.cos(th)/math.sin(th)
                b = r/math.sin(th)

                ymin = max(-n,round(a*(m)+b))
                ymax = min(n,round(a*(-m)+b))
                for y in np.arange(ymin,ymax):

                    # FLOOR CASE
                    x = min(math.floor((y-b)/a),m-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]
                    
                    # CEILING CASE
                    x = min(math.ceil((y-b)/a),m-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]

        # Case 2 : 45 < theta <= 90
        if th > np.pi/4 and th <= np.pi/2:
            for i,r in enumerate(rho_space):
                a = -math.cos(th)/math.sin(th)
                b = r/math.sin(th)

                xmin = max(-m,round((n-b)/a))
                xmax = min(m,round((-n-b)/a))
                for x in np.arange(xmin,xmax):

                    # FLOOR CASE
                    y = min(math.floor(a*x+b),n-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]
                    
                    # CEILING CASE
                    y = min(math.ceil(a*x+b),n-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]


        # Case 3 : 90 < theta <= 135
        if th > np.pi/2 and th <= 3*np.pi/4:
            for i,r in enumerate(rho_space):
                a = -math.cos(th)/math.sin(th)
                b = r/math.sin(th)

                xmin = max(-m,round((-n-b)/a))
                xmax = min(m,round((n-b)/a))
                for x in np.arange(xmin,xmax):

                    # FLOOR CASE
                    y = min(math.floor(a*x+b),n-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]
                    
                    # CEILING CASE
                    y = min(math.ceil(a*x+b),n-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]

        # Case 4 : 135 < theta <= 180
        if th > 3*np.pi/4 and th <= np.pi:
            for i,r in enumerate(rho_space):
                a = -math.cos(th)/math.sin(th)
                b = r/math.sin(th)

                ymin = max(-n,round(a*(-m)+b))
                ymax = min(n,round(a*(m)+b))
                for y in np.arange(ymin,ymax):

                    # FLOOR CASE
                    x = min(math.floor((y-b)/a),m-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]
                    
                    # CEILING CASE
                    x = min(math.ceil((y-b)/a),m-1)
                    x = max(-(m-1),x)
                    y = max(-(n-1),y)
                    sinogram[-i,j] = sinogram[-i,j] + I[y+(n-1),x+(m-1)]

    return sinogram,rho_space

# Define our iradon function based on griddata fnx
def iradon_FFT(sinogram,rho_space,ang_space,ramp=False,rt=2):

    plt.imshow(sinogram,extent =[ang_space.min(),ang_space.max(),rho_space.min(),rho_space.max()])
    plt.xlabel('Angle (deg)')
    plt.ylabel(r'$\rho$')
    plt.show()

    # Fourier transform the rows of the sinogram

    sinogram_fft_cols=np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(sinogram,axes=0),
            axis=0),
        axes=0)

    plt.figure()
    plt.subplot(121)
    plt.title("Sinogram rows FFT (real)")
    plt.imshow(np.real(sinogram_fft_cols),vmin=-50,vmax=50)
    plt.subplot(122)
    plt.title("Sinogram rows FFT (imag)")
    plt.imshow(np.imag(sinogram_fft_cols),vmin=-50,vmax=50)
    plt.show()

    S = rho_space.shape[0]

    # Extract angular space and radius space as flattened vector 
    a=np.deg2rad(ang_space) # express in radians
    r=np.arange(S)-S/2
    a,r=np.meshgrid(a,r) # angle,radius space -> y:r, x:angle
    r=r.flatten()
    a=a.flatten()

    # Coordinate transform from a,r -> x,y
    srcx=(S//2)+r*np.cos(a) # zero-centered x as a fnx of r and a 
    srcy=(S//2)+r*np.sin(a) # zero-centered y as a fnx of r and a 

    # Coordinates of regular grid in 2D FFT space
    dstx,dsty=np.meshgrid(np.arange(S),np.arange(S))
    #dstx,dsty = np.mgrid[range(S),range(S)]
    dstx=dstx.flatten()
    dsty=dsty.flatten()

    # Let the central slice theorem work its magic!
    # Interpolate the 2D Fourier space grid from the transformed sinogram rows
    fft2=scipy.interpolate.griddata(
    (srcy,srcx),
    sinogram_fft_cols.flatten(),
    (dsty,dstx),
    method='cubic',
    fill_value=0.0
    ).reshape((S,S))

    plt.figure()
    plt.subplot(121)
    plt.title("FFT2 (real)")
    plt.imshow(np.real(fft2),vmin=-50,vmax=50)
    plt.subplot(122)
    plt.title("FFT2 (imag)")
    plt.imshow(np.imag(fft2),vmin=-50,vmax=50)
    plt.show()

    # Apply ramp filter if ramp == True
    if ramp:
        fft2 = ramp_filter(fft2,rt)
    

    # Transform from 2D Fourier space back to a reconstruction of the target
    recon=np.real(
    scipy.fftpack.fftshift(
        scipy.fftpack.ifft2(
            scipy.fftpack.ifftshift(fft2)
            )
        )
    )[::-1,::-1] # This is to make the zero-center symmetry to happen (undo ifftshift earlier)

    plt.figure()
    plt.title("Reconstruction")
    plt.imshow(recon/recon.max(),vmin=0.0,vmax=1)

    plt.show()

if __name__ == '__main__':

    # Part 1: Initiate Image
    M = 501
    N = 501
    m = math.ceil(M/2)
    n = math.ceil(N/2)
    I = shepp_logan_phantom()
    I = resize(I, (M,N),anti_aliasing=True)

    plt.figure()
    plt.title(f"Original for size: {(M,N)}")
    plt.imshow(I,vmin=0.0,vmax=1)
    plt.show()



    # Part 2: Define angle space
    ang_space = np.linspace(0, 180, num=250, endpoint=True)
    sinogram,rho_space = radon_FFT(I,ang_space)


    # # Part 3: Find Sinogram
    # sinogram,rho_space = radon_FFT(I,ang_space)

    # # Part 4: Perform iradon_FFT 
    ramp = False
    rt = 2
    iradon_FFT(sinogram,rho_space,ang_space,ramp,rt)
    # Iout = iradon(sinogram, theta=ang_space)
    # plt.figure()
    # plt.title("Reconstruction")
    # plt.imshow(Iout,vmin=0.0,vmax=1)
    # plt.show()


    # # Evaluate runtime for different dimensions
    # sz = [11,21,31,71,101,201,401]
    # time_list_RT = []
    # time_list_iRT = []
    # iteration = 5
    # for s in sz:
    #     print(f'Now with size {s}')
    #     # Part 1: Initiate Image
    #     M = s
    #     N = s
    #     m = math.ceil(M/2)
    #     n = math.ceil(N/2)
    #     I = shepp_logan_phantom()
    #     I = resize(I, (M,N),anti_aliasing=True)
    #     ang_space = np.linspace(0, 180, num=250, endpoint=True)
    #     stime = time()
    #     sinogram = None
    #     rho_space = None
    #     for t in range(iteration):
    #         sinogram,rho_space = radon_FFT(I,ang_space)
    #     etime = time()
    #     time_list_RT.append((etime-stime)/iteration)

    #     # Time for iRT
    #     print(f'Now iRT with size {s}')
    #     stime = time()
    #     for t in range(iteration):
    #         iradon_FFT(sinogram,rho_space,ang_space)
    #     etime = time()
    #     time_list_iRT.append((etime-stime)/iteration)
    
    # plt.figure()
    # plt.scatter(sz,time_list_RT)
    # plt.title('Runtime for RT')
    # plt.xlabel('Image Size')
    # plt.ylabel('Avg Runtime (s)')
    # plt.show()


    # plt.figure()
    # plt.scatter(sz,time_list_iRT)
    # plt.title('Runtime for Inverse RT')
    # plt.xlabel('Image Size')
    # plt.ylabel('Avg Runtime (s)')
    # plt.show()