from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Default seed 
SEED = 42

"""
All functions should generate 3 files
1. t (time array)
2. x (dimension/space array if applicable)
3. y (data values with dimension len(t) x len(x)) 

The data is standardized to allow comparison between models.
10% of the data and removed before saving the data (to account for transient behaviour)
"""

## CREATING DATA ##
def lorenz_96(F, output_dim, tN, delta_t, seed = SEED, prefix = "lorenz"):
    """
    Utilises Runge-Kutta method of order 5 to generate a .npy file of the data.
    Does not return anything
    
    args
    ======
        F: forcing constant (larger F represents higher complexity)
        output_dim: dimension of data
        tN: end time
        delta_t: interval to sample data
        seed: seed for random starting point
        prefix: file name to save the data 
    """    
    def model(t, x):
        return np.roll(x, shift = 1) * (np.roll(x, shift = -1) - np.roll(x, shift = 2)) - x + F
    
    x0 = np.random.normal(size = output_dim)
    t = np.linspace(0, tN, int(tN / delta_t))
    solved_model = solve_ivp(model, [0, tN], x0, t_eval = t)
    
    time = solved_model.t
    spatial = np.array([i for i in range(output_dim)])
    yns = solved_model.y.T   
    
    yns = (yns - np.mean(yns)) / np.std(yns)
    
    transient = int(0.1 * yns.shape[0])
    time = time[:-transient]
    yns = yns[transient:]
    
    time.dump(f"data/{prefix}{F}_time.npy")
    spatial.dump(f"data/{prefix}{F}_space.npy")
    yns.dump(f"data/{prefix}{F}_data.npy")
    
    print(f"Lorenz_96 with F = {F}, dt = {delta_t} generated. \nTotal number of points: {yns.shape[0]}")
        
def lyapunov_lorenz(n_iters, epsilon, data, delta_t, F = 8):
    """
    Returns the maximal lyapunov exponent for the Lorenz-96 model
    
    n_iters: number of trajectories created
    epsilon: perturbation amount
    data: training data (after transient removed)
    delta_t: time interval
    F : forcing constant
    """
    def model(t, x):
        return np.roll(x, shift = 1) * (np.roll(x, shift = -1) - np.roll(x, shift = 2)) - x + F
    
    ########################
    
    np.random.seed(SEED)
    
    T = 10
    N = int(T / delta_t) # Number of points to generate
    data_size = data.shape[0]
    t = np.linspace(0, T, N)
    
    # Calculate expected distance
    exp_dist = []
    for i in range(n_iters**2):
        pt1 = np.random.randint(0, data_size)
        pt2 = np.random.randint(0, data_size)
        dist = np.linalg.norm(data[pt1] - data[pt2])
        exp_dist.append(dist)
    
    log_exp_dist = np.log(np.mean(exp_dist))
    
    all_slope = []
    all_log_abs = []
    for _ in range(n_iters):
        idx = np.random.randint(0, data_size)
        X0 = data[idx]
        X0_pert = X0 + np.random.normal(0, epsilon, np.shape(X0))
        
        log_abs_error_lst = []
        
        for i in range(N):
            X0 = solve_ivp(model, [0, delta_t], X0.flatten(), t_eval = [delta_t]).y.T
            X0_pert = solve_ivp(model, [0, delta_t], X0_pert.flatten(), t_eval = [delta_t]).y.T
            log_abs_error = np.log(np.linalg.norm(X0 - X0_pert))
            log_abs_error_lst.append(log_abs_error)
        thresh = np.where(log_abs_error_lst < 0.9 * log_exp_dist)[0][-1]
        Y = np.array(log_abs_error_lst[:thresh])
        X = np.linspace(0, len(Y)*delta_t, len(Y)) 
        slope = ((X * Y).mean() - X.mean() * Y.mean()) / ((X**2).mean() - (X.mean())**2)
        
        all_slope.append(slope)
        all_log_abs.append(np.array(log_abs_error_lst))
    
    return np.mean(all_slope)

def ks(L, nx, tN, delta_t, splits, seed = SEED, prefix = "ks"):
    """
    Utilises Runge-Kutta method of order 5 to generate a .npy file of the data.
    Does not return anything
    
    args
    ======
        L: max spatial dimension. Controls stability (high L results in unstable dynamics)
        nx: number of spatial  
        t0: start time
        tN: end time
        delta_t = time interval
        splits = data is generated at dt/splits rate, then every splits-th data is taken
    """    
    def generate_data(t0, tN, u0):
        """
        Adapted from https://scicomp.stackexchange.com/questions/37336/solving-numerically-the-1d-kuramoto-sivashinsky-equation-using-spectral-methods
        """
        ddt = delta_t / splits
        ddx = nx
        nu = 1
        
        nt = int((tN - t0) / ddt)    # Total number of time intervals

        # wave number mesh
        k = np.arange(-ddx/2, ddx/2, 1)

        t = np.linspace(start=t0, stop=tN, num=nt)
        x = np.linspace(start=0, stop=L, num=ddx)

        # solution mesh in real space
        u = np.ones((ddx, nt))
        # solution mesh in Fourier space
        u_hat = np.ones((ddx, nt), dtype=complex)

        u_hat2 = np.ones((ddx, nt), dtype=complex)
        
        # Initial condition u0 set by create_traj function

        # Fourier transform of initial condition
        u0_hat = (1 / ddx) * np.fft.fftshift(np.fft.fft(u0))

        u0_hat2 = (1 / ddx) * np.fft.fftshift(np.fft.fft(u0**2))

        # set initial condition in real and Fourier mesh
        u[:,0] = u0
        u_hat[:,0] = u0_hat

        u_hat2[:,0] = u0_hat2

        # Fourier Transform of the linear operator
        FL = (((2 * np.pi) / L) * k) ** 2 - nu * (((2 * np.pi) / L) * k) ** 4
        # Fourier Transform of the non-linear operator
        FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)

        # resolve EDP in Fourier space
        for j in range(0,nt-1):
            uhat_current = u_hat[:,j]
            uhat_current2 = u_hat2[:,j]
            if j == 0:
                uhat_last = u_hat[:,0]
                uhat_last2 = u_hat2[:,0]
            else:
                uhat_last = u_hat[:,j-1]
                uhat_last2 = u_hat2[:,j-1]

            # compute solution in Fourier space through a finite difference method
            # Cranck-Nicholson + Adam 
            u_hat[:,j+1] = (1 / (1 - (ddt / 2) * FL)) * ( (1 + (ddt / 2) * FL) * uhat_current + ( ((3 / 2) * FN) * (uhat_current2) - ((1 / 2) * FN) * (uhat_last2) ) * ddt )
            # go back in real space
            u[:,j+1] = np.real(ddx * np.fft.ifft(np.fft.ifftshift(u_hat[:,j+1])))
            u_hat2[:,j+1] = (1 / ddx) * np.fft.fftshift(np.fft.fft(u[:,j+1]**2))
        
        return t, x, u.T
    
    np.random.seed(seed)
    
    t_all = []
    u_all = []
    
    curr_t = 0
    u0 = np.random.normal(size = nx)
    interval = 200
    
    while curr_t < tN:
        end_t = min(curr_t + interval, tN)
        t, spatial, u = generate_data(curr_t, end_t, u0)
        u_all.append(u)
        t_all.append(t)
        u0 = u[-1]
        curr_t = end_t
    
    time = np.concatenate(t_all)
    yns = np.concatenate(u_all)
    
    yns = yns[::splits, :]
    time = time[::splits]
        
    yns = (yns - np.mean(yns)) / np.std(yns)
    
    transient = int(0.1 * yns.shape[0])
    time = time[:-transient]
    yns = yns[transient:]
    
    time.dump(f"data/{prefix}{L}_time.npy")
    spatial.dump(f"data/{prefix}{L}_space.npy")
    yns.dump(f"data/{prefix}{L}_data.npy")
    
    print(f"KS Equation with L = {L}, dt = {delta_t} generated. \nTotal number of points: {yns.shape[0]}")


## DATA LOADING ##
class TSData():
    def __init__(self, name, time, space, data):
        self.name = name
        self.time = time
        self.space = space
        self.data = data
        
    def plot_sample(self, n = 4000, title = None):
        """
        args
        =====
            n: number of points to plot
            title: title of graph. Defaults to object name if not stated
        """
        fig, ax = plt.subplots(figsize=(15,3))
        xx, tt = np.meshgrid(self.space, self.time[:n])
        cs = ax.contourf(tt, xx, self.data[:n])
        fig.colorbar(cs)

        ax.set_xlabel("t")
        ax.set_ylabel("x")
        if not title:
            ax.set_title(self.name)
        else:
            ax.set_title(title)
        plt.show()

def plot_sample(data, n = 4000, title = None, save_name = False, cmap = "RdBu"):
    """
    args
    =====
        data: TSData type
        n: number of points to plot
        title: title of graph. Defaults to object name if not stated
    """
    if not isinstance(data, TSData):
        raise Exception("Invalid data type")
    fig, ax = plt.subplots(figsize=(20,4))
    xx, tt = np.meshgrid(data.space, data.time[:n])
    cs = ax.contourf(tt, xx, data.data[:n], cmap = cmap)
    fig.colorbar(cs)

    ax.set_xlabel("t")
    ax.set_ylabel("x")
    if not title:
        ax.set_title(data.name)
    else:
        ax.set_title(title)
    if save_name:
        plt.savefig(save_name + ".png", facecolor = "white", bbox_inches = "tight")
    plt.show()
    
def load_data(name, prefix, train_percent):
    """
    Loads data into an object. Returns a train and test dataset
    
    args 
    ======
        name: name of object
        prefix: location of file
        train_percent: precent of data to be used as training data
        
    returns
    ======
        train: TSData object containing the training data
        test: TSData object containing the test data
    """
    
    time = np.load(f"{prefix}_time.npy", allow_pickle = True)
    space = np.load(f"{prefix}_space.npy", allow_pickle = True)
    data = np.load(f"{prefix}_data.npy", allow_pickle = True)
            
    T_total, data_dim = data.shape
    T_train = int(T_total * train_percent)
    T_test = T_total - T_train

    train_data = data[:T_train]
    test_data = data[T_train:]
    train_time = time[:T_train]
    test_time = time[:T_test]
    
    return TSData(name + " Train", train_time, space, train_data), \
           TSData(name + " Test", test_time, space, test_data)
    

