import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pickle
import os

NUM_TEST = 100 # Number of testing samples
EARLY_STOPPING = 30 # Early stopping for algorithms that use backpropogation
WARM_UP_TEST = 2000 # warm-up steps in testing (number of times input point is repeated)
N_TRAJ_MVE = 100 # Number of test trajectories generated for Mean-Variance Estimation Sampling
THRESH_PICP = 0.8 # Threshold value for PICP accuracy 
LORENZ_LT = 0.61 # 1 Lyapunov Time for Lorenz 96, F = 8
KS_LT = 11.2  # 1 Lyapunov Time for KS, L = 60

class PointExperimentResult():
    """
    Valid for point estimate (only performing prediction of next point)
    """
    def __init__(self, test_error, details, params = None):
        self.params = params
        self.error = test_error    
        self.details = details
        self.rmse = np.sqrt(np.mean(self.error**2, axis=2))
        self.quantile_list = None
        
    def get_rmse(self):
        return self.rmse
    
    def plot_rmse(self, error_thresh = 1, dt = 0.01):
        if self.quantile_list is None:
            self.quantile_list = np.quantile(self.rmse, q = [0.25,0.50,0.75], axis=0)
            
        L_forecast = self.error.shape[1]

        plt.fill_between(x = np.arange(L_forecast),
                         y1 = self.quantile_list[0,:],
                         y2 = self.quantile_list[2,:],
                         color="red",
                         alpha=0.5)
        plt.plot(np.arange(L_forecast), self.quantile_list[1,:], "b-", lw = 3)
        plt.grid(True)
        plt.axhline(error_thresh, c = "r", ls = "--")
        plt.ylabel("NRMSE")
        plt.yscale("log")
        plt.xlabel("Forecast Horizon (time steps)")                
        plt.show()
        
        first_max = np.argmax(self.quantile_list[1, :] > error_thresh)
        print(f"It takes around t = {first_max * dt:.2f} for mean error to exceed {error_thresh}")

    def get_loss(self, time, dt = 0.01):
        if self.quantile_list is None:
            raise Exception("Quantiles not generated. Run plot_rmse first")
        median = self.quantile_list[1, :]
        if type(time) == float or type(time) == int:
            print(f"Median NRMSE at t = {time}: {median[int(time / dt)]:.3f}")
        else:
            for t in time:
                print(f"Median NRMSE at t = {t}: {median[int(t / dt)]:.3f}")

class PointExperimentResultLyapunov():
    """
    Measures statistics of predictions in terms of lyapunov time
    """
    def __init__(self, test_error, kind, params = None):
        self.params = params
        self.error = test_error    
        self.kind = kind.lower()
        if self.kind == "lorenz":
            self.lt = LORENZ_LT
            self.dt = 0.01
        elif self.kind == "ks":
            self.lt = KS_LT
            self.dt = 0.25
        else:
            raise Exception("Invalid system") 
        self.rmse = np.sqrt(np.mean(self.error**2, axis=2))
        self.quantile_list = None
        
    def get_rmse(self):
        return self.rmse
    
    def plot_rmse(self, error_thresh = 0.5, save_name = False):
        if self.quantile_list is None:
            self.quantile_list = np.quantile(self.rmse, q = [0.25,0.50,0.75], axis=0)
            
        L_forecast = self.error.shape[1]

        plt.fill_between(x = np.arange(L_forecast) * self.dt / self.lt,
                         y1 = self.quantile_list[0,:],
                         y2 = self.quantile_list[2,:],
                         color="red",
                         alpha=0.5)
        plt.plot(np.arange(L_forecast) * self.dt / self.lt, self.quantile_list[1,:], "b-", lw = 3)
        plt.grid(True)
        plt.axhline(error_thresh, c = "r", ls = "--")
        plt.ylabel("NRMSE")
        plt.xlabel("Forecast Horizon (Lyapunov Time)")      
        if save_name:
            plt.savefig(save_name + ".png", facecolor = "white", bbox_inches = "tight")
        plt.show()
        
        first_max = np.argmax(self.quantile_list[1, :] > error_thresh)
        print(f"It takes around {first_max * self.dt / self.lt:.2f} Lyapunov Time for mean error to exceed {error_thresh}")

    def get_loss(self, time = np.array([0.5, 1, 2, 5])):
        if self.quantile_list is None:
            raise Exception("Quantiles not generated. Run plot_rmse first")
        median = self.quantile_list[1, :]
        if type(time) == float or type(time) == int:
            print(f"Median NRMSE at {time} Lyapunov Time: {median[int(time * self.lt / self.dt)]:.3f}")
        else:
            for t in time:
                print(f"Median NRMSE at {t} Lyapunov Time: {median[int(t * self.lt / self.dt)]:.3f}")
                
class LearningRateScheduler():
    def __init__(self):
        self.learning_rate_schedule = np.array([])
        
    def generate_sched(self, n_epoch):
        self.learning_rate_schedule = np.zeros(n_epoch)
        return self.learning_rate_schedule
    
    def plot_sched(self):
        plt.plot(self.learning_rate_schedule)
        plt.title("Learning rate")
        plt.xlabel("Iteration #")
        plt.ylabel("$\eta$")
        plt.show()

class CyclicLearning(LearningRateScheduler):
    """ 
    Cyclical Learning Rate - Exponential Range 
    https://github.com/bckenstler/CLR
    """
    def __init__(self, base_lr, max_lr, step_size, gamma):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        
    def generate_sched(self, n_epoch):
        self.learning_rate_schedule = np.zeros(n_epoch)
        for i in range(n_epoch):
            cycle = np.floor(1 + i / (2 * self.step_size))
            x = np.abs(i / self.step_size - 2 * cycle + 1)
            lr = self.base_lr + ((self.max_lr - self.base_lr)* np.maximum(0, (1 - x)) * self.gamma**i)
            self.learning_rate_schedule[i] = lr
        return self.learning_rate_schedule
    
class WarmupLearning(LearningRateScheduler):
    def __init__(self, base_lr, factor, initial, step_size):
        super().__init__()
        self.base_lr = base_lr
        self.factor = factor
        self.initial = initial
        self.step_size = step_size
        
    def generate_sched(self, n_epoch):
        self.learning_rate_schedule = np.zeros(n_epoch)
        self.learning_rate_schedule[:2] = self.initial
        for i in range(2, n_epoch):
            cycle = i // self.step_size
            self.learning_rate_schedule[i] = self.base_lr * self.factor**cycle
        return self.learning_rate_schedule
        
def plot_predictions(mean_pred, y_test, max_lt, idx = 0, save_name = False):
    fig, ax = plt.subplots(3, 1, figsize = (20, 9))
    
    cbar_range = max(np.max(mean_pred[idx]), np.abs(np.min(mean_pred[idx])))

    ax[0].imshow(mean_pred[idx].T, cmap = "RdYlBu", aspect = "auto")
    ax[0].tick_params(axis='both', which='both', left = False, bottom=False, labelleft = False, labelbottom=False) 
    ax[0].set_title("Predicted")

    ax[1].imshow(y_test[idx].T, cmap = "RdYlBu", aspect = "auto")
    ax[1].tick_params(axis='both', which='both', left = False, bottom=False, labelleft = False, labelbottom=False) 
    ax[1].set_title("Actual")

    im = ax[2].imshow(y_test[idx].T - mean_pred[idx].T, cmap = "RdYlBu", aspect = "auto", extent = [0, max_lt, 0, 40],
                     vmin = -cbar_range, vmax = cbar_range)
    ax[2].tick_params(axis='both', which='both', left = False, labelleft = False) 
    ax[2].set_title("Error")
    ax[2].set_xlabel("Lyapunov Time")

    fig.colorbar(im, ax = ax.ravel().tolist())
    if save_name:
        plt.savefig(save_name + ".png", facecolor = "white", bbox_inches = "tight")
    plt.show()



@jax.jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def save_obj(obj, folder, filename):
    with open(os.path.join(folder, filename), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(filename):
    with open(filename, "rb") as f:
        output = pickle.load(f)
    return output