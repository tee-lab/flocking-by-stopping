from attr import dataclass
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

# Indices for the corresponding rates in the rate array

rng = np.random.default_rng()

@dataclass
class StoppingModel():
    # Rates of the different interactions
    sC : float
    sS : float
    sM : float
    cC : float
    cS : float
    cM : float
    h : float

    def simulate(self, N, T, s_init, phi_init):

        # clock = np.empty(7)
        SC, SS, SM, CC, CS, CM, H = range(7)
        
        # Interaction rates as a numpy array
        r = np.array([self.sC, self.sS, self.sM, self.cC, self.cS, self.cM, self.h])

        timestamps = np.zeros(T)
        s = np.zeros((T, N), dtype=int) # Speed
        e = np.zeros((T, N, 2))         # Orientation (vector)
        timestamps[0] = 0.
        s[0, :] = s_init
        e[0, ...] = np.stack((np.cos(phi_init), np.sin(phi_init))).T

        for t in range(1, T):
            s[t], e[t] = s[t - 1], e[t - 1]
            N_stopped = np.sum(s[t] == 0)
            N_moving = N - N_stopped

            # Interaction propensities
            a = r * np.array([N_moving, N_moving, N_stopped,
                              N_moving * N_moving, N_moving * N_stopped, N_moving * N_stopped,
                              N_moving * N_moving])
            a_tot = np.sum(a)

            next_event_time = rng.exponential(1 / a_tot)
            next_event_type = np.random.choice(7, p = a / a_tot)
            
            if next_event_type in [SC, SS, CC, CS, H]:
                next_event_ind = rng.choice(np.where(s[t, :] == 1)[0])
            else:
                next_event_ind = rng.choice(np.where(s[t, :] == 0)[0])

            if next_event_type == SC:
                theta = rng.uniform(-np.pi, np.pi)
                e[t, next_event_ind, :] = np.stack((np.cos(theta), np.sin(theta)))

            elif next_event_type == SS:
                s[t, next_event_ind] = 0

            elif next_event_type == SM:
                s[t, next_event_ind] = 1
            
            if next_event_type == CC:
                # print(next_event_ind)
                # print(np.where(s[t, :] == 1)[0])
                if np.sum(s[t, :] == 1) <= 1:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                neighbor_id = rng.choice(np.setdiff1d(np.where(s[t, :] == 1)[0], next_event_ind))
                e[t, next_event_ind, :] = e[t, neighbor_id, :]

            elif next_event_type == CS:
                if np.sum(s[t, :] == 0) == 0:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                # neighbor_id = rng.choice(np.where(s[t, :] == 0))
                s[t, next_event_ind] = 0

            elif next_event_type == CM:
                # print(next_event_ind)
                # print(np.where(s[t, :] == 1)[0])
                if np.sum(s[t, :] == 1) <= 1:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                neighbor_id = rng.choice(np.where(s[t, :] == 1)[0])
                s[t, next_event_ind] = 1
                e[t, next_event_ind, :] = e[t, neighbor_id, :]

            elif next_event_type == H:
                # print(next_event_ind)
                # print(np.where(s[t, :] == 1)[0])
                if np.sum(s[t, :] == 1) == 0:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                neighbor_id = rng.choice(np.where(s[t, :] == 1)[0])
                # print(next_event_ind)
                # print(neighbor_id)
                s[t, next_event_ind] = (np.dot(e[t, next_event_ind, :], e[t, neighbor_id, :]) > 0)

            timestamps[t] = timestamps[t - 1] + next_event_time
        
        steady_state = slice(int(T * 0.5), None)
        s_ss = s[steady_state]
        e_ss = e[steady_state]
        t_ss = timestamps[steady_state]

        e_ss = s_ss[:, :, None] * e_ss
        m_ss = np.mean(e_ss, axis=1)
        mx_ss = m_ss[:, 0]
        my_ss = m_ss[:, 1]
        modm_ss = np.linalg.norm(m_ss, axis=1)

        return mx_ss, my_ss, modm_ss
        
@dataclass
class StoppingModel_Timeseries():
    # Rates of the different interactions
    sC : float
    sS : float
    sM : float
    cC : float
    cS : float
    cM : float
    h : float

    def simulate(self, N, T, s_init, phi_init):

        # clock = np.empty(7)
        SC, SS, SM, CC, CS, CM, H = range(7)
        
        # Interaction rates as a numpy array
        r = np.array([self.sC, self.sS, self.sM, self.cC, self.cS, self.cM, self.h])

        timestamps = np.zeros(T)
        s = np.zeros((T, N), dtype=int) # Speed
        e = np.zeros((T, N, 2))         # Orientation (vector)
        timestamps[0] = 0.
        s[0, :] = s_init
        e[0, ...] = np.stack((np.cos(phi_init), np.sin(phi_init))).T

        for t in range(1,T) :
            s[t], e[t] = s[t - 1], e[t - 1]
            N_stopped = np.sum(s[t] == 0)
            N_moving = N - N_stopped

            # Interaction propensities
            a = r * np.array([N_moving, N_moving, N_stopped,
                              N_moving * N_moving, N_moving * N_stopped, N_moving * N_stopped,
                              N_moving * N_moving])
            a_tot = np.sum(a)

            next_event_time = rng.exponential(1 / a_tot)
            next_event_type = np.random.choice(7, p = a / a_tot)
            
            if next_event_type in [SC, SS, CC, CS, H]:
                next_event_ind = rng.choice(np.where(s[t, :] == 1)[0])
            else:
                next_event_ind = rng.choice(np.where(s[t, :] == 0)[0])

            if next_event_type == SC:
                theta = rng.uniform(-np.pi, np.pi)
                e[t, next_event_ind, :] = np.stack((np.cos(theta), np.sin(theta)))

            elif next_event_type == SS:
                s[t, next_event_ind] = 0

            elif next_event_type == SM:
                s[t, next_event_ind] = 1
            
            if next_event_type == CC:
                # print(next_event_ind)
                # print(np.where(s[t, :] == 1)[0])
                if np.sum(s[t, :] == 1) <= 1:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                neighbor_id = rng.choice(np.setdiff1d(np.where(s[t, :] == 1)[0], next_event_ind))
                e[t, next_event_ind, :] = e[t, neighbor_id, :]

            elif next_event_type == CS:
                if np.sum(s[t, :] == 0) == 0:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                # neighbor_id = rng.choice(np.where(s[t, :] == 0))
                s[t, next_event_ind] = 0

            elif next_event_type == CM:
                # print(next_event_ind)
                # print(np.where(s[t, :] == 1)[0])
                if np.sum(s[t, :] == 1) <= 1:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                neighbor_id = rng.choice(np.where(s[t, :] == 1)[0])
                s[t, next_event_ind] = 1
                e[t, next_event_ind, :] = e[t, neighbor_id, :]

            elif next_event_type == H:
                # print(next_event_ind)
                # print(np.where(s[t, :] == 1)[0])
                if np.sum(s[t, :] == 1) == 0:
                    timestamps[t] = timestamps[t - 1] + next_event_time
                    continue
                neighbor_id = rng.choice(np.where(s[t, :] == 1)[0])
                # print(next_event_ind)
                # print(neighbor_id)
                s[t, next_event_ind] = (np.dot(e[t, next_event_ind, :], e[t, neighbor_id, :]) > 0)

            timestamps[t] = timestamps[t - 1] + next_event_time
        
        e = s[:, :, None]*e
        m = np.mean(e, axis=1)
        modm = np.linalg.norm(m, axis=1)
        v = np.mean(s, axis=1)
        return timestamps, v, modm

if __name__ == '__main__':
    N = 2000
    iterations = 1000
    T_steps = 100000  
    H = [0, 2.0]
    
    parameters = []
    for h in H:
        model = StoppingModel(sC=0.2, sS=0.2, sM=0.2, cC=0.2, cS=0.2, cM=2, h=h)
        for _ in range(iterations):
            s_init = rng.choice([0, 1], size=N)
            phi_init = rng.uniform(-np.pi, np.pi, size=N)
            parameters.append((model, s_init, phi_init, h))
   
    pooled_data = {h: {'mx': [], 'my': [], 'modm': []} for h in H}

    with parallel_config(backend='loky', n_jobs=-1):
        results_gen = Parallel(return_as="generator")(
            delayed(model.simulate)(N, T_steps, s_init, phi_init) 
            for model, s_init, phi_init, h in parameters
        )

        for i, (mx, my, modm) in enumerate(tqdm(results_gen, total=len(parameters))):
            h_current = parameters[i][3]
            pooled_data[h_current]['mx'].extend(mx)
            pooled_data[h_current]['my'].extend(my)
            pooled_data[h_current]['modm'].extend(modm)

    fig, axes = plt.subplots(len(H), 3, figsize=(20, 5 * len(H)),constrained_layout=True)

    for row_idx, h in enumerate(H):
        data = pooled_data[h]
        
        axes[row_idx, 1].hist(data['modm'], bins=100, range=(0,1), density=True, color='steelblue', edgecolor='white',histtype='stepfilled',rasterized=True)
        axes[row_idx, 1].set_title(r'$\mathbf{|m|}$ Histogram',fontsize=21)

        axes[row_idx,1].set_xlabel(r'Polarization $\mathbf{|m|}$',fontsize=18)
        axes[row_idx,1].set_ylabel('Density',fontsize=18)
        
        _, _, _, im = axes[row_idx, 2].hist2d(
        data['mx'], data['my'], 
        bins=100, 
        range=[[-1, 1], [-1, 1]], density=True,
        cmap='plasma',rasterized=True)
        cbar = fig.colorbar(im, ax=axes[row_idx, 2])
        cbar.ax.tick_params(labelsize=14)
        axes[row_idx,2].set_title(r'$\mathbf{m}$ Histogram',fontsize=21)

        axes[row_idx,2].set_xlabel(r'$m_{x}$',fontsize=18)
        axes[row_idx,2].set_ylabel(r'$m_{y}$',fontsize=18)
   
        axes[row_idx,0].tick_params(axis='both', which='major', labelsize=14)
        axes[row_idx,1].tick_params(axis='both', which='major', labelsize=14)
        axes[row_idx,2].tick_params(axis='both', which='major', labelsize=14)
        
    for row_idx,h in enumerate(H):
        model = StoppingModel_Timeseries(sC=.2, sS=.2, sM=.2, cC=.2, cS=.2, cM=2, h=h)
        t, v, m = model.simulate(N=N, T=1000000, 
                             s_init = rng.choice([0, 1], size=N),
                             phi_init=rng.uniform(-np.pi, np.pi, size=N))
        axes[row_idx,0].plot(t,v,label='Speed',color='darkorange')
        axes[row_idx,0].plot(t,m,label='Polarization',color='steelblue')
        #axes[row_idx,0].set_xlim(0, 1.1)
        axes[row_idx,0].set_ylim(0, 1.1)
        axes[row_idx,0].set_title(f"Sample Trajectory",fontsize=21)
        axes[row_idx,0].set_xlabel('Time', fontsize=18)
        axes[row_idx,0].set_ylabel('State', fontsize=18)
        axes[row_idx,0].legend(loc='lower right', fontsize=15)
        axes[row_idx,0].grid(True)
    plt.show()