import numpy as np
import sim
import utils
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


OBS_CENTER = np.array([[0.9, 0], [0.25, 0.5], [-0.3, 0.5], [-1, 0.1], [0.3, -0.8]])
OBS_RADIUS = np.array([0.5, 0.3, 0.2, 0.5, 0.4])
SPH_RADIUS = 0.02

FK_Solver = utils.FKSolver() # forward kinematics solver


########## Task 1: Particle Weights Calculation ##########

def dist_to_closest_obs(x, y):
    """
    Find the distance between the stick's sphere end centered at (x, y) 
    to its closest cylinder obstacle.
    args:       x: The x-coordinate of the spherical end.
                y: The y-coordinate of the spherical end. 
    returns: dist: The distance to the closest obstacle.
    """
    ########## TODO ##########
    dist = 0.0  # initialize distance (will store the closest distance)
    min_abs = float('inf')  # track the smallest absolute distance to any obstacle

    for (cx, cy), r in zip(OBS_CENTER, OBS_RADIUS):
        # compute surface-to-surface distance:
        # center distance minus (cylinder radius + sphere radius)
        d = np.sqrt((x - cx)**2 + (y - cy)**2) - (r + SPH_RADIUS)

        # we want the obstacle whose distance is closest to zero
        # (either touching, penetrating, or nearest outside)
        if abs(d) < min_abs:
            min_abs = abs(d)  # update closest absolute distance
            dist = d          # store the signed distance (can be negative if penetrating)
    ##########################
    return dist

def cal_weights(particles, obv, sigma=0.05):
    """
    Calculate the weights for particles based on the given observation.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
               sigma: The standard deviation of the Gaussian distribution
                      for calculating likelihood (default: 0.05).
    returns: weights: The weights of all particles.
                      Type: numpy.ndarray of shape (# of particles,)
    """
    ########## TODO ##########
    weights = None  # placeholder (will store normalized particle weights)
    N = particles.shape[0]  # number of particles
    likelihoods = np.zeros(N)  # store unnormalized likelihood for each particle

    # compute end-effector position (in robot base frame) from observation (joint angles)
    x_ee, y_ee = FK_Solver.forward_kinematics_2d(obv)

    for i in range(N):
        x_i, y_i, theta_i = particles[i]  # particle state: base position and orientation

        # transform end-effector position from robot base frame to world frame
        x_world = x_i + np.cos(theta_i) * x_ee - np.sin(theta_i) * y_ee
        y_world = y_i + np.sin(theta_i) * x_ee + np.cos(theta_i) * y_ee

        # compute distance between end-effector and closest obstacle
        d = dist_to_closest_obs(x_world, y_world)

        # compute likelihood using Gaussian model (higher when d is close to 0)
        likelihoods[i] = np.exp(-0.5 * (d / sigma) ** 2)

    # normalize likelihoods to obtain valid probability distribution (sum to 1)
    weights = likelihoods / np.sum(likelihoods)
    ##########################
    return weights

def most_likely_particle(particles, obv):
    """
    Find the most likely particle.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
    returns:     idx: The index of the most likely particle
                      Type: int
    """
    ########## TODO ##########
    idx = 0  # initialize index (will store index of most likely particle)
    weights = cal_weights(particles, obv)  # compute weights for all particles based on observation
    idx = np.argmax(weights)  # find the index of the particle with the highest weight
    return idx  # return the index of the most likely particle
    ##########################
    return idx


########## Task 2: Particle Filter ##########

def particle_filter(panda_sim, obvs, num_particles, sigma=0.05, delta=0.01, plot=True):
    """
    The Particle Filtering algorithm. 
    args:    panda_sim: The instance of the simulation. 
                        Type: sim.PandaSim (provided)
                  obvs: The given observations (the robot's joint angles)
                        Type: numpy.ndarray of shape (# of observations, 7)
         num_particles: The number of particles. 
                        Type: int
                 sigma: The standard deviation of the Gaussian distribution
                        for calculating likelihood (default: 0.05).
                 delta: The scale of the Gaussian for perturbing particles.
                        (default: 0.01)
                  plot: Whether to enable particle plot (True or False).
    returns:       est: The estimate of the pose of the robot base. 
                        Type: numpy.ndarray of shape (3,)
    """
    # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    for obv in obvs:
        panda_sim.set_joint_values(obv)

        ########## TODO ##########
        # inside particle_filter loop, replace the TODO block with this:

        # update weights based on current observation
        weights = cal_weights(particles, obv, sigma)

        # resample particles according to weights
        indices = np.random.choice(len(particles), size=len(particles), p=weights)
        particles = particles[indices]

        # add Gaussian noise to maintain diversity
        noise = np.random.normal(0, delta, size=particles.shape)
        particles = particles + noise
        ##########################

        # plot the particles in the visualization
        if plot:
            utils.plot_pf(ax, particles, panda_sim.loc)
            plt.pause(0.01)
    est = particles.mean(0)
    return est


########## Task 3: Generate Observations Online ##########

def get_one_obv(panda_sim):
    """
    Control the robot in simulation to obtain an observation.
    args: panda_sim: The instance of the simulation. 
                     Type: sim.PandaSim (provided)
    returns:    obv: One observation found by this function.
                     Type: numpy.ndarray of shape (7,)
    """
    ########## TODO ##########

def get_one_obv(panda_sim):
    obv = None

    # persistent velocity (NOT resetting each loop)
    v = np.array([0.05, 0.02, -0.05, 0, 0, 0])

    step = 0
    while obv is None:
        panda_sim.execute(v)

        if panda_sim.is_collision():
            # bounce (change direction)
            v = -v
            print("[collision] reversing direction")

        if panda_sim.is_touch():
            print(f"[TOUCH at step {step}]")
            jpos, _, _ = panda_sim.get_joint_states()
            obv = np.array(jpos[:7])

        step += 1

    return obv

    ##########################
    return obv

def particle_filter_online(panda_sim, num_particles, sigma=0.05, delta=0.01, plot=True):
    """
    The online Particle Filtering algorithm. 
    args:     panda_sim: The instance of the simulation. 
                         Type: sim.PandaSim (provided)
          num_particles: The number of particles. 
                         Type: int
                  sigma: The standard deviation of the Gaussian distribution
                         for calculating likelihood (default: 0.05).
                  delta: The scale of the Gaussian for perturbing particles.
                         (default: 0.01)
                   plot: Whether to enable particle plot (True or False).
    returns:        est: The estimate of the pose of the robot base. 
                         Type: numpy.ndarray of shape (3,)
    """
    # # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    ########## TODO ##########
    num_iters = 100
    for it in range(num_iters):
        print(f"\n[PF] iteration {it}")

        obv = get_one_obv(panda_sim)
        print("[PF] got observation")

        weights = cal_weights(particles, obv, sigma)
        print(f"[PF] weights stats: min={weights.min()}, max={weights.max()}")

        indices = np.random.choice(len(particles), size=len(particles), p=weights)
        particles = particles[indices]
        print("[PF] resampled")

        noise = np.random.normal(0, delta, size=particles.shape)
        particles = particles + noise
        print("[PF] added noise")

        if plot:
            utils.plot_pf(ax, particles, panda_sim.loc)
            plt.pause(0.01)
    ##########################
    est = particles.mean(0)
    return est
