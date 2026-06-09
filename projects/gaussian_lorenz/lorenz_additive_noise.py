import numpy as np
from matplotlib import pyplot as plt

# Lorenz parameters - I chose the same as in Sindy-paper
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Initial condition I chose it being consistent with the figure caption in the paper.
x0 = np.array([-8.0, 7.0, 27.0])

# Time grid for training data.
dt = 0.002
t_train = np.arange(0.0, 25.0 + dt, dt)

diffusion_strength = np.array([2.5, 2.5, 2.5])

dt_stoch = 0.01
t_stoch = np.arange(0.0, 3.0 + dt_stoch, dt_stoch)



def lorenz63_drift(state, sigma=sigma, beta=beta, rho=rho):
    """
    Return the deterministic Lorenz-63 drift.

    The function is vectorized over the leading dimensions of `state`.
    """
    state = np.asarray(state)
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]

    return np.stack(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        axis=-1,
    )


def sample_initial_conditions(n_samples, rng):
    """
    Draw initial conditions in a box covering a representative part of the attractor.
    """
    return np.column_stack(
        [
            rng.uniform(-12.0, 12.0, size=n_samples),
            rng.uniform(-15.0, 15.0, size=n_samples),
            rng.uniform(10.0, 35.0, size=n_samples),
        ]
    )


def simulate_stochastic_lorenz63_em(x0, t_grid, diffusion=diffusion_strength, rng=None):
    """
    Simulate the additive-noise stochastic Lorenz-63 system with Euler-Maruyama.

    Parameters
    ----------
    x0 : array, shape (3,)
        Initial condition.
    t_grid : array, shape (m,)
        Uniform time grid.
    diffusion : array, shape (3,)
        Diagonal diffusion coefficients.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    X : array, shape (m, 3)
        Simulated trajectory.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = t_grid[1] - t_grid[0]
    sqrt_dt = np.sqrt(dt)

    X = np.zeros((len(t_grid), 3), dtype=float)
    X[0] = x0

    for k in range(len(t_grid) - 1):
        drift = lorenz63_drift(X[k])
        noise = diffusion * sqrt_dt * rng.standard_normal(3)
        X[k + 1] = X[k] + drift * dt + noise

    return X

def generate_trajectories(n_train_trajectories = 3, n_test_trajectories = 2, random_seed_stoch = 123):
    # Generate training and testing trajectories.

    rng_ic = np.random.default_rng(random_seed_stoch)
    initial_conditions = sample_initial_conditions(
        n_train_trajectories + n_test_trajectories,
        rng_ic,
    )

    trajectories = []
    for i, x0_i in enumerate(initial_conditions):
        rng_i = np.random.default_rng(random_seed_stoch + 1000 + i)
        Xi = simulate_stochastic_lorenz63_em(
            x0_i,
            t_stoch,
            diffusion=diffusion_strength,
            rng=rng_i,
        )
        trajectories.append(Xi)

    trajectories = np.asarray(trajectories)

    X_train_traj = trajectories[:n_train_trajectories]
    X_test_traj = trajectories[n_train_trajectories:]

    return X_train_traj, X_test_traj

def create_dataset_csv():
    pass

def main_visualization():
    n_train_trajectories = 3
    X_train_traj, _ = generate_trajectories(n_train_trajectories, 0)
    # Visualize a few stochastic trajectories.

    fig = plt.figure(figsize=(13, 4))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for i in range(n_train_trajectories):
        ax1.plot(
            X_train_traj[i, :, 0],
            X_train_traj[i, :, 1],
            X_train_traj[i, :, 2],
            lw=0.9,
            alpha=0.9,
            label=f"train traj {i + 1}",
        )
    ax1.set_title("Stochastic Lorenz-63 training trajectories")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t_stoch, X_train_traj[0, :, 0], label="x", lw=1.0)
    ax2.plot(t_stoch, X_train_traj[0, :, 1], label="y", lw=1.0)
    ax2.plot(t_stoch, X_train_traj[0, :, 2], label="z", lw=1.0)
    ax2.set_title("One stochastic training trajectory")
    ax2.set_xlabel("t")
    ax2.set_ylabel("state value")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('stoch_lorenz_070426.pdf')
    plt.show()

if __name__ == '__main__':
    main_visualization()