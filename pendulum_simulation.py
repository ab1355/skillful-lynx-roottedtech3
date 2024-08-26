import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_ode(y, t, L, g):
    theta, omega = y
    dydt = [omega, -g/L * np.sin(theta)]
    return dydt

def simulate_pendulum(L, g, theta0, omega0, tmax, dt):
    t = np.arange(0, tmax, dt)
    y0 = [theta0, omega0]
    sol = odeint(pendulum_ode, y0, t, args=(L, g))
    return t, sol

def plot_results(t, sol, L, g):
    theta = sol[:, 0]
    omega = sol[:, 1]

    # Calculate energy
    E = 0.5 * L**2 * omega**2 + g * L * (1 - np.cos(theta))

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, theta, 'b', label='θ(t)')
    plt.plot(t, omega, 'g', label='ω(t)')
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad) / Angular velocity (rad/s)')
    plt.title('Pendulum Motion')

    plt.subplot(2, 1, 2)
    plt.plot(t, E, 'r', label='Total Energy')
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy Conservation')

    plt.tight_layout()
    plt.savefig('pendulum_simulation.png')
    print("Plot saved as 'pendulum_simulation.png'")

def main():
    L = 1.0  # Length of pendulum (m)
    g = 9.81  # Acceleration due to gravity (m/s^2)
    theta0 = np.pi/4  # Initial angle (45 degrees)
    omega0 = 0.0  # Initial angular velocity
    tmax = 10.0  # Maximum simulation time (s)
    dt = 0.01  # Time step (s)

    t, sol = simulate_pendulum(L, g, theta0, omega0, tmax, dt)
    plot_results(t, sol, L, g)

    print("Pendulum simulation completed.")
    print(f"Maximum angle: {np.max(np.abs(sol[:, 0])):.2f} rad")
    print(f"Maximum angular velocity: {np.max(np.abs(sol[:, 1])):.2f} rad/s")

if __name__ == "__main__":
    main()