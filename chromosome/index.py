import time
import math
import numpy as np
import matplotlib.pyplot as plt

SLOPE_LENGTH = 37.5 / 100
SLOPE_HEIGHT = 3.5 / 100
SLOPE_WIDTH = 11 / 100
G = 9.81  # ms^(-2)

THETA = np.arctan(SLOPE_HEIGHT / SLOPE_LENGTH)

MU_R = 0.01
MU_G = 0.01
MASS = 0.1  # kg


MAGNET_CAPTURE_RANGE = 1.0 / 100  # 1cm capture range
MAX_MAGNETIC_FORCE = 50  # N

TOTAL_TIME = 5  # s


class Chromosome:
    def __init__(self, time_step):
        self.dt = time_step

        # Create base positions
        base_positions = np.linspace(0, SLOPE_LENGTH, 4)

        max_offset = 0.02
        offsets = np.random.uniform(-max_offset, max_offset, 4)

        # Add offsets while ensuring no position goes beyond bounds
        slope_positions = np.clip(base_positions + offsets, 0, SLOPE_LENGTH)

        # Create two sets of y coordinates for parallel positions
        y_coords = np.array([-0.01 + 0.055, 0.01 + 0.055])

        # Using meshgrid to create all combinations of slope positions and y coordinates
        slope_grid, y_grid = np.meshgrid(slope_positions, y_coords)

        self.magnet_positions = np.column_stack(
            (
                slope_grid.flatten() * np.cos(THETA),  # x coordinates
                y_grid.flatten(),  # y coordinates
                SLOPE_HEIGHT - slope_grid.flatten() * np.sin(THETA),  # z coordinates
            )
        )

    def to_dict(self):
        return {
            "coords": [
                (float(x), float(y)) for (x, y) in self.translate_to_real_coords()
            ]
        }

    def translate_to_real_coords(self):
        real_coords = []
        for pos in self.magnet_positions:
            x = pos[0]  # cartesian x
            z = SLOPE_HEIGHT - pos[2]  # height difference from start

            # Calculate distance along slope using Pythagorean theorem
            slope_distance = np.sqrt(x**2 + z**2)

            # Keep the y coordinate as is
            real_coords.append((slope_distance, pos[1]))

        return real_coords

    def magnetic_force(self, position):
        # Convert position to slope coordinates
        x, y, z = position

        total_force_parallel = 0
        total_grip_effect = 0

        for magnet_pos in self.magnet_positions:
            # Only consider vertical (y) distance for magnetic effect
            dy = abs(y - magnet_pos[1])  # Absolute vertical distance to magnet

            # Check if ball is roughly above/below the magnet in x-z plane
            dx = abs(x - magnet_pos[0])

            dy = round(dy, 2)

            # Only apply magnetic force if ball is roughly above/below magnet
            if dx <= 0.025 and dy <= MAGNET_CAPTURE_RANGE:
                normalized_distance = dy / MAGNET_CAPTURE_RANGE
                force_magnitude = MAX_MAGNETIC_FORCE * (
                    1 / (1 + normalized_distance * 5)
                )
                total_grip_effect += force_magnitude / MAX_MAGNETIC_FORCE

        # Normalize grip effect
        total_grip_effect = min(total_grip_effect, 1.0)

        # No direct parallel force - magnets only affect normal force
        total_force_parallel = 0

        return total_force_parallel, total_grip_effect

    def calculate_slope_motion(self):
        positions = []
        velocities = []

        # Initial conditions
        pos = np.array([0, 0.055, SLOPE_HEIGHT])
        vel = np.array([0.0, 0.0, 0.0])
        t = 0

        while pos[2] > 0 and t < TOTAL_TIME:
            # Store current state
            positions.append(pos.copy())
            velocities.append(vel.copy())

            # Calculate parallel component of gravity
            g_parallel = G * np.sin(THETA)

            # Calculate magnetic force
            mag_force, grip_effect = self.magnetic_force(pos)

            # Normal force (perpendicular to slope)
            N = MASS * G * np.cos(THETA) + mag_force * np.sin(THETA)

            base_friction = MU_R * N if N > 0 else 0
            grip_friction = base_friction * (1 + grip_effect * 5)

            # Net acceleration (reduced by grip effect)
            a_parallel = g_parallel + (mag_force / MASS)
            a_parallel *= 1 - grip_effect * 0.5

            # Net acceleration parallel to slope
            a_parallel -= grip_friction / MASS

            # Convert parallel acceleration to x and z components
            ax = a_parallel * np.cos(THETA)
            az = -a_parallel * np.sin(THETA)

            # Update velocity (reduced by grip effect)
            vel_reduction = 1 - grip_effect * 0.3  # Velocity damping from grip
            vel_new = np.array(
                [
                    vel[0] + ax * self.dt,
                    0,  # y-velocity remains 0
                    vel[2] + az * self.dt,
                ]
            )

            # Calculate average velocity for position update
            vel_avg = 0.5 * (vel + vel_new)

            # Update position
            pos_new = pos + vel_avg * self.dt

            # Check if we've reached the end of the slope
            if pos_new[0] > SLOPE_LENGTH * np.cos(THETA):
                # Linearly interpolate to find exact exit point
                t_exit = (SLOPE_LENGTH * np.cos(THETA) - pos[0]) / (pos_new[0] - pos[0])
                pos_new = pos + vel_avg * (self.dt * t_exit)
                break

            # Check if position is still on slope surface
            expected_z = SLOPE_HEIGHT - pos_new[0] * np.tan(THETA)
            if abs(pos_new[2] - expected_z) > 1e-6:
                # Project back onto slope
                pos_new[2] = expected_z

            pos = pos_new
            vel = vel_new
            t += self.dt

        self.slope_positions = np.array(positions)
        self.t_slope = t

        # Final velocity components
        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]

        return self.slope_positions, self.vx, self.vy, self.vz

    def calculate_projectile_motion(self):
        # Initial positions (where slope ended)
        x0 = self.slope_positions[-1, 0]
        y0 = self.slope_positions[-1, 1]
        z0 = self.slope_positions[-1, 2]

        # Time for projectile to reach ground
        # Using quadratic formula to solve: z0 + vz*t - 0.5*g*t^2 = 0
        a = -0.5 * G
        b = self.vz
        c = z0

        # Time to reach ground
        self.t_air = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t = np.arange(0, self.t_air, self.dt)

        # Projectile motion equations
        x = x0 + self.vx * t
        y = y0 + self.vy * t
        z = z0 + self.vz * t - 0.5 * G * t**2

        self.projectile_positions = np.column_stack((x, y, z))

        # Calculate impact position and velocity
        self.impact_position = np.array(
            [x0 + self.vx * self.t_air, y0, 0]  # z = 0 at ground level
        )

        # Calculate impact velocity (horizontal component remains the same)
        self.vx_impact = self.vx
        self.vz_impact = self.vz - G * self.t_air

        return self.projectile_positions, self.impact_position

    def calculate_ground_motion(self):
        # Time after impact
        remaining_time = TOTAL_TIME - (self.t_slope + self.t_air)
        if remaining_time <= 0:
            return np.array([]).reshape(0, 3), self.impact_position

        t = np.arange(0, remaining_time, self.dt)

        # Calculate deceleration due to friction on ground
        a_friction = -MU_G * G

        # Calculate velocity and position along ground
        vx = self.vx_impact + a_friction * t  # decreasing velocity

        # Set velocity to 0 once the ball stops
        stop_idx = np.where(vx <= 0)[0]
        if len(stop_idx) > 0:
            t = t[: stop_idx[0]]
            vx = vx[: stop_idx[0]]

        # Calculate positions
        x = self.impact_position[0] + np.trapz(vx, t)  # integrate velocity
        y = np.zeros_like(t) + self.impact_position[1]
        z = np.zeros_like(t)  # stays at ground level

        ground_positions = np.column_stack((x * np.ones_like(t), y, z))

        # Final position is the last position calculated
        self.final_position = (
            ground_positions[-1] if len(ground_positions) > 0 else self.impact_position
        )

        return ground_positions, self.final_position

    def simulate(self):
        # Calculate slope motion
        slope_pos, vx, vy, vz = self.calculate_slope_motion()

        # Calculate projectile motion
        proj_pos, impact_pos = self.calculate_projectile_motion()

        # Calculate ground motion
        ground_pos, final_pos = self.calculate_ground_motion()

        self.all_positions = np.vstack(
            (
                slope_pos,
                proj_pos,
                ground_pos if len(ground_pos) > 0 else impact_pos.reshape(1, 3),
            )
        )
        self.final_position = final_pos

        actual_time = self.t_slope + self.t_air
        if len(ground_pos) > 0:
            actual_time += len(ground_pos) * self.dt

    def fitness_function(self):
        self.simulate()
        fitness = 1 / (1 + np.abs(0 - self.final_position[0]))
        return fitness

    def visualize(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot magnets as markers on the slope
        for magnet_pos in self.magnet_positions:
            ax.scatter(
                magnet_pos[0],  # x coordinate
                magnet_pos[1],  # y coordinate
                magnet_pos[2],  # z coordinate
                color="red",  # or any color you prefer
                marker="s",  # square marker for magnets
                s=100,  # size of marker
                alpha=0.7,  # slight transparency
            )

        # Add a range indicator for each magnet (optional)
        for magnet_pos in self.magnet_positions:
            # Create a circle or sphere to show magnet range
            theta = np.linspace(0, 2 * np.pi, 20)
            x = magnet_pos[0] + MAGNET_CAPTURE_RANGE * np.cos(theta)
            y = magnet_pos[1] + MAGNET_CAPTURE_RANGE * np.sin(theta)
            z = np.ones_like(theta) * magnet_pos[2]
            ax.plot(x, y, z, "r--", alpha=0.3)  # Dashed line showing range

        # Plot slope
        X = np.array([0, SLOPE_LENGTH])
        Y = np.array([0, SLOPE_WIDTH])
        X, Y = np.meshgrid(X, Y)
        Z = SLOPE_HEIGHT - X * np.tan(THETA)
        ax.plot_surface(X, Y, Z, alpha=0.3, color="gray", label="Slope")

        # Plot trajectory on slope (blue)
        ax.plot(
            self.slope_positions[:, 0],
            self.slope_positions[:, 1],
            self.slope_positions[:, 2],
            "b-",
            label="Slope Motion",
        )

        # Plot projectile trajectory (red)
        ax.plot(
            self.projectile_positions[:, 0],
            self.projectile_positions[:, 1],
            self.projectile_positions[:, 2],
            "r--",
            label="Projectile Motion",
        )

        # Plot ground motion (green)
        if len(self.all_positions) > len(self.slope_positions) + len(
            self.projectile_positions
        ):
            ground_motion = self.all_positions[
                len(self.slope_positions) + len(self.projectile_positions) :
            ]
            ax.plot(
                ground_motion[:, 0],
                ground_motion[:, 1],
                ground_motion[:, 2],
                "g-",
                label="Ground Motion",
            )

        # Mark key points
        ax.scatter(0, 0.055, SLOPE_HEIGHT, color="green", s=100, label="Start")
        ax.scatter(
            self.slope_positions[-1, 0],
            self.slope_positions[-1, 1],
            self.slope_positions[-1, 2],
            color="yellow",
            s=100,
            label="Leave Slope",
        )
        ax.scatter(
            self.final_position[0],
            self.final_position[1],
            self.final_position[2],
            color="red",
            s=100,
            label="Final Position",
        )

        # Add ground plane
        ground_x_limit = float(self.final_position[0]) + 0.1
        X_ground, Y_ground = np.meshgrid(
            np.array([0, ground_x_limit]), np.array([0, SLOPE_WIDTH])
        )
        Z_ground = np.zeros_like(X_ground)
        ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color="green")

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Ball Motion Simulation (Mass: {MASS} kg)")

        # Add legend
        ax.legend()

        # Adjust view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.savefig(f"results/{time.time()}.png")

        # Print detailed results
        print("\nSimulation Results:")
        print(f"Time on slope: {self.t_slope:.3f} seconds")
        print(f"Time in air: {self.t_air:.3f} seconds")
        print(f"Total time: {TOTAL_TIME:.3f} seconds")
        print("\nFinal Position:")
        print(f"X (horizontal): {self.final_position[0]:.3f} m")
        print(f"Y (width): {self.final_position[1]:.3f} m")
        print(f"Z (height): {self.final_position[2]:.3f} m")
        print(f"\nTotal horizontal distance: {self.final_position[0]:.3f} m")
        plt.close()
