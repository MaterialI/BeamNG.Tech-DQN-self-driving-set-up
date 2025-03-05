import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import AdvancedIMU
import json

# -------------------------
# Set up BeamNG.tech session
# -------------------------
# Update these parameters as needed:
BEE_MOTOR_PORT = 64256
with open("paths.json", "r") as file:
    paths = json.load(file)

BEAMNG_HOME = paths.get("globalPath")

# Create a BeamNGpy instance (using localhost)
bng = BeamNGpy("localhost", 64256, home=BEAMNG_HOME)
bng.open(launch=True)

scenario = Scenario("hirochi_raceway", "camera_streaming")

# Mesh sensor configuration

ego = Vehicle("ego", model="etk800", color="White")

# Create and attach the mesh sensor

# In your main loop, poll the sensor data

scenario.add_vehicle(
    ego,
    pos=(-402.42, 248.61, 26),
    rot_quat=(0.01671021, 0.00486505, -0.29160145, 0.95639205),
)


# Generate and load the scenario
scenario.make(bng)
bng.load_scenario(scenario)
bng.start_scenario()


# -------------------------
# Attach the Advanced IMU sensor
# -------------------------

imu = AdvancedIMU("advanced_imu", bng, ego, pos=(0, -1, 1.5), is_send_immediately=True)
# Note: Check your BeamNG.tech documentation for the exact configuration.
# Here we assume that the sensor ID is 'advanced_imu'

time.sleep(1)  # allow some time for sensor initialization

# -------------------------
# Set up real-time plotting (5-second rolling window)
# -------------------------
window_seconds = 10.0

# Deques to hold timestamps and sensor data
times_deque = deque()
acc_x_deque = deque()
acc_y_deque = deque()
acc_z_deque = deque()
prod_deque = deque()

# Create matplotlib figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Prepare lines for acceleration axes
(line_x,) = ax1.plot([], [], label="Acc X", color="tab:blue")
(line_y,) = ax1.plot([], [], label="Acc Y", color="tab:orange")
(line_z,) = ax1.plot([], [], label="Acc Z", color="tab:green")
ax1.set_title("Accelerometer Data (m/s²)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Acceleration (m/s²)")
ax1.legend()
ax1.grid(True)

# Prepare line for product of accelerations
(line_prod,) = ax2.plot([], [], label="Acc Product", color="tab:red")
ax2.set_title("Product of Accelerations (m/s⁶)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Product")
ax2.legend()
ax2.grid(True)


# -------------------------
# Update function for matplotlib animation
# -------------------------
def update_plot(frame):
    current_time = time.time()

    # Poll sensor data from the Advanced IMU sensor
    sensor_data = imu.poll()
    accRaw = sensor_data.get("accRaw", None)

    if sensor_data is not None:

        # Extract acceleration values (adjust keys if needed)
        acc_x = sensor_data.get("accX", 0.0)
        acc_y = sensor_data.get("accY", 0.0)
        acc_z = sensor_data.get("accZ", 0.0)
        # Calculate product of all three axes (could be negative if values have sign)
        prod = acc_x * acc_y * acc_z

        # Append new sensor readings with current timestamp
        times_deque.append(current_time)
        acc_x_deque.append(acc_x)
        acc_y_deque.append(acc_y)
        acc_z_deque.append(acc_z)
        prod_deque.append(prod)

    # Remove data older than window_seconds
    while times_deque and (current_time - times_deque[0] > window_seconds):
        times_deque.popleft()
        acc_x_deque.popleft()
        acc_y_deque.popleft()
        acc_z_deque.popleft()
        prod_deque.popleft()

    # Use relative time (starting at zero)
    if times_deque:
        t0 = times_deque[0]
        rel_times = np.array([t - t0 for t in times_deque])
    else:
        rel_times = np.array([])

    # Update acceleration lines
    line_x.set_data(rel_times, list(acc_x_deque))
    line_y.set_data(rel_times, list(acc_y_deque))
    line_z.set_data(rel_times, list(acc_z_deque))
    ax1.relim()
    ax1.autoscale_view()

    # Update product line
    line_prod.set_data(rel_times, list(prod_deque))
    ax2.relim()
    ax2.autoscale_view()

    return line_x, line_y, line_z, line_prod


# Create animation: update every 100ms (adjust interval if needed)
ani = animation.FuncAnimation(fig, update_plot, interval=100)

# -------------------------
# Start the real-time plot
# -------------------------
plt.tight_layout()
plt.show()

# When done, close the scenario properly:
bng.close()
