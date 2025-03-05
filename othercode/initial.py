import multiprocessing
import beamngpy
import time
import matplotlib.pyplot as plt
import json
from beamngpy import BeamNGpy, Scenario, Vehicle
import keyboard  # Add this import


def run_simulation(bng, scenario, vehicle, stop_event):
    bng.scenario.load(scenario)
    bng.scenario.start()

    while not stop_event.is_set():
        vehicle.ai.set_mode("span")
        vehicle.ai_set_aggression(1)
        vehicle.ai_drive_in_lane(True)
        vehicle.acc = 1

        bng.step()
        time.sleep(0.01)  # Adjust sleep time as needed

    bng.close()


def poll_data(bng, vehicle, queue, stop_event):
    while not stop_event.is_set():
        sensors = bng.poll_sensors(vehicle)
        color_image = sensors["camera"]["colour"]
        depth_image = sensors["camera"]["depth"]
        lidar_points = sensors["lidar"]["points"]
        vehicle_pos = vehicle.state["pos"]

        queue.put((color_image, depth_image, lidar_points, vehicle_pos))
        time.sleep(0.01)  # Adjust sleep time as needed


if __name__ == "__main__":
    with open("paths.json", "r") as file:
        paths = json.load(file)

    globalPath = paths.get("globalPath")
    bng = BeamNGpy("localhost", 64256, home=globalPath)
    bng.open()

    scenario = Scenario("west_coast_usa", "example")
    vehicle = Vehicle("ego_vehicle", model="etk800", license="PYTHON")
    scenario.add_vehicle(
        vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795)
    )
    scenario.make(bng)

    queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    simulation_process = multiprocessing.Process(
        target=run_simulation, args=(bng, scenario, vehicle, stop_event)
    )
    data_processing_process = multiprocessing.Process(
        target=poll_data, args=(bng, vehicle, queue, stop_event)
    )

    simulation_process.start()
    data_processing_process.start()

    # Listen for PageUp key press to set the stop_event
    keyboard.add_hotkey("page up", lambda: stop_event.set())

    try:
        while not stop_event.is_set():
            if not queue.empty():
                color_image, depth_image, lidar_points, vehicle_pos = queue.get()

                # Process the data (e.g., display images, analyze LiDAR points)

                # Display color image
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 3, 1)
                plt.title("Color Image")
                plt.imshow(color_image)
                plt.axis("off")

                # Display depth image
                plt.subplot(1, 3, 2)
                plt.title("Depth Image")
                plt.imshow(depth_image, cmap="gray")
                plt.axis("off")

                # Display LiDAR points
                plt.subplot(1, 3, 3)
                plt.title("LiDAR Points")
                plt.scatter(lidar_points[:, 0], lidar_points[:, 1], s=1)
                plt.axis("equal")

                plt.show()
            time.sleep(0.01)  # Adjust sleep time as needed
    except KeyboardInterrupt:
        stop_event.set()

    simulation_process.join()
    data_processing_process.join()
