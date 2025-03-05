from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import GPS

import matplotlib.pyplot as plt
import json


def main():
    set_up_simple_logging()

    with open("paths.json", "r") as file:
        paths = json.load(file)

    globalPath = paths.get("globalPath")
    bng = BeamNGpy("localhost", 64256, home=globalPath)
    bng.open()

    vehicle = Vehicle("ego_vehicle", model="etki", licence="PYTHON", color="Red")
    scenario = Scenario("italy", "GPS_Trajectory", description="GPS")
    scenario.add_vehicle(
        vehicle,
        pos=(245.11, -906.94, 247.46),
        rot_quat=(0.0010, 0.1242, 0.9884, -0.0872),
    )
    scenario.make(bng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    ref_lon, ref_lat = 8.8017, 53.0793

    gps_front = GPS(
        "front",
        bng,
        vehicle,
        pos=(0, 1.5, 2.0),
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        is_visualised=True,
    )
    gps_rear = GPS(
        "rear",
        bng,
        vehicle,
        pos=(0, -1.5, 2.0),
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        is_visualised=True,
    )

    vehicle.ai.set_mode("traffic")
    front_lon, front_lat, rear_lon, rear_lat = [], [], [], []

    print("Press Ctrl+C to stop real-time polling.")
    sleep(5.0)
    try:
        while True:
            data_front = gps_front.poll()
            print(data_front)
            data_rear = gps_rear.poll()
            print(data_rear)
            front_latest = data_front[0]
            rear_latest = data_rear[0]
            front_lon.append(front_latest["lon"])
            front_lat.append(front_latest["lat"])
            rear_lon.append(rear_latest["lon"])
            rear_lat.append(rear_latest["lat"])
            print(
                f"Front: lon={front_latest['lon']}, lat={front_latest['lat']} | "
                f"Rear: lon={rear_latest['lon']}, lat={rear_latest['lat']}"
            )
            # plot updated trajectory graph
            plt.plot(front_lon, front_lat, "r", label="GPS front")
            plt.plot(rear_lon, rear_lat, "b", label="GPS rear")
            plt.plot(front_lon, front_lat, "ro")
            plt.plot(rear_lon, rear_lat, "bo")
            plt.legend()
            plt.pause(0.01)

            sleep(1.0)
    except KeyboardInterrupt:
        pass

    gps_front.remove()
    gps_rear.remove()
    bng.ui.show_hud()

    fig, ax = plt.subplots()
    ax.set(xlabel="Longitude", ylabel="Lattitude", title="Vehicle Trajectory")
    ax.plot(front_lon, front_lat, "r", label="GPS front")
    ax.plot(rear_lon, rear_lat, "b", label="GPS rear")
    ax.plot(front_lon, front_lat, "ro")
    ax.plot(rear_lon, rear_lat, "bo")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
