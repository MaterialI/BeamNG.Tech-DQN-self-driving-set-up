from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Mesh
import json


with open("paths.json", "r") as file:
    paths = json.load(file)
globalPath = paths.get("globalPath")
bng = BeamNGpy("localhost", 64256, home=globalPath)
bng.open(launch=False)
ego = Vehicle("ego", model="etk800")
ego.connect(bng)
ego.ai_set_mode("manual")
ego.control(throttle=0.2, steering=0)
print(bng)
