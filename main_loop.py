from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Mesh, RoadsSensor, GPS, AdvancedIMU
import json
import cv2
import numpy as np
from time import sleep
import time
import os
import matplotlib.pyplot as plt
from utils.stack_paths import Checkpointpath
from collections import deque
import numpy as np
import torch
from models.DQN import DQN
import random
import gc

# import pygame
import keyboard


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
epsilon_value = 1
EPSILON_DECAY = 0.99999  # in 1000 54.6% of the time it will be random
len_replay = 10000
replay_buffer = deque(
    maxlen=len_replay
)  # 12000 crashed (after optimization will be 23000)
# data_buffer = deque(maxlen=(len_replay / 2) + 1)  # stores magnitude of the rewards


image_fuse_buffer = deque(maxlen=10)
image_fuse_buffer_wide = deque(maxlen=10)

max_acc_norm = 120
# turn on tensor core operations
torch.backends.cudnn.benchmark = True


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def init_cameras(bng, ego):
    camera = Camera(
        "front_center_camera",
        bng,
        ego,
        requested_update_time=0.025,
        is_visualised=True,
        is_using_shared_memory=True,
        pos=(0, -2.18, 0.5),
        dir=(0, -1, 0),
        field_of_view_y=70,
        near_far_planes=(0.1, 1000),
        resolution=(300, 300),
        is_streaming=True,
        is_render_annotations=True,
        is_render_instance=True,
        is_render_depth=True,
        is_depth_inverted=True,
    )

    camera1 = Camera(
        "back_center_camera",
        bng,
        ego,
        requested_update_time=0.025,
        is_visualised=True,
        is_using_shared_memory=True,
        pos=(0, 2.28, 0.5),
        dir=(0, 1, 0),
        field_of_view_y=70,
        near_far_planes=(0.1, 1000),
        resolution=(300, 300),
        is_streaming=True,
        is_render_annotations=True,
        is_render_instance=True,
        is_render_depth=True,
        is_depth_inverted=True,
    )

    camera2 = Camera(
        "left_front_fender_camera",
        bng,
        ego,
        requested_update_time=0.025,
        is_visualised=True,
        is_using_shared_memory=True,
        pos=(0.9, -1.4, 0.75),
        dir=(1, 0, 0),
        field_of_view_y=120,
        near_far_planes=(0.1, 1000),
        resolution=(300, 300),
        is_streaming=True,
        is_render_annotations=True,
        is_render_instance=True,
        is_render_depth=True,
        is_depth_inverted=True,
    )

    camera3 = Camera(
        "right_front_fender_camera",
        bng,
        ego,
        requested_update_time=0.025,
        is_visualised=True,
        is_using_shared_memory=True,
        pos=(-0.9, -1.4, 0.75),
        dir=(-1, 0, 0),
        field_of_view_y=120,
        near_far_planes=(0.1, 1000),
        resolution=(300, 300),
        is_streaming=True,
        is_render_annotations=True,
        is_render_instance=True,
        is_render_depth=True,
        is_depth_inverted=True,
    )

    camera4 = Camera(
        "front_wide_camera",
        bng,
        ego,
        requested_update_time=0.025,
        is_visualised=True,
        is_using_shared_memory=True,
        pos=(0, -1, 1.5),
        dir=(0, -1, 0),
        field_of_view_y=120,
        near_far_planes=(0.1, 1000),
        resolution=(300, 300),
        is_streaming=True,
        is_render_annotations=True,
        is_render_instance=True,
        is_render_depth=True,
        is_depth_inverted=True,
    )
    return camera, camera1, camera2, camera3, camera4


def merge_depth_annotation(depth_frame, annotation_frame):
    depth_norm = cv2.normalize(
        (depth_frame) ** 6, None, 0, 1.0, cv2.NORM_MINMAX
    ).astype(np.float32)

    # 2. If depth_norm is single channel (i.e. shape is (H, W)) and annotation_frame has 3 channels,
    #    expand the depth image so that it has the same number of channels.
    if len(depth_norm.shape) == 2:
        # Add a third dimension so that shape becomes (H, W, 1)
        depth_norm = depth_norm[..., np.newaxis]

    # Repeat the single channel depth image along the channel axis to get a (H, W, 3) array.
    depth_norm_3c = np.repeat(depth_norm, 4, axis=2)

    # 3. Convert the annotation frame to float (to avoid overflow) and multiply by the depth scaling.
    #    This scales the RGB channels of the annotation based on the depth.
    annotation_frame_scaled = annotation_frame.astype(np.float32) * depth_norm_3c

    # 4. Clip the result to the valid range [0, 255] and convert back to uint8.
    annotation_frame_scaled = np.clip(annotation_frame_scaled, 0, 255).astype(np.uint8)

    return annotation_frame_scaled


# reward is based on time taken to reach the next checkpoint, distance to the next checkpoint, and the acceleration data (if more than 75 units, then it is collision and should be quite penalized)
def calculate_reward(deltaDistance, acc_data, achieved_checkpoint):
    """
    Calculate the reward based on the distance to the next checkpoint, time taken to reach the next checkpoint, and the acceleration data.

    Parameters:
    deltaDistance (float): The distance taken since last step.
    acc_data (list): The acceleration data, magnitude of the acceleration.
    achieved_checkpoint (bool): Whether the vehicle has reached the checkpoint.

    Returns:
    float: The reward value.
    """
    reward = -0.025

    if abs(deltaDistance) < 10:
        reward -= (deltaDistance) * abs(deltaDistance)
    # squaring the distance with preservation of sign
    reward += int(achieved_checkpoint) * 8
    if acc_data > max_acc_norm:
        reward -= 10
    return reward

    # Each timestep, call push_observation with new data.
    # The oldest item is automatically removed if the queue is full.


def map_action_to_control(action):
    """
    Map the action to the control of the vehicle.

    Parameters:
    action (int): The action to be mapped.
    0 - Throttle Forward, 1 - Throttle Backward, 2 - Turn Left, 3 - Turn Right, 4- Front Left, 5 - Front Right, 6 - Back Left, 7 - Back Right, 8 - Neutral

    Returns:
    dict: The control of the vehicle. The control is a dictionary with keys "throttle", "steering", and "brake".
    """
    control = dict()
    if action == 0:
        control = {"throttle": 1.0, "steering": 0.0, "brake": 0.0}
    if action == 1:
        control = {"throttle": 0.0, "steering": 0.0, "brake": 1.0}
    if action == 2:
        control = {"throttle": 0.0, "steering": 1.0, "brake": 0.0}
    if action == 3:
        control = {"throttle": 0.0, "steering": -1.0, "brake": 0.0}
    if action == 4:
        control = {"throttle": 1.0, "steering": 1.0, "brake": 0.0}
    if action == 5:
        control = {"throttle": 1.0, "steering": -1.0, "brake": 0.0}
    if action == 6:
        control = {"throttle": 0.0, "steering": 1.0, "brake": 1.0}
    if action == 7:
        control = {"throttle": 0.0, "steering": -1.0, "brake": 1.0}
    if action == 8:
        control = {"throttle": 0.0, "steering": 0.0, "brake": 0.0}
    return control


def scale(obs, maximum, minimum):
    """
    Scale the observation to a range of [0, 1].

    Parameters:
    obs (float or float array): The observation value.
    max (float): The maximum value of the observation.
    min (float): The minimum value of the observation.

    Returns:
    float: The scaled observation value. If the observation is an array, the function returns an array of scaled values.
    Observations exceeding the maximum value are set to 1, and observations below the minimum value are set to -1.

    Examples:
    >>> scale(10, 20, 0)
    0.0
    >>> scale([10, 20], 20, 0)
    [0.0, 1.0]
    >>> scale(-10, 20, 0)
    -1.0
    >>> scale([-100, 20], 20, 0)
    [-1.0, 1.0]

    """
    if isinstance(obs, (int, float)):
        scaled = 2.0 * ((obs - minimum) / (maximum - minimum)) - 1.0
        return max(-1, min(1, scaled))
    else:
        result = []
        for o in obs:
            scaled = 2.0 * ((o - minimum) / (maximum - minimum)) - 1.0
            result.append(max(-1, min(1, scaled)))
        return result


def path_reset():
    global path
    path = Checkpointpath((-372.6633, 193.6576))
    path.push((-461.9179, 483.0262))
    path.push((-305.0654, 391.5097))
    path.push((-147.3450, 317.4250))
    path.push((-159.2452, 199.4327))
    path.push((101.4353, -364.9311))
    path.push((293.9953, -290.1348))  # furthest
    path.push((395.1891, -131.5349))
    path.push((197.7489322, 178.8479191))
    path.push((-288.6571, 106.1199))  # first checkpoint
    path.push((-324.9976, 132.5635))
    path.push((-392.5179, 233.5088))


def epsilon_greedy(q_values):
    """
    Choose the action based on the epsilon-greedy policy.
    q_values (tensor): The Q-values of the model.
    choose random action with probability epsilon_value, otherwise choose the action with the highest Q-value.

    Returns:
    int: Index The action to be taken.
    """
    global epsilon_value

    epsilon_value *= EPSILON_DECAY
    if random.random() < epsilon_value:
        return random.randint(0, q_values.size(1) - 1)
    else:
        return torch.argmax(q_values).item()


def merge_images_buffer(buffer):
    """
    input: buffer of images each image is a numpy array of shape (300, 300, 4)
    output: a single image that is the merge of all images in the buffer along all of 4 channels do averaging


    """
    buffer = np.array(buffer)
    return np.mean(
        buffer, axis=2
    )  # averaging the images in the buffer along the channel axis


action = 8
respawn_point = (-402.42, 248.61, 25.3)
if __name__ == "__main__":

    with open("paths.json", "r") as file:
        paths = json.load(file)

    globalPath = paths.get("globalPath")

    bng = BeamNGpy("localhost", 64256, home=globalPath)
    bng.open(launch=True)

    scenario = Scenario("hirochi_raceway", "camera_streaming")

    # Mesh sensor co\nfiguration

    ego = Vehicle("ego", model="etk800", color="White")

    # Create and attach the mesh sensor

    # In your main loop, poll the sensor data

    scenario.add_vehicle(
        ego,
        pos=(-402.42, 248.61, 25.3),
        rot_quat=(0.01671021, 0.00486505, -0.29160145, 0.95639205),
    )

    scenario.make(bng)
    bng.settings.set_deterministic(120)
    bng.scenario.load(scenario)
    bng.scenario.start()

    imu = AdvancedIMU(
        "advanced_imu",
        bng,
        ego,
        pos=(0, -2, 1.0),
        is_send_immediately=True,
        accel_window_width=5,
        gyro_window_width=5,
        # visualize=True,
    )

    imu_back = AdvancedIMU(
        "advanced_imu_back",
        bng,
        ego,
        pos=(0, 2.3, 0.75),
        is_send_immediately=True,
        accel_window_width=5,
        gyro_window_width=5,
        # visualize=True,
    )

    gps_sensor = GPS(
        "my_gps_sensor",
        bng,
        ego,
        pos=(0, -1, 1.5),
    )
    camera, camera1, camera2, camera3, camera4 = init_cameras(bng, ego)
    # Cameras

    path_reset()

    sleep(5)
    runs_per_second = 0
    start_time = time.time()

    observation_queue = deque(
        maxlen=20
    )  # important, this is the queue that will store the observations for the DQN model. Each observation is a tensor containing telemetry data.

    # Images will be passed as a separate input to the model. (For now singular observation) Later will be converted to a convLSTM model.
    def push_observation(obs):
        observation_queue.append(obs)

    prev_observation = {}
    prev_reward = 0
    prev_action = 0

    prev_distance = 0

    # initialize the model
    model = DQN().to(torch.bfloat16).to("cuda")
    # define adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # define loss function
    loss_fn = torch.nn.SmoothL1Loss()  # Huber loss
    done = False
    while True:
        try:
            # Step the simulation once to ensure all sensor data is updated synchronously
            # bng.control.step(1)

            # check time to run the simulation

            start = time.time()
            bng.control.step(10)

            # 7) Pause again, ensuring that the simulator won't keep going
            bng.pause()
            # print(f"Time to run the simulation: {time.time() - start}")

            print(f"Replay buffer size: {len(replay_buffer)}")
            # Get the current state of the sensors
            start_time = time.time()
            scenario.update()
            state = ego.sensors["state"]
            vel = state["vel"]  # velocity
            direction = state["dir"]
            up = state["up"]
            speed = np.linalg.norm(vel)
            speed = scale(
                speed, 100, -100
            )  # max min 100 m/s and -100 m/s <- to pass to the model
            direction = np.array(state["dir"], dtype=np.float32)
            dir_mag = np.linalg.norm(direction)
            if dir_mag != 0:
                direction /= dir_mag  # <- to pass to the model

            vel = scale(vel, 100, -100)  # max min 100 and -100 <- to pass to the model
            # print(f"Time to get the state: {time.time() - start_time}")
            # passing to qnet these parameters:
            # (1) Two annotated frames
            # (2) Speed, normalized velocity vector, and normalized direction
            # (3) Distance to the next checkpoint, normalized checkpoint vector
            # (4) Acceleration data
            start_time = time.time()
            images = camera.stream()
            images4 = camera4.stream()

            depth_frame = np.array(images["depth"], dtype=np.float32)
            annotation_frame = np.array(images["annotation"], dtype=np.float32)

            depth_frame_wide = np.array(images4["depth"], dtype=np.float32)
            annotation_frame_wide = np.array(images4["annotation"], dtype=np.float32)

            # print(f"Time to get the images: {time.time() - start_time}")

            start_time = time.time()

            runs_per_second += 1
            elapsed = time.time() - start_time

            # ----- Scaling the Annotation Frame with the Depth -----
            annotation_frame_scaled = merge_depth_annotation(
                depth_frame, annotation_frame
            )

            annotation_frame_wide_scaled = merge_depth_annotation(
                depth_frame_wide, annotation_frame_wide
            )

            # print(f"Time to merge the images: {time.time() - start_time}")
            # image_fuse_buffer.append(annotation_frame_scaled)
            # image_fuse_buffer_wide.append(annotation_frame_wide_scaled)
            # collect acceleration data -------------------------
            # print(sensor_data)
            start_time = time.time()
            sensor_data = imu.poll()
            if isinstance(sensor_data, list):
                continue
            accRaw = sensor_data.get("accRaw", None)

            accbackRaw = imu_back.poll()
            if isinstance(accbackRaw, list):
                continue
            accbackRaw = accbackRaw.get("accRaw", None)
            # print(accRaw)
            # if "acc_data" not in globals():
            #     acc_data = []

            accbackRawNorm = np.linalg.norm(accbackRaw)
            accRawNorm = np.linalg.norm(accRaw)
            accRaw = scale(
                accRaw, 980, -980
            )  # max min 10g and -10g <- to pass to the model
            accbackRaw = scale(
                accbackRaw, 980, -980
            )  # max min 10g and -10g <- to pass to the model

            gyroRaw = sensor_data.get("angVel", None)
            gyroRaw = scale(
                gyroRaw, 13, -13
            )  # max min 13 rad/s and -13 rad/s <- to pass to the model
            # print(
            #     f"Time to get the accelerometer sensor data: {time.time() - start_time}"
            # )
            # to process next checkpoint -------------------------
            # parses if next checkpoint is reached
            # if reached, then calculate reward and set new checkpoint as the next checkpoint
            # if not reached, then continue to the next iteration

            # poll gps data
            start_time = time.time()
            gps_data = gps_sensor.poll()  # global
            if len(gps_data) == 0:
                continue
            latest_location = gps_data[0]
            longitude = latest_location["x"]
            latitude = latest_location["y"]

            # check distance to next checkpoint

            checkpoint0, triggered = path.get(
                longitude, latitude, time.time()
            )  # global
            if triggered:
                respawn_point = checkpoint0
            if checkpoint0 is None:
                distance = -10000
            else:
                distance = np.sqrt(
                    (latitude - checkpoint0[0]) ** 2 + (longitude - checkpoint0[1]) ** 2
                )

            # if distance < 10:
            #     print("Checkpoint reached")

            direction_vector = np.array(
                [checkpoint0[0] - latitude, checkpoint0[1] - longitude]
            )

            magnitude = np.linalg.norm(direction_vector)
            if magnitude != 0:
                direction_vector /= magnitude  # <- to pass to the model

            # scale data
            if prev_distance == 0:
                deltaDistance = 0
            elif triggered:
                deltaDistance = 0
            else:
                deltaDistance = distance - prev_distance
            prev_distance = distance

            # print(f"Time to get the gps data: {time.time() - start_time}")
            # append to the observation queue all data
            # observation = (
            #     accRaw,  # acceleration data (x, y, z) ->3
            #     accbackRaw,
            #     gyroRaw,  # gyro data (x, y, z) ->3
            #     direction,  # direction of the vehicle (x, y, z) ->3
            #     vel,  # velocity of the vehicle (x, y, z) ->3
            #     speed,  # speed of the vehicle scalar  ->1
            #     deltaDistance,  # distance to the next checkpoint scalar ->1
            #     direction_vector,  # direction vector to the next checkpoint (x, y) ->2
            # )  # total 16

            start_time = time.time()
            obs_1d = np.hstack(
                [
                    accRaw,  # acceleration data (x, y, z) ->3
                    accbackRaw,  # acceleration data (x, y, z) ->3
                    [max(accRawNorm, accbackRawNorm)],  # max acceleration ->1
                    gyroRaw,  # gyro data (x, y, z) ->3
                    direction,  # direction of the vehicle (x, y, z) ->3
                    vel,  # velocity of the vehicle (x, y, z) ->3
                    [speed],  # speed of the vehicle scalar  ->1
                    [deltaDistance],  # distance to the next checkpoint scalar ->1
                    direction_vector,  # direction vector to the next checkpoint (x, y) ->2
                    action,  # action taken by the model ->1
                ]
            )  # total 21

            push_observation(obs_1d)

            # Convert the observation queue into a tensor
            obs_array = np.array(observation_queue)
            obs_tensor = (
                torch.tensor(obs_array, dtype=torch.bfloat16).to("cuda").unsqueeze(0)
            )  # TODO: 16 bits precision should be enough for the observations (acceleration, gyro, direction, velocity, speed, distance, direction vector) 16 * 16 = 256 bits = 32 bytes

            # TODO: normalize images
            # Scale the images to a range of [0, 1]
            annotation_frame_scaled = (
                annotation_frame_scaled / 255.0  # 8 bits per channel
            )
            annotation_frame_wide_scaled = (
                annotation_frame_wide_scaled / 255.0  # 8 bits per channel
            )
            # Convert the images to tensors
            image_tensor = (
                torch.tensor(
                    annotation_frame_scaled, dtype=torch.bfloat16
                )  # TODO: do in uint 8 bits each channel Do tensor core support. 8 bits per channel
                .unsqueeze(0)
                .to("cuda")
            )
            # print(image_tensor.shape)
            image_tensor_wide = (
                torch.tensor(
                    annotation_frame_wide_scaled, dtype=torch.bfloat16
                )  # TODO: do in uint 8 bits each channel Do tensor core support. 8 bits per channel
                .unsqueeze(0)
                .to("cuda")
            )
            # print(
            #     f"Time to convert the images to tensors and append data: {time.time() - start_time}"
            # )
            # print(image_tensor_wide.shape)
            # Pass the tensors to the model

            if len(observation_queue) >= 20:

                # if we already done imitation learning data collection
                if len(replay_buffer) > -1:
                    with torch.no_grad():
                        q_values = model.forward(
                            image_tensor, image_tensor_wide, obs_tensor
                        )

                    # Get the action with the highest Q-value
                    action = epsilon_greedy(q_values.cpu().detach())

                    # Map the action to the control of the vehicle

                    control = map_action_to_control(action)
                    # Apply the control to the vehicle
                    ego.control(
                        **control
                    )  # throttle, steering, brake <- to pass to the model

                    # If we finished the immitation learning, then we can use the model to control the vehicle
                else:
                    # get keyboard input to control the vehicle
                    # pygame.init()
                    action = 8
                    up = keyboard.is_pressed("up")
                    down = keyboard.is_pressed("down")
                    left = keyboard.is_pressed("left")
                    right = keyboard.is_pressed("right")

                    action = 8

                    if up and not left and not right:
                        action = 0
                    elif down and not left and not right:
                        action = 1
                    elif left and not up and not down:
                        action = 3
                    elif right and not up and not down:
                        action = 2
                    elif up and left:
                        action = 5
                    elif up and right:
                        action = 4
                    elif down and left:
                        action = 7
                    elif down and right:
                        action = 6

                    print("ACTION TAKEN: ", action)
                    ego.control(**map_action_to_control(action))
                    # pygame.quit()

                # Calculate the reward
                reward = calculate_reward(
                    deltaDistance, max(accRawNorm, accbackRawNorm), triggered
                )
                deltaDistance = scale(deltaDistance, 100, -100)  # max min 100 and -100
                image_tensor = image_tensor.detach().cpu()
                image_tensor_wide = image_tensor_wide.detach().cpu()
                image_tensor = (image_tensor * 255).to(torch.uint8)
                image_tensor_wide = (image_tensor_wide * 255).to(torch.uint8)
                obs_tensor = obs_tensor.detach().cpu()
                observation = {
                    "img1": image_tensor,
                    "img2": image_tensor_wide,
                    "obs": obs_tensor,
                }
                del obs_tensor
                del image_tensor
                del image_tensor_wide
                torch.cuda.empty_cache()
                gc.collect()
                print_gpu_memory()
                # Append the data to the replay buffer
                replay_buffer.append(
                    [
                        prev_observation,
                        prev_action,
                        prev_reward,
                        observation,
                        done,
                    ]
                )

                # Append the magnitude of the reward to the weight buffer
                # weight_buffer.append(abs(reward))

                # tra
                # in the model if the replay buffer is full
                if len(replay_buffer) >= 100:

                    data_list = list(replay_buffer)[1:]
                    # weight_list = np.array(weight_buffer)[1:]

                    # # normalize the weights
                    # probabilities = weight_list / weight_list.sum()
                    indices = np.random.choice(len(data_list), size=64, replace=False)
                    batch = np.array([data_list[i] for i in indices], dtype=object)

                    # print(batch)
                    # train the model
                    for sample in batch:
                        for key, tensor in sample[0].items():
                            sample[0][key] = tensor.to(torch.bfloat16).to("cuda")

                        for key, tensor in sample[3].items():
                            sample[3][key] = tensor.to(torch.bfloat16).to("cuda")

                    loss = model.train_step(optimizer, loss_fn, batch)

                    if "loss_history" not in globals():
                        loss_history = []
                    if "reward_history" not in globals():
                        reward_history = []

                    loss_history.append(loss)
                    reward_history.append(reward)

                    loss_to_plot = loss_history[-500:]
                    reward_to_plot = reward_history[-500:]

                    plt.figure("Loss Over Time")
                    plt.clf()
                    plt.plot(loss_to_plot, label="Loss")
                    plt.legend()
                    plt.pause(0.01)

                    plt.figure("Reward Over Time")
                    plt.clf()
                    plt.plot(reward_to_plot, label="Reward")
                    plt.legend()
                    plt.pause(0.01)

                    for sample in batch:
                        for key, tensor in sample[0].items():
                            sample[0][key] = tensor.detach().cpu()

                        for key, tensor in sample[3].items():
                            sample[3][key] = tensor.detach().cpu()
                    # Append the data to the replay buffer

                prev_observation = observation
                prev_reward = reward
                prev_action = action
                # print(q_values)
                if done:

                    done = False
                    observation = observation
                    reward = reward
                    action = 8
                    # path_reset()
            if distance < 0 or max(accRawNorm, accbackRawNorm) > max_acc_norm:
                done = True
                action = 8
                ego.control(**map_action_to_control(action))
                # restart the car (if it crashes)
                ego.teleport(
                    (-402.42, 248.61, 25.3),
                    (0.01671021, 0.00486505, -0.29160145, 0.95639205),
                )
                path_reset()
                sleep(10)
            if distance < 0:
                print("Training is done")
                break

            # plot the acceleration data -------------------------
            # if "raw_acc_data" not in globals():
            #     raw_acc_data = {"time": [], "acc_x": []}

            # current_time = time.time()
            # raw_acc_data["time"].append(current_time)
            # raw_acc_data["acc_x"].append(max(accRawNorm, accbackRawNorm))

            # # Limit to the last 100 observations
            # if len(raw_acc_data["time"]) > 100:
            #     times = raw_acc_data["time"][-100:]
            #     acc_x = raw_acc_data["acc_x"][-100:]

            # else:
            #     times = raw_acc_data["time"]
            #     acc_x = raw_acc_data["acc_x"]

            # plt.figure("Raw Acceleration Data")
            # plt.clf()
            # plt.plot(times, acc_x, label="Acc X")
            # plt.legend()
            # plt.pause(0.001)

            # ---------------------------------------------------
            # if elapsed >= 1.0:
            #     print("Latitude:", latitude)
            #     print("Longitude:", longitude)
            #     # print(obs_1d)
            #     # print(path.stack)
            #     # print("Acceleration Raw:", accRaw)
            #     # print("Direction Vector:", direction_vector)
            #     # print(distance)
            #     start_time = time.time()

        except KeyboardInterrupt:
            break
    model.save("model.pth")

    cv2.destroyWindow("Annotation")
    input("Press Enter to exit...")
    bng.close()


def visualize():

    # visualize acceleration data -------------------------
    if accRaw is not None:
        now = time.time()
        acc_data.append((now, accRaw))
        acc_data = [entry for entry in acc_data if (now - entry[0]) <= 10]
        if len(acc_data) > 500:
            acc_data = acc_data[-500:]

        x_vals = [d[1][0] for d in acc_data]
        y_vals = [d[1][1] for d in acc_data]
        z_vals = [d[1][2] for d in acc_data]

        if "vel_data" not in globals():
            vel_data = []
        vel_data.append((now, vel))
        vel_data = [entry for entry in vel_data if (now - entry[0]) <= 10]
        if len(vel_data) > 500:
            vel_data = vel_data[-500:]

        vx_vals = [d[1][0] for d in vel_data]
        vy_vals = [d[1][1] for d in vel_data]
        vz_vals = [d[1][2] for d in vel_data]

        def smooth(data, window=5):
            return np.convolve(data, np.ones(window) / window, mode="same")

        x_smoothed = smooth(x_vals)
        y_smoothed = smooth(y_vals)
        z_smoothed = smooth(z_vals)
        vx_smoothed = smooth(vx_vals)
        vy_smoothed = smooth(vy_vals)
        vz_smoothed = smooth(vz_vals)

        plt.clf()
        plt.plot(x_smoothed, label="AccX")
        plt.plot(y_smoothed, label="AccY")
        plt.plot(z_smoothed, label="AccZ")
        plt.plot(vx_smoothed, label="VelX")
        plt.plot(vy_smoothed, label="VelY")
        plt.plot(vz_smoothed, label="VelZ")
        plt.ylim(-100, 100)
        plt.legend()
        plt.pause(0.01)

    # ---------------------------------------------------
