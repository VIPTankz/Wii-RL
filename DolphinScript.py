# slave.py

print("Script Started!")

from dolphin import event, gui, savestate, memory, controller

import sys
import os
import inspect
from pathlib import Path

script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
script_directory = Path(script_directory)

shared_site_path = script_directory / "shared_site.txt"

# add libraries from your python install (needs to match dolphin version (currently 3.12))
if shared_site_path.exists() and shared_site_path.is_file():
    with open(shared_site_path, 'r', encoding='utf-8') as file:
        site_path = file.read()
    
    sys.path.append(site_path)

# Now we can import other libraries safely
try:
    import time
    import traceback
    import random
    import math
    import numpy as np
    from collections import deque
    from multiprocessing.connection import Client
    from PIL import Image, ImageEnhance
    from multiprocessing import shared_memory
    from copy import deepcopy
except Exception as e:
    print(e)
    raise Exception("stop")

def increment_alive(path='alive.txt'):
    path = script_directory / Path(path)
    alive_num = int(path.read_text().strip()) if path.exists() else 0
    path.write_text(str(alive_num + 1))
    return alive_num

save_states_path = script_directory / "MarioKartSaveStates"

instance_info_folder = script_directory / Path('instance_info')

# Read pid from pid_num.txt
pid = int((instance_info_folder / 'pid_num.txt').read_text().strip())

# Read our specific ID
id = int((instance_info_folder / f'instance_id{pid}.txt').read_text().strip())

# Write our own PID into script_pid{id}.txt
(instance_info_folder / f'script_pid{id}.txt').write_text(str(os.getpid()))

alive_num = increment_alive()

num_envs = int((instance_info_folder / 'num_envs.txt').read_text().strip())

log_path = instance_info_folder / f'slave_{id}.log'
def log_exc(exc: BaseException):
    with open(log_path, 'a') as f:
        f.write('— Exception occurred —\n')
        traceback.print_exc(file=f)
        f.write('\n\n')


FILE_PATH = script_directory / "shared_value.txt"

if not FILE_PATH.exists():
    FILE_PATH = script_directory.parent / "shared_value.txt"

print(f"FILE_PATH: {FILE_PATH}")

def get_value() -> float:
    """
    Read a float from shared_value.txt in the current directory.
    If it doesn’t exist yet, create it with INITIAL_VALUE.
    """
    if not FILE_PATH.exists():
        raise Exception("File doesn't exist!")
    return float(FILE_PATH.read_text().strip())

def set_value(new_val: float):
    """
    Overwrite shared_value.txt in the current directory with the given float.
    """
    FILE_PATH.write_text(str(float(new_val)))

class Memory:
    class Addresses:
        def __init__(self):
            # RaceManagerPlayer
            self.RaceCompletion = self.resolve_address(0x809BD730, [0xC, 0x0, 0xC])
            # LapCompletion was on a per-checkpoint basis, RaceCompletion is interpolated

            self.currentLap = self.resolve_address(0x809BD730, [0xC, 0x0, 0x24])
            self.countdownTimer = self.resolve_address(0x809BD730, [0x22])
            self.stage = self.resolve_address(0x809BD730, [0x28])

            # KartDynamics - Iterate 3 times with 4 bytes offset to get X, Y and Z.
            self.position = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x18])
            self.acceleration_KartDynamics = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0x80])
            self.mainRotation = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0xF0])
            self.internalVelocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0x14C])
            self.externalVelocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0x74])
            self.angularVelocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0xA4])
            self.velocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0xD4])

            # KartMove
            self.speed = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x20])
            self.acceleration_KartMove = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x30])
            self.miniturboCharge = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x44, 0xFE])

            # Can be used as a mushroom timer as well
            self.offroadInvincibility = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x148])

            self.wheelieFrames = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x2A8])
            self.wheelieCooldown = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x2B6])
            self.leanRot = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x294])

            # KartState
            self.bitfield2 = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x4, 0xC])

            # KartCollide
            self.surfaceFlags = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x18, 0x2C])

            # Misc
            self.mushroomCount = self.resolve_address(0x809C3618, [0x14, 0x90])
            self.hopPos = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x44, 0x22C])

            # I added
            self.mt_boost_timer = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x102])
            self.airtime = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x4, 0x1C])
            self.allmt = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x10C])
            self.mush_and_boost = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x110])
            self.floor_collision_count = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x40])
            self.race_position = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x3C])
            self.respawn_timer = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x18, 0x48])

            # this is called m_types in KartPhysics->CollisionGroup->CollisionData
            self.wall_collide = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x8, 0x8])

            self.soft_speed_limit = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x18])

            self.trickableTimer = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x4, 0xA6])

            self.trick_cooldown = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x258, 0x38])

        @staticmethod
        def resolve_address(base_address, offsets):
            """
            This is a helper function to allow multiple ptr dereferences in
            quick succession. base_address is dereferenced first, and then
            offsets are applied.
            """
            current_address = memory.read_u32(base_address)
            for offset in offsets:
                value_address = current_address + offset
                current_address = memory.read_u32(current_address + offset)

            return value_address

    def __init__(self):
        self.addresses = self.Addresses()

        # RaceManagerPlayer
        self.RaceCompletion: float = 0.0
        self.currentLap: int = 0
        self.countdownTimer: int = 0
        self.stage: int = 0

        # KartDynamics = list[float]
        self.position = np.array([0.0, 0.0, 0.0])
        self.acceleration_KartDynamics = np.array([0.0, 0.0, 0.0])
        self.mainRotation = np.array([0.0, 0.0, 0.0, 0.0])
        self.mainRotationEuler = np.array([0.0, 0.0, 0.0])
        self.internalVelocity = np.array([0.0, 0.0, 0.0])
        self.externalVelocity = np.array([0.0, 0.0, 0.0])
        self.angularVelocity = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

        # KartMove
        self.speed: float = 0.0
        self.acceleration_KartMove: float = 0.0
        self.miniturboCharge: int = 0
        self.offroadInvincibility = False
        self.wheelieFrames: int = 0
        self.wheelieCooldown: int = 0
        self.leanRot: float = 0.0

        # KartState
        self.bitfield2: int = 0
        self.isWheelie = False

        # KartCollide
        self.surfaceFlags: int = 0
        self.isAboveOffroad = False
        self.isTouchingOffroad = False

        # Misc
        self.mushroomCount = 0
        self.hopPos = 0

        # Oil Spill Radians and Distance
        self.oilSpillRadians = 0.0
        self.oilSpillDistance = 0.0

        # Boost Panel Radians and Distance
        self.boostPanelRadians = 0.0
        self.boostPanelDistance = 0.0

        # I added
        self.mt_boost_timer = 0
        self.airtime = 0
        self.allmt = 0
        self.mush_and_boost = 0
        self.floor_collision_count = 2
        self.race_position = 12
        self.respawn_timer = 0
        self.wall_collide = 0
        self.speed_limit = 100

        self.trickableTimer = 0
        self.trick_cooldown = 0

    def update(self):

        # my ones
        self.mt_boost_timer = memory.read_u16(self.addresses.mt_boost_timer)
        self.airtime = memory.read_u16(self.addresses.airtime)

        self.allmt = memory.read_u16(self.addresses.allmt)
        self.mush_and_boost = memory.read_u16(self.addresses.mush_and_boost)
        self.floor_collision_count = memory.read_u16(self.addresses.floor_collision_count)

        self.race_position = memory.read_u8(self.addresses.race_position)
        self.respawn_timer = memory.read_u16(self.addresses.respawn_timer)
        self.wall_collide = memory.read_u32(self.addresses.wall_collide)

        self.speed_limit = memory.read_f32(self.addresses.soft_speed_limit)

        # RaceManagerPlayer
        self.RaceCompletion = memory.read_f32(self.addresses.RaceCompletion)

        self.currentLap = memory.read_u16(self.addresses.currentLap)
        self.countdownTimer = memory.read_u16(self.addresses.countdownTimer)
        self.stage = memory.read_u32(self.addresses.stage)

        # KartMove
        self.speed = memory.read_f32(self.addresses.speed)

        self.offroadInvincibility = memory.read_u16(self.addresses.offroadInvincibility)

        # KartCollide
        self.isTouchingOffroad = self.surfaceFlags & (1 << (7 - 1)) != 0

        # Misc
        self.mushroomCount = memory.read_u32(self.addresses.mushroomCount)
        self.hopPos = memory.read_f32(self.addresses.hopPos)

        self.trickableTimer = memory.read_u16(self.addresses.trickableTimer)
        self.trick_cooldown = memory.read_u16(self.addresses.trick_cooldown)

    @staticmethod
    def Quat2Euler(quaternion):

        x, y, w, z = quaternion

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])

class DolphinInstance:
    def __init__(self, id):

        address = ('localhost', 26330 + id)
        print(f"Connecting to master at {address}...")
        self.conn = Client(address, authkey=b'secret password')
        print("Connected to master!")

        self.window_x = 140
        self.window_y = 75

        self.bestL1 = 999999
        self.bestL2 = 999999
        self.best_time = get_value()
        self.save_idx = 10

        self.reset_frame_buffer = False

        self.env_id = id

        self.framestack = 4
        self.frameskip = 4
        self.frames = deque([], maxlen=self.framestack) #frame buffer for framestacking
        self.num_envs = num_envs
        print(f"Num envs{self.num_envs}")

        try:
            # setup shared memory
            if(sys.version_info[1] < 13):
                self.shm = shared_memory.SharedMemory(name="states_shm")
            else:
                # Make sure that the shared memory doesn't get deleted on upon exiting script by setting track=False
                self.shm = shared_memory.SharedMemory(name="states_shm", track=False)
            self.states = np.ndarray(
                (self.num_envs, self.framestack, self.window_y, self.window_x),
                dtype=np.uint8,
                buffer=self.shm.buf
            )
        except Exception as e:
            print(e)
            print("Error when creating shared memory")

        self.define_action_space()
        self.reset()

    def define_action_space(self):

        self.wii_dic = {
            "Left": False, "Right": False, "Down": False,
            "Up": False, "Z": False, "R": False, "L": False,
            "A": True, "B": False, "X": False, "Y": False,
            "Start": False, "StickX": 0, "StickY": 0, "CStickX": 0,
            "CStickY": 0, "TriggerLeft": 0, "TriggerRight": 0,
            "AnalogA": 0, "AnalogB": 0, "Connected": True
        }

        # Define discrete action values
        self.stickX_values = [-1, -0.4, 0, 0.4, 1]
        self.r_values = [False, True]
        self.up_values = [False, True]
        self.l_values = [False, True]
        # Compute total number of discrete actions
        self.n_actions = (len(self.stickX_values) *
                          len(self.r_values) *
                          len(self.up_values) *
                          len(self.l_values))

    def send_init_state(self, img):
        self.states[self.env_id] = img
        self.conn.send("Sent initial states")

    def recieve_action(self):
        self.applied_action = self.conn.recv()

    def send_transition(self, reward, terminal, trun, new_img):
        # write into shared memory

        if self.reset_frame_buffer:
            # Overwrite the entire frame stack with the new frame
            self.states[self.env_id, ...] = new_img
            self.reset_frame_buffer = False
        else:
            # Shift frames left: frames 1..end → 0..end-1
            self.states[self.env_id, :-1] = self.states[self.env_id, 1:]
            # Add new frame at the end (index -1)
            self.states[self.env_id, -1] = new_img

        # send the rest over the socket
        self.conn.send((reward, terminal, trun, {}))


    def process_indiv_frame(self, img):
        # img == (834, 456), (this is 832 is the X, this is a widescreen image)

        # greyscale
        image = img.convert("L")

        # resize image to our size
        image = image.resize((self.window_x, self.window_y))

        # convert image to numpy uint8
        image = np.asarray(image).astype(np.uint8) #process Uint
        # (x, y)
        return image

    def process_frame(self, img):
        """
        :param img: numpy array of most recent (frame_pool) frames
        :return: returns the pooled observation
        """

        observation = np.maximum(img[0], img[1])
        return observation

    def get_mem_values(self):
        self.memory_tracker.update()

        self.mem_speed = self.memory_tracker.speed

        self.mem_race_pos = self.memory_tracker.race_position

        # max race completion
        self.mem_race_com = self.memory_tracker.RaceCompletion

        self.mem_offroad_invin = self.memory_tracker.offroadInvincibility
        self.mem_touching_offroad = self.memory_tracker.isTouchingOffroad

        self.mem_race_stage = self.memory_tracker.stage

    def reset(self):

        self.ep_length = 0

        # this action will be applied directly before the frame is drawn
        self.applied_action = 0

        self.frames_since_chkpt = 0

        self.num_checkpoints_per_lap = 10  # 3 laps total
        self.checkpoints = []
        self.current_checkpoint = 0

        num_checkpoints_per_lap = 10
        num_laps = 3
        start = 1.0
        end = 4.0
        lap_length = (end - start) / num_laps  # 1.0

        for lap in range(num_laps):
            lap_start = start + lap * lap_length
            step = lap_length / num_checkpoints_per_lap
            # Exclude the lap_start itself (since your tracker starts at 1.0, not 0.0)
            for i in range(1, num_checkpoints_per_lap + 1):
                self.checkpoints.append(round(lap_start + i * step, 10))  # rounding for floating point issues

        # just make sure we don't list index out of range
        self.checkpoints.append(9999.)

        # pick random state to reset to
        x = random.randint(2, 8)

        # reset environment back to savestate
        save_state_path = save_states_path / f"RMCP01.s0{x}"
        savestate.load_from_file(str(save_state_path))

        self.memory_tracker = Memory()

        self.get_mem_values()

        # move our current checkpoint to where we are based on spawn location
        while self.mem_race_com > self.checkpoints[self.current_checkpoint]:
            self.current_checkpoint += 1


    def apply_action(self, action):
        assert 0 <= action < self.n_actions, f"Action must be in 0..{self.n_actions-1}"

        # reset dictionary to default state (A is always held down)
        self.wii_dic = {
            "Left": False, "Right": False, "Down": False,
            "Up": False, "Z": False, "R": False, "L": False,
            "A": True, "B": False, "X": False, "Y": False,
            "Start": False, "StickX": 0, "StickY": 0, "CStickX": 0,
            "CStickY": 0, "TriggerLeft": 0, "TriggerRight": 0,
            "AnalogA": 0, "AnalogB": 0, "Connected": True
        }

        self.get_mem_values()

        # Decode indices. Can't lie ChatGPT did this, idn wtf is going on here
        stick_idx = action // (2 * 2 * 2)
        rem = action % (2 * 2 * 2)
        r_idx = rem // (2 * 2)
        rem = rem % (2 * 2)
        up_idx = rem // 2
        l_idx = rem % 2

        # Set relevant fields
        self.wii_dic["StickX"] = self.stickX_values[stick_idx]
        self.wii_dic["R"] = self.r_values[r_idx]
        self.wii_dic["Up"] = self.up_values[up_idx]
        self.wii_dic["L"] = self.l_values[l_idx]

        self.applied_action = action
        controller.set_gc_buttons(0, self.wii_dic)

    def get_reward_terminal_trun(self):
        reward = 0.
        terminal = False
        trun = False

        # refresh memory values
        self.get_mem_values()

        self.ep_length += 1

        # checkpoint bonus
        if self.mem_race_com > self.checkpoints[self.current_checkpoint]:
            reward += 1.
            self.current_checkpoint += 1
            self.frames_since_chkpt = 0

        # reward for finishing race and set terminal
        if self.mem_race_com >= 4.0:
            # reward based on position
            reward = (13 - self.mem_race_pos) / 2
            terminal = True
        # race has ended, reset
        elif self.mem_race_stage == 4:
            reward = -1
            terminal = True
        # reset condition.
        elif self.frames_since_chkpt > 700:
            reward = -1.
            terminal = True

        self.frames_since_chkpt += 1

        return reward, terminal, trun

for i in range(4):
    await event.frameadvance()

env = DolphinInstance(id)

for i in range(8):
    await event.frameadvance()

(width, height, data) = await event.framedrawn()
img = Image.frombytes('RGB', (width, height), data, 'raw')

print("Processing init state...")
img = env.process_indiv_frame(img)
img = np.array([img for _ in range(env.frameskip)])

env.send_init_state(img)

print("Sent init state")

def my_callback():
    env.apply_action(env.applied_action)

event.on_frameadvance(my_callback)
# make sure we apply the action every single frame. Otherwise this can lead to some weird stuttering
# behaviour

reward = 0
terminal = False
trun = False

frames_pooled = 2
print("Starting Main Loop...")
# atari pools the most recent two frames, don't blame me why its so confusing
frame_data = np.zeros((frames_pooled, env.window_y, env.window_x), dtype=np.uint8)
while True:

    # get action from main Dolphin Script
    env.recieve_action()

    for i in range(env.frameskip):
        if i >= env.frameskip - frames_pooled:
            # get frame data
            (width, height, data) = await event.framedrawn()
            new_img = env.process_indiv_frame(Image.frombytes('RGB', (width, height), data, 'raw'))
            frame_data[i - frames_pooled] = new_img
        else:
            # no frame data, just skip frame
            await event.frameadvance()

        rewardN, terminalN, trunN = env.get_reward_terminal_trun()

        if not terminal and not trun:
            terminal = terminal or terminalN
            trun = trun or trunN
            reward += rewardN

        if terminal or trun:
            # send transition so we can carry going on while resetting
            new_img = Image.frombytes('RGB', (width, height), data, 'raw')
            new_img = env.process_indiv_frame(new_img)

            for i in range(frames_pooled):
                frame_data[i] = new_img

            new_img = env.process_frame(np.array(frame_data).copy())
            env.send_transition(reward, terminal, trun, new_img.copy())

            # add some time here or dolphin seems to freeze up sometimes
            for _ in range(2):
                await event.frameadvance()

            env.reset()

            for _ in range(1):
                await event.frameadvance()

            # reset frame_buffer
            env.reset_frame_buffer = True
            break

    if not (terminal or trun):
        new_img = env.process_frame(np.array(frame_data).copy())
        env.send_transition(reward, terminal, trun, new_img)

    reward = 0
    terminal = False
    trun = False
