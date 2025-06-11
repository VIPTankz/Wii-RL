from dolphin import savestate,  controller
import random
import cv2
import numpy as np
from collections import deque
from multiprocessing.connection import Client
from multiprocessing import shared_memory
from MKWMemory import MKWMemory
class DolphinInstance:
    def __init__(self, id,num_envs,save_states_path=None,share_value_path=None):
        address = ('localhost', 26330 + id)
        print(f"Connecting to master at {address}...")
        self.conn = Client(address, authkey=b'secret password')
        print("Connected to master!")

        self.window_x = 140
        self.window_y = 75

        self.bestL1 = 999999
        self.bestL2 = 999999
        self.best_time = self.get_value(share_value_path)
        self.save_idx = 10

        self.reset_frame_buffer = False

        self.env_id = id
        self.save_states_path = save_states_path

        self.framestack = 4
        self.frameskip = 4
        self.frames = deque([], maxlen=self.framestack) #frame buffer for framestacking
        self.num_envs = num_envs
        print(f"Num envs: {self.num_envs}")

        try:
            # setup shared memory
            self.shm = shared_memory.SharedMemory(name="states_shm")
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
    
    def process_indiv_frame(self, width, height, data):
        # Step 1: convert raw bytes → NumPy array
        img_np = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)

        # Step 2: convert RGB → Grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Step 3: resize to target size
        resized = cv2.resize(gray, (self.window_x, self.window_y), interpolation=cv2.INTER_AREA)

        # Return processed frame (window_y, window_x), uint8
        return resized

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
        savestate.load_from_file(self.save_states_path + f"RMCP01.s0{x}")

        self.memory_tracker = MKWMemory()

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
    

    def get_value(self,share_value_path) -> float:
        """
        Read a float from shared_value.txt in the current directory.
        If it doesn’t exist yet, create it with INITIAL_VALUE.
        """
        if not share_value_path.exists():
            raise Exception("File doesn't exist!")
        return float(share_value_path.read_text().strip())