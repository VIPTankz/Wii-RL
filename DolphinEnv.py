# master.py

import time
import subprocess
from pathlib import Path
from multiprocessing.connection import Listener
import site
import gymnasium as gym
import numpy as np
import os
from multiprocessing import shared_memory
import threading
import signal
import hashlib
import platform

# remove any existing shared memory
try:
    existing_shm = shared_memory.SharedMemory(name="states_shm")
    print("Found existing shared memory, cleaning up...")
    existing_shm.close()
    existing_shm.unlink()
except FileNotFoundError:
    # no existing shared memory, good
    pass
except Exception as e:
    print(f"Error cleaning old shared memory: {e}")

FILE_PATH = Path.cwd() / "shared_value.txt"
SITE_FILE_PATH = Path.cwd() / "shared_site.txt"
INITIAL_VALUE = 99999.

def get_value() -> float:
    """
    Read a float from shared_value.txt in the current directory.
    If it doesn’t exist yet, create it with INITIAL_VALUE.
    """
    if not FILE_PATH.exists():
        FILE_PATH.write_text(str(INITIAL_VALUE))
        return INITIAL_VALUE
    return float(FILE_PATH.read_text().strip())

def set_value(new_val: float):
    """
    Overwrite shared_value.txt in the current directory with the given float.
    """
    FILE_PATH.write_text(str(float(new_val)))


def calculate_md5(file_path):
    """
    Calculates a file's md5
    """
    md5_hash = hashlib.md5()
    
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()
  
def set_shared_site():
    """
    Writes the path to site-packages for DolphinEnv to use
    """
    sites_paths = site.getsitepackages()
    for path in sites_paths:
        if 'site-packages' in path:
            SITE_FILE_PATH.write_text(path)
            break 

class DolphinEnv:
    def __init__(self, num_envs, gamename="LC", gamefile="mkw.iso", project_folder=None,
                 games_folder=None, headless=False):

        script_directory = Path(os.path.dirname(os.path.abspath(__file__)))

        if(project_folder == None):
            project_folder = script_directory

        if(games_folder == None):
            games_folder = script_directory / "game"

        self.num_envs = num_envs
        self.gamename = gamename
        self.gamefile = gamefile
        self.headless = headless

        set_value(99999.)

        self.framestack = 4
        self.window_x = 140
        self.window_y = 75

        self.action_space = [gym.spaces.Discrete(40) for i in range(num_envs)]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.framestack, self.window_y, self.window_x),
            dtype=np.uint8
        )

        self.project_folder = Path(project_folder) if not isinstance(project_folder, Path) else project_folder
        self.games_folder = Path(games_folder) if not isinstance(games_folder, Path) else games_folder
        self._check_iso_validity()
        self.instance_info_folder = self.project_folder / Path('instance_info')
        self.instance_info_folder.mkdir(exist_ok=True)

        # write the number of envs for the slaves to read
        (self.instance_info_folder / 'num_envs.txt').write_text(str(self.num_envs))

        self.ids = list(range(self.num_envs))
        self.script_pids = [-1] * self.num_envs

        set_shared_site()

        self.shm = shared_memory.SharedMemory(create=True,
                                              size=self.num_envs * self.framestack * self.window_x * self.window_y,
                                              name="states_shm")
        self.states = np.ndarray(
            (self.num_envs, self.framestack, self.window_y, self.window_x),
            dtype=np.uint8,
            buffer=self.shm.buf
        )

        # prepare storage for raw listeners + per‐slot connections
        self.raw_listeners   = []
        self.listeners       = [None] * self.num_envs
        self.processes       = [None] * self.num_envs
        self.last_restart    = [0.0] * self.num_envs

        self.timeout = 8.

        self.is_resetting = [0] * self.num_envs

        self.firsts = [False] * self.num_envs

        # bind one raw listener per port
        self.raw_listeners = []
        for i in range(self.num_envs):
            raw = Listener(('localhost', 26330 + i), authkey=b'secret password')
            self.raw_listeners.append(raw)

        # reset pid counter and launch all envs
        (self.instance_info_folder / 'pid_num.txt').write_text('0')
        for i in range(self.num_envs):
            self.create_dolphin(i)

    def _check_iso_validity(self):
        valid_numbers = [
            'e7b1ff1fabb0789482ce2cb0661d986e',
            'ba68b5d7602bb6cd3d51f301f205e3dd'
        ]

        game_path = self.games_folder / self.gamefile

        if(not os.path.exists(game_path)):
            raise FileNotFoundError(f"File {game_path} doesn't exist.")
        elif(calculate_md5(game_path) not in valid_numbers):
            print("Warning: Unsupported ROM, might encounter some issues. Please refer to README.md.")
        else:
            print("The ROM provided is valid.")

    def increment_alive(self, path='alive.txt'):
        path = self.project_folder / Path(path)
        alive_num = int(path.read_text().strip()) if path.exists() else 0
        path.write_text(str(alive_num + 1))
        return alive_num

    def _accept_and_store(self, i):
        """Run (in a daemon thread) to accept the next Dolphin on slot i."""
        print(f"[Master] listening on port {26330+i} for Dolphin {i} …")
        conn = self.raw_listeners[i].accept()
        self.listeners[i] = conn
        print(f"[Master] Dolphin {i} connected!")

    def create_dolphin(self, i):
        print(f"Creating new Dolphin for process {i}")
        """Launch Dolphin i, then block on accept() for its new connection."""
        alive_num = self.increment_alive()

        # tell that slave its ID
        (self.instance_info_folder/'pid_num.txt').write_text(str(i))
        (self.instance_info_folder/f'instance_id{i}.txt').write_text(str(i))

        script_path = self.project_folder / 'DolphinScript.py'

        # launch the process
        platform_name = platform.system()
        if(platform_name == "Windows"):
            exe_path = self.project_folder / f'dolphin{i}' / 'Dolphin.exe'
            cmd = (
                f'cmd /c {exe_path} '
                f'--no-python-subinterpreters '
                f'--script "{script_path}" '
                f'\\b --exec="{self.games_folder/self.gamefile}"'
            )
        elif(platform_name == "Linux"):
            exe_path = self.project_folder / f'dolphin{i}' / 'dolphin-emu'
            if self.headless:
                cmd = (
                    f'{exe_path}',
                    '-b',
                    f'--no-python-subinterpreters',
                    f'--script', f'{self.project_folder}/DolphinScript.py',
                    f'--exec={self.games_folder/self.gamefile}'
                )
            else:
                cmd = (
                    f'{exe_path}',
                    f'--no-python-subinterpreters',
                    f'--script', f'{self.project_folder}/DolphinScript.py',
                    f'--exec={self.games_folder/self.gamefile}'
                )
        elif(platform_name == "Darwin"):
            exe_path = self.project_folder / f'dolphin{i}' / 'DolphinQt.app' / 'Contents' / 'MacOS' / 'DolphinQt'
            if self.headless:
                cmd = (
                    f'{exe_path}',
                    '-b',
                    f'--no-python-subinterpreters',
                    f'--script', f'{self.project_folder}/DolphinScript.py',
                    f'--exec={self.games_folder/self.gamefile}'
                )
            else:
                cmd = (
                    f'{exe_path}',
                    f'--no-python-subinterpreters',
                    f'--script', f'{self.project_folder}/DolphinScript.py',
                    f'--exec={self.games_folder/self.gamefile}'
                )
        else:
            raise RuntimeError(f"The operating system '{platform_name}' is not supported.")

        print(f"[Master] Opening File: {cmd}")
        env = os.environ.copy()
        if platform_name == "Darwin":
            # Find Python 3.13 home for MPS compatibility
            python_homes = [
                os.path.expanduser('~/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none'),
                '/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13',
                '/opt/homebrew/Cellar/python@3.13/3.13.11/Frameworks/Python.framework/Versions/3.13',
                '/Library/Frameworks/Python.framework/Versions/3.13',
            ]
            for path in python_homes:
                if os.path.exists(path):
                    env['PYTHONHOME'] = path
                    break
        self.processes[i] = subprocess.Popen(cmd, env=env)

        # wait for that slave to connect on our pre-bound socket
        print(f"[Master] Waiting for Dolphin {i} to connect…")
        conn = self.raw_listeners[i].accept()
        self.listeners[i] = conn
        print(f"[Master] Dolphin {i} connected!")

        # now proceed with your alive.txt + PID‐file handshake
        alive_path = self.project_folder / Path('alive.txt')
        while int(alive_path.read_text()) < alive_num + 2:
            time.sleep(0.05)

        pid_file = self.instance_info_folder / f'script_pid{i}.txt'
        while not pid_file.exists():
            time.sleep(0.01)
        self.script_pids[i] = int(pid_file.read_text())

        # final init‐state message
        got = self.listeners[i].recv()
        if got != "Sent initial states":
            raise RuntimeError(f"Env {i} failed init handshake")

    def reset(self):
        # return states and list of dicts for info
        return np.copy(self.states), [{} for i in range(self.num_envs)]

    def step_async(self, actions):
        for i in range(self.num_envs):

            if self.is_resetting[i] != 0:
                continue

            try:
                # normal send
                self.listeners[i].send(actions[i].item())
            except Exception as e:
                print(f"[WARN] Slave {i} connection broken at step_async: {e}")
                time.sleep(0.5)


    def step_wait(self):
        # this waits for all envs to finish processing, and returns the standard Gymnasium API

        rewards = []
        dones = []
        truns = []
        infos = {"final_observation":[], "Ignore": np.array([False for i in range(self.num_envs)]),
                 "First": np.array([False for i in range(self.num_envs)])}

        for i in range(self.num_envs):

            if self.is_resetting[i] != 0:
                infos["Ignore"][i] = True
                self.is_resetting[i] -= 1

                rewards.append(0.0)
                dones.append(False)
                truns.append(False)
                infos["final_observation"].append(None)
                self.firsts[i] = True

                continue

            try:
                if self.listeners[i].poll(self.timeout):
                    reward, done, trun, info = self.listeners[i].recv()
                else:
                    # no data arrived in time
                    raise TimeoutError(f"Slave {i} did not respond within {self.timeout}s")

                rewards.append(reward)
                dones.append(done)
                truns.append(trun)

                if done or trun:
                    # Grace period for reset: 2 frames before + 4 frames after state load + buffer
                    # Slave does ~6 frame advances during reset, with frameskip=4 that's ~2 env steps
                    # Add extra buffer for state load I/O latency
                    self.is_resetting[i] = 20

                infos["final_observation"].append(None)
                if self.firsts[i]:
                    self.firsts[i] = False
                    infos["First"][i] = True

            except Exception:
                # mark dead immediately

                rewards.append(0.)
                dones.append(False)
                truns.append(True)
                infos["final_observation"].append(np.copy(self.states[i]))

                self.restart_instance(i)

                continue

        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        truns = np.array(truns, dtype=np.bool_)

        # infos is meant to stay a list of dicts in gymnasium API

        # read states from shared memory
        states = np.copy(self.states)

        # convert info into correct form
        infos["final_observation"] = np.stack(
            [x if x is not None else np.zeros_like(self.states[0])
             for x in infos["final_observation"]],
            axis=0
        )

        # these should all be a batch of (num_envs)
        return states, rewards, dones, truns, infos

    def restart_instance(self, i):
        """Kill env i and then re‐call create_dolphin(i)."""
        print(f"[Master] Restarting Dolphin env {i}…")

        # kill the Python wrapper process
        p = self.processes[i]
        if p and p.poll() is None:
            try:
                p.kill()
                p.wait(timeout=5)
            except:
                pass

        # Kill the script process (cross-platform)
        try:
            import platform
            if platform.system() == "Windows":
                subprocess.check_output("Taskkill /PID %d /F" % self.script_pids[i])
            else:
                # macOS/Linux: use kill
                os.kill(self.script_pids[i], signal.SIGKILL)
            print("Minor Crash... Recovering successfully")
        except Exception as e:
            print(f"Failed to kill script PID {self.script_pids[i]}: {e}")

        # 3) close old socket
        old = self.listeners[i]
        if old:
            try: old.close()
            except: pass
        self.listeners[i] = None

        # wait to allow the process to die
        time.sleep(0.3)

        # 4) re‐launch and accept
        self.create_dolphin(i)
        print(f"[Master] Dolphin env {i} restarted!")



if __name__ == "__main__":
    current_keys = set()
    action = 0
    def on_press(key):
        global action
        try:
            if key is not None:
                if hasattr(key, 'char') and key.char:
                    current_keys.add(key.char)

            if 't' in current_keys:
                action = 1
            elif 'y' in current_keys:
                action = 4
            elif 'r' in current_keys:
                action = 5
            elif 'e' in current_keys:
                action = 4
            elif 'i' in current_keys:
                action = 3
            elif 'h' in current_keys:
                action = 7

            elif 'g' in current_keys:
                action = 8

        except Exception as e:
            print(f"Error in on_press: {e}")


    def on_release(key):
        global action
        try:
            if hasattr(key, 'char') and key.char:
                current_keys.discard(key.char)

            on_press(None)

            # if 'w' in current_keys:
            #     action = 6
            # elif 'a' in current_keys:
            #     action = 2
            # elif 'd' in current_keys:
            #     action = 3
            # elif 'q' in current_keys:
            #     action = 4
            # elif 'e' in current_keys:
            #     action = 5
            # else:
            #     action = 0
            if len(current_keys) == 0:
                action = 0

        except Exception as e:
            print(f"Error in on_release: {e}")

    num_envs = 1

    dolphin_env = DolphinEnv(
        num_envs=num_envs,
        gamename="LC",
        gamefile="mkw.iso"
    )

    from pynput import keyboard

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    tot_reward = 0
    while True:
        actions = np.array([action], dtype=int)
        dolphin_env.step_async(actions)

        states, rewards, dones, truns, infos = dolphin_env.step_wait()
        time.sleep(0.03)
        if abs(np.sum(rewards)) > 0:
            print(f"reward: {rewards}")
            tot_reward += rewards[0]

        if dones[0] or truns[0]:
            print(f"Total Episode Reward: {tot_reward}\n\n")
            tot_reward = 0


