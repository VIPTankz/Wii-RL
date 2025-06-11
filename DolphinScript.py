print("Script Started!")
from dolphin import event
import sys
import os
import inspect
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__name__)))
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
shared_site_path = Path(script_directory) / "shared_site.txt"

# add libraries from your python install (needs to match dolphin version (currently 3.12))
if shared_site_path.exists() and shared_site_path.is_file():
    with open(shared_site_path, 'r', encoding='utf-8') as file:
        site_path = file.read()
    
    sys.path.append(site_path)

# the cwd is the dir of the instance not the root folder of the project
cwd = os.path.dirname(os.getcwd())
scripts_dir = os.path.abspath(os.path.join(cwd, "scripts"))
print(f"Added scripts directory to sys.path: {scripts_dir}")
sys.path.append(scripts_dir)
try:
    import numpy as np
    from dolphin_instance import DolphinInstance
except Exception as e:
    print(e)
    raise Exception("stop")

save_states_path = script_directory + f"\\MarioKartSaveStates\\"

instance_info_folder = Path('instance_info')

# Read pid from pid_num.txt
pid = int((instance_info_folder / 'pid_num.txt').read_text().strip())

# Read our specific ID
id = int((instance_info_folder / f'instance_id{pid}.txt').read_text().strip())

# Write our own PID into script_pid{id}.txt
(instance_info_folder / f'script_pid{id}.txt').write_text(str(os.getpid()))

num_envs = int((instance_info_folder / 'num_envs.txt').read_text().strip())
FILE_PATH = Path.cwd() / "shared_value.txt"

if not FILE_PATH.exists():
    FILE_PATH = Path.cwd().parent / "shared_value.txt"

print(f"FILE_PATH: {FILE_PATH}")

for i in range(4):
    await event.frameadvance()

env = DolphinInstance(id=id, num_envs=num_envs,save_states_path=save_states_path, share_value_path=FILE_PATH)

(width, height, data) = await event.framedrawn()

print("Processing init state...")

img_np = env.process_indiv_frame(width, height, data)
img_stack = np.array([img_np for _ in range(env.frameskip)])

env.send_init_state(img_stack)

print("Sent init state")

def my_callback():
    env.apply_action(env.applied_action)
# make sure we apply the action every single frame. Otherwise this can lead to some weird stuttering behavior
event.on_frameadvance(my_callback)

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
            new_img = env.process_indiv_frame(width, height, data)
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
            new_img = env.process_indiv_frame(width, height, data)
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
