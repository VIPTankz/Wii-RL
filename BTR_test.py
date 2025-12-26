import time
import argparse
import multiprocessing as mp
import numpy as np
import torch
from torchsummary import summary
from DolphinEnv import DolphinEnv

from BTR import Agent, non_default_args, format_arguments

def main():
    parser = argparse.ArgumentParser()

    # environment setup
    parser.add_argument('--game', type=str, default="MarioKart")

    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--frames', type=int, default=10000)
    parser.add_argument('--eval_envs', type=int, default=1)

    parser.add_argument('--bs', type=int, default=256)

    parser.add_argument('--repeat', type=int, default=0)

    parser.add_argument('--framestack', type=int, default=4)

    # agent setup
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--nstep', type=int, default=3)
    parser.add_argument('--maxpool_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--munch_alpha', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=int, default=10)

    parser.add_argument('--spectral', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--taus', type=int, default=8)
    parser.add_argument('--c', type=int, default=500)
    parser.add_argument('--linear_size', type=int, default=512)
    parser.add_argument('--model_size', type=float, default=2)

    parser.add_argument('--ncos', type=int, default=64)
    parser.add_argument('--per_alpha', type=float, default=0.2)
    parser.add_argument('--per_beta_anneal', type=int, default=0)
    parser.add_argument('--layer_norm', type=int, default=0)
    parser.add_argument('--eps_steps', type=int, default=2000000)
    parser.add_argument('--eps_disable', type=int, default=1)

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)


    game = args.game
    envs = args.envs
    bs = args.bs
    c = args.c
    lr = args.lr
    framestack = args.framestack
    device_name = args.device
    nstep = args.nstep
    maxpool_size = args.maxpool_size
    munch_alpha = args.munch_alpha
    grad_clip = args.grad_clip
    spectral = args.spectral
    discount = args.discount
    linear_size = args.linear_size
    taus = args.taus
    model_size = args.model_size
    frames = args.frames // 4
    ncos = args.ncos
    per_alpha = args.per_alpha
    eps_steps = args.eps_steps
    eps_disable = args.eps_disable
    layer_norm = args.layer_norm
    model_path = args.model_path

    agent_name = "BTR_test"

    replay_period = 64 / envs
    spi = 1

    if len(formatted_string) > 2:
        agent_name += '_' + formatted_string

    print("Agent Name:" + str(agent_name))

    num_envs = envs
    n_steps = frames

    print("Currently Playing Game: " + str(game))

    if(device_name == None):
        gpu = "0"
        device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"\nDevice: {device}")


    env = DolphinEnv(envs)
    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 75, 140], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=True, batch_size=bs, lr=lr,
                  maxpool_size=maxpool_size, target_replace=c, spectral=spectral, discount=discount, taus=taus,
                  model_size=model_size, linear_size=linear_size, ncos=ncos, replay_period=replay_period,
                  framestack=framestack, per_alpha=per_alpha, layer_norm=layer_norm,
                  eps_steps=eps_steps, eps_disable=eps_disable, n=nstep,
                  munch_alpha=munch_alpha, grad_clip=grad_clip, imagex=140, imagey=75, spi=spi)

    agent.load_models(model_path)

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for _ in range(num_envs)]
    scores = []
    observation, info = env.reset()
    processes = []

    # torchsummary only supports cuda/cpu, skip for mps
    if device.type in ('cuda', 'cpu'):
        summary(agent.net, (framestack, 75, 140), device=str(device))
    else:
        print(f"Skipping model summary (torchsummary doesn't support {device.type})")

    while steps < n_steps:
        steps += num_envs
        try:
            action = agent.choose_action(observation)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Observation: {observation}")
            raise Exception("Stop! Error Occurred")

        env.step_async(action)
        observation_, reward, done_, trun_, info = env.step_wait()

        for i in range(num_envs):
            scores_count[i] += reward[i]
            if done_[i] or trun_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0

        for stream in range(num_envs):

            if info["Ignore"][stream]:
                continue

            if info["First"][stream]:
                observation[stream] = observation_[stream]

            next_obs = observation_[stream] if not trun_[stream] else np.array(info["final_observation"][stream])

        observation = observation_

        if steps % 600 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} avg score {:.2f} total_timesteps {:.0f} fps {:.2f} games {}'
                      .format(agent_name, avg_score, steps,
                              (steps - last_steps) / (time.time() - last_time), episodes), flush=True)
                last_steps = steps
                last_time = time.time()

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
