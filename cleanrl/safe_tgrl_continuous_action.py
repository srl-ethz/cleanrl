# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import collections
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.ppo_continuous_action import Agent

"""
This implementation of the TGRL algorithm introduces a new automatically tuned coefficient,
beta, that controls how important student episode lengths/failures are in the combined reward.
The change should allow safe teacher guided learning on real hardware systems.

For clarity, I (@fbenedek) also added documentation to the TGRL implementation.
"""

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=6,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--checkpoint-frequency", type=int, default=10,
        help="model saving frequency")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v2",
        help="the id of the environment")
    parser.add_argument("--teacher-folder", type=str, default=None,
        help="the name of the folder containing the weights for the teacher agent")
    parser.add_argument("--teacher-coef", type=float, default=1.0,
        help="coefficient of the entropy")
    parser.add_argument("--teacher-coef-update", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--coefficient-frequency", type=int, default=1000,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--beta", type=float, default=0.2,
            help="Termination regularization coefficient.")
    parser.add_argument("--finetune-teacher", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if true, the actor network will start from the teacher and will be finetuned during training")
    parser.add_argument("--autotune_beta", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="switches off automatic tuning of the termination regularizer beta")
    parser.add_argument("--autotune_alpha", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--history_length", type=int, default=20, help="The number of previous rewards to take into account for TGRL update.")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        print('SET SNEED TO: ', seed)
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, torch.distributions.Normal(mean, std)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    actor_aux = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    qf1_kl = SoftQNetwork(envs).to(device)
    qf2_kl = SoftQNetwork(envs).to(device)
    qf1_target_kl = SoftQNetwork(envs).to(device)
    qf2_target_kl = SoftQNetwork(envs).to(device)
    qf1_target_kl.load_state_dict(qf1_kl.state_dict())
    qf2_target_kl.load_state_dict(qf2_kl.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()) + list(qf1_kl.parameters()) + list(qf2_kl.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()) + list(actor_aux.parameters()), lr=args.policy_lr)

    # Load the teacher
    assert args.teacher_folder is not None
    if args.finetune_teacher:
        print('Loading pretrained actor...')
        teacher = Actor(envs).to(device)
        teacher_dir = os.path.join(os.getcwd(), 'wandb', args.teacher_folder, "files", "actor.pt")
        actor.load_state_dict(torch.load(teacher_dir, map_location=device))
        print('Loaded pretrained actor!')
    else:
        teacher = Agent(envs).to(device)
        teacher_dir = os.path.join(os.getcwd(), 'wandb', args.teacher_folder, "files", "agent.pt")
    teacher.load_state_dict(torch.load(teacher_dir, map_location=device))
    teacher.eval()
    for i, param in enumerate(teacher.parameters()):
            param.requires_grad_(False)

    # Automatic entropy tuning
    if args.autotune_alpha:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    teacher_coef = args.teacher_coef
    current_actor = actor
    actor_performance = collections.deque(args.history_length * [0], args.history_length)
    actor_aux_performance = collections.deque(args.history_length*[0], args.history_length)
    performance_difference = 0
    # forces obs space as NotImplementedError is thrown by get_obs_shape otherwise
    print(envs.single_observation_space)
    # envs.single_observation_space.dtype = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(11,), dtype=np.float32)
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions_random = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            if args.finetune_teacher:
                actions_teacher, _, _, _ = teacher.get_action(torch.Tensor(obs).to(device))
                actions_teacher = actions_teacher.detach().cpu().numpy()
            else:
                actions_teacher = np.array(teacher.get_action_and_value(torch.Tensor(obs).to(device))[0].detach().cpu())

            # for elem in action_and_val_teacher:
            #     print('teacher action shape: ', elem.shape)
            actions = actions_teacher


        else:
            actions, _, _, _ = current_actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                if current_actor is actor:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    actor_performance.appendleft(info["episode"]["r"])
                    current_actor = actor_aux
                else:
                    writer.add_scalar("charts/episodic_return_aux", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length_aux", info["episode"]["l"], global_step)
                    actor_aux_performance.appendleft(info["episode"]["r"])
                    current_actor = actor
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # for idx, d in enumerate(dones):
        #     if d:
        #         real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, student_dist_next_obs = actor.get_action(data.next_observations.float())
                qf1_next_target = qf1_target(data.next_observations.float(), next_state_actions.float())
                qf2_next_target = qf2_target(data.next_observations.float(), next_state_actions.float())
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                if args.finetune_teacher:
                    _, _, _, teacher_dist_next_obs = teacher.get_action(data.next_observations.float())
                else:
                    _, _, _, _, teacher_dist_next_obs = teacher.get_action_and_value(data.next_observations.float())
                kl_next_obs = kl_divergence(student_dist_next_obs, teacher_dist_next_obs).sum(1)
                next_q_kl_value = (1 - data.dones.flatten()) * args.gamma * ((min_qf_next_target).view(-1) + (kl_next_obs).view(-1))

            qf1_a_values = qf1(data.observations.float(), data.actions.float()).view(-1)
            qf2_a_values = qf2(data.observations.float(), data.actions.float()).view(-1)
            qf1_kl_a_values = qf1_kl(data.observations.float(), data.actions.float()).view(-1)
            qf2_kl_a_values = qf2_kl(data.observations.float(), data.actions.float()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf1_kl_loss = F.mse_loss(qf1_kl_a_values, next_q_kl_value)
            qf2_kl_loss = F.mse_loss(qf2_kl_a_values, next_q_kl_value)
            qf_loss = qf1_loss + qf2_loss + qf1_kl_loss + qf2_kl_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _, student_dist = actor.get_action(data.observations.float())
                    pi_aux, log_pi_aux, _, _ = actor_aux.get_action(data.observations.float())
                    if args.finetune_teacher:
                        # removed an argument from here as I'll work with a pretrained Actor network as teacher
                        _, _, _, teacher_dist = teacher.get_action(data.observations.float())
                    else:
                        _, _, _, _, teacher_dist = teacher.get_action_and_value(data.observations.float())
                    cross_entropy = kl_divergence(student_dist, teacher_dist).sum(1)
                    qf1_pi = qf1(data.observations.float(), pi)
                    qf2_pi = qf2(data.observations.float(), pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    qf1_pi_aux = qf1(data.observations.float(), pi_aux)
                    qf2_pi_aux = qf2(data.observations.float(), pi_aux)
                    min_qf_pi_aux = torch.min(qf1_pi_aux, qf2_pi_aux).view(-1)
                    qf1_kl_pi = qf1_kl(data.observations.float(), pi)
                    qf2_kl_pi = qf2_kl(data.observations.float(), pi)
                    min_qf_kl_pi = torch.min(qf1_kl_pi, qf2_kl_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_loss_aux = ((alpha * log_pi_aux) - min_qf_pi_aux).mean()
                    cross_entropy_loss = (cross_entropy - min_qf_kl_pi).mean()
                    overall_loss = actor_loss + teacher_coef * cross_entropy_loss + actor_loss_aux

                    actor_optimizer.zero_grad()
                    overall_loss.backward()
                    actor_optimizer.step()

                    if args.autotune_alpha:
                        with torch.no_grad():
                            _, log_pi, _, _ = actor.get_action(data.observations.float())
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.coefficient_frequency == 0:
                performance_difference = np.mean(actor_performance) - np.mean(actor_aux_performance)
                if performance_difference > 0:
                    teacher_coef = teacher_coef + args.teacher_coef_update
                else:
                    teacher_coef = teacher_coef - args.teacher_coef_update
                if teacher_coef < 0:
                    teacher_coef = 0

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1_kl.parameters(), qf1_target_kl.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2_kl.parameters(), qf2_target_kl.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # save actor so we can reuse it later and start finetuning
            if global_step % 1000 == 0:
                torch.save(actor.state_dict(), f"runs/{run_name}/actor.pt")

            if global_step % 100 == 0:

                writer.add_scalar("losses/performance_difference", performance_difference, global_step)
                writer.add_scalar("losses/qf1_kl_loss", qf1_kl_loss.mean().item(), global_step)
                writer.add_scalar("losses/qf2_kl_loss", qf2_kl_loss.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 4.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss_aux", actor_loss_aux.item(), global_step)
                writer.add_scalar("losses/cross_entropy_loss", cross_entropy_loss.item(), global_step)
                writer.add_scalar("losses/cross_entropy", cross_entropy.mean().item(), global_step)
                writer.add_scalar("losses/teacher_coef", teacher_coef, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune_alpha:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
