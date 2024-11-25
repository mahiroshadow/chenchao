"""
运行主函数
"""
import torch
import numpy as np
import torch.nn.functional as F
from module import TD3,Buffer
from parser import args
from env import UAVTASKENV,create_UE_cluster

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def train(env:UAVTASKENV,model:TD3,memory:Buffer,n_epoches:int,device:str):
    timestamp=200
    score=0
    total_steps=0
    episode_step=0
    for epoch in range(n_epoches):
        obs = env.reset()
        score = 0
        episode_step = 0
        for ts in range(timestamp):
            # for idx,uav in enumerate(obs):
            uav=torch.from_numpy(obs[0]).to(device)
            action=model.actor(uav)
            actions=[action.detach().cpu().numpy()]
            obs_, reward, done, info = env.step(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            actor_states, states, actions, rewards,actor_new_states, states_ , not_done= memory.sample_buffer()
            states,rewards,states_ , not_done,actions=torch.from_numpy(states).float().to(device),torch.from_numpy(rewards).float().to(device),\
            torch.from_numpy(states_).float().to(device),torch.from_numpy(not_done).float().to(device),torch.from_numpy(actions[0]).float().to(device)
            with torch.no_grad():
                noise = (torch.randn_like(action) * model.policy_noise).clamp(-model.noise_clip, model.noise_clip)
                # print(states_.dtype)
                next_action = (model.actor_target(states_) + noise).clamp(-model.max_action, model.max_action)
                target_Q1, target_Q2 = model.critic_target(states_, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards+ not_done * model.discount * target_Q
            current_Q1, current_Q2 = model.critic(states, actions)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            model.critic_optimizer.zero_grad()
            critic_loss.backward()
            model.critic_optimizer.step()
            if total_steps % 10 == 0:
                actor_loss = -model.critic.Q1(states, model.actor(states)).mean()
                model.actor_optimizer.zero_grad()
                actor_loss.backward()
                model.actor_optimizer.step()
            for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                target_param.data.copy_(model.tau * param.data + (1 - model.tau) * target_param.data)
            for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
                target_param.data.copy_(model.tau * param.data + (1 - model.tau) * target_param.data)

            obs = obs_
            score += reward
            total_steps += 1
            episode_step += 1
        print(f"score:{score}")

def test():
    pass


if __name__ == '__main__':
    n_agents = args.n_agents
    n_actions = args.n_actions
    n_epoches = args.n_epoches
    device = args.device
    ue_cluster_1 = create_UE_cluster(400, 450, 470, 520)
    ue_cluster_2 = create_UE_cluster(30,30,100,100)
    env = UAVTASKENV(ue_cluster_1, ue_cluster_2)
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(3)
    critic_dims = sum(actor_dims)
    model=TD3(3,n_actions,2.0)
    memory = Buffer(1000000, critic_dims, actor_dims,
                        n_actions, n_agents, batch_size=100)
    train(env,model,memory,n_epoches,device)
    