import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import vista

#choose the dimensions appropriately
state_dim = 4
hidden_dim = 5
action_dim = 2


def tensor(shape, flag=False):
    if flag:
        val = torch.zeros(shape)
    #set the standard deviation
    else:
        std = 1.0 / np.sqrt(shape[0])
        val = torch.randn(shape) * std
    return val.detach().requires_grad_(True)

model = {'w1': tensor((state_dim, hidden_dim)),'b1': tensor((hidden_dim,), flag=True),'w_actor': tensor((hidden_dim, action_dim)),'b_actor': tensor((action_dim,), flag=True),'w_critic': tensor((hidden_dim, 1)),'b_critic': tensor((1,), flag=True),'log_std': torch.zeros(action_dim, requires_grad=True)}

optimizer = optim.Adam(model.values(), lr=1e-4)



def get_predictions(state):
    #use the matmul module for multiplication
    h_linear = torch.matmul(state, model['w1']) + model['b1']
    h = torch.tanh(h_linear)
    
    mu_linear = torch.matmul(h, model['w_actor']) + model['b_actor']
    mu = torch.tanh(mu_linear)
    std = torch.exp(model['log_std'])
    
    value = torch.matmul(h, model['w_critic']) + model['b_critic']
    
    return Normal(mu, std), value

trace_path = "/Users/riddhi/Desktop/aiea/vista_traces/20210726-131322_lexus_devens_center"
world = vista.World([trace_path])
car = world.spawn_agent({})
world.reset()

epilson = 0.2     
gamma = 0.99 
t = 100       
iterations = 200
reward_history = []




for i in range(iterations):
    states = []
    actions = [] 
    rewards = [] 
    prob = [] 
    values = []
    for j in range(t):
        if car.done:
            world.reset()
        
        #create a multidimensional array
        vector = torch.tensor([car.relative_state.x, car.relative_state.yaw, car.speed, car.ego_dynamics.steering], dtype=torch.float32)
    
        dist, val =  get_predictions(vector)
        act = dist.sample()
        lp = dist.log_prob(act).sum(-1) 
        
        #move car
        car.step_dynamics(act.detach().numpy().flatten())
        #use the reward formula: 1 -|x|
        reward = 1.0 - abs(car.relative_state.x) 
        #add to the lists
        states.append(vector)
        actions.append(act)
        rewards.append(reward)
        prob.append(lp.detach())
        values.append(val.detach())
    s_batch = torch.stack(states)
    a_batch = torch.stack(actions)
    old_lp_batch = torch.stack(prob).view(-1)
    v_batch = torch.stack(values).view(-1)
    returns = []
    g = 0
    for r in reversed(rewards):
        g = r + (gamma * g)
        returns.insert(0, g)
    returns = torch.tensor(returns, dtype=torch.float32)
    advs = returns - v_batch
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    new_dist, new_val = get_predictions(s_batch)
    new_lp = new_dist.log_prob(a_batch).sum(-1).view(-1)
    ratio = torch.exp(new_lp - old_lp_batch) 
    #use the surrogate clip model
    surrogate1 = ratio * advs
    #set the range using torch.clamp
    surrogate2 = torch.clamp(ratio, 1.0 - epilson, 1.0 + epilson) * advs
    #compute the policy loss 
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    value_loss = 0.5 * (new_val.view(-1) - returns).pow(2).mean()
    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()
    
    reward_history.append(np.mean(rewards))
    if i % 5 == 0:
        x = np.mean(rewards)
        print(f"Iteration {i} | Average Reward: {x:.4f}")

plt.plot(reward_history)
plt.title("PPO Algorithim")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()
