# Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

## Project Overview

This project implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train two agents to play tennis collaboratively. The agents learn to keep a ball in play by hitting it over a net, demonstrating emergent cooperative behavior through continuous action control.

### Environment Details

**Goal**: Train two tennis players to bounce a ball over a net for as many time steps as possible, achieving an average score of +0.5 over 100 consecutive episodes.

**Reward Structure**:
- +0.1 for each step that an agent hits the ball over the net
- -0.01 if an agent lets the ball hit the ground or hits it out of bounds

**State Space**: 
- 8 variables per agent corresponding to position and velocity of ball and racket
- Each agent receives its own local observation
- Total observation space: 24 dimensions (8 × 3 observations per agent)

**Action Space**:
- 2 continuous actions per agent
- Movement toward/away from the net
- Jumping

**Success Criteria**: The environment is considered solved when the agents achieve an average score of +0.5 over 100 consecutive episodes.

## Algorithm Implementation

### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

MADDPG extends the DDPG algorithm to multi-agent environments by using:
- **Centralized Training**: Agents share experiences during learning
- **Decentralized Execution**: Each agent acts independently using its own policy
- **Actor-Critic Architecture**: Each agent maintains both policy (actor) and value (critic) networks

### Key Components

#### Agent Architecture (`maddpg_agent.py`)
- Single agent managing multiple environment entities
- **Prioritized Experience Replay (PER)** for improved sample efficiency
- Shared experience replay buffer for all agents
- Ornstein-Uhlenbeck noise process for exploration
- Soft target network updates for training stability

#### Neural Network Models (`model.py`)

**Actor Network (Policy)**:
- Input: State (24 dimensions)
- Hidden Layers: 100 → 75 → 75 nodes
- Activation: ReLU (with batch normalization on first layer)
- Output: 2 continuous actions (tanh activation, bounded [-1, 1])

**Critic Network (Value Function)**:
- Input: State (24 dimensions) + Actions (4 dimensions from both agents)
- Hidden Layers: 100 → 100 nodes
- Activation: ReLU (with batch normalization on first layer)
- Output: Single Q-value (no activation)
- Architecture: Late fusion (state processed first, then concatenated with actions)

### Training Process

1. **Environment Reset**: Initialize episode with random starting positions
2. **Action Selection**: Each agent selects actions using current policy + exploration noise
3. **Environment Step**: Execute actions and observe rewards, next states, and done signals
4. **Experience Storage**: Store transitions in shared replay buffer
5. **Learning**: Sample mini-batches and update actor-critic networks
6. **Target Network Updates**: Soft updates to target networks for stability

## Hyperparameters

```python
BUFFER_SIZE = 100,000      # Replay buffer size
BATCH_SIZE = 128           # Mini-batch size
GAMMA = 0.99              # Discount factor
TAU = 1e-3                # Soft update parameter
LR_ACTOR = 1e-4           # Actor learning rate
LR_CRITIC = 1e-3          # Critic learning rate
WEIGHT_DECAY = 0          # L2 weight decay
PRIORITIZED_REPLAY = True  # Use Prioritized Experience Replay
ALPHA = 0.6               # PER alpha (prioritization strength)
BETA = 0.4                # PER beta (importance sampling)
```

## Results

The agent successfully solved the environment in **1,260 episodes**, achieving an average score of +0.5040 over 100 consecutive episodes at episode 1,360.

### Training Performance

![Training Results](img.png)

*Figure: Episode scores (blue), smoothed scores (dark blue), and running mean over 100 episodes (red). The green dashed line indicates the target score of 0.5.*

### Training Progress Milestones
- **Episodes 1-800**: Initial exploration phase with minimal learning (≈0.000 average)
- **Episode 900**: First signs of learning (0.0066 average)
- **Episode 1000**: Steady improvement begins (0.0376 average)
- **Episode 1100**: Learning accelerates (0.0639 average)
- **Episode 1300**: Breakthrough performance (0.2480 average)
- **Episode 1360**: **Environment solved!** (0.5040 average)

### Analysis of Training Results

**Characteristic Learning Pattern**: The training exhibited a typical multi-agent learning curve with three distinct phases:

1. **Exploration Phase (Episodes 1-800)**: Extended period of minimal rewards as agents learn basic environment dynamics
2. **Learning Phase (Episodes 900-1200)**: Gradual improvement as agents discover cooperative strategies  
3. **Breakthrough Phase (Episodes 1300+)**: Rapid convergence once effective collaboration emerges

**Why MADDPG with PER Succeeded**:
- **Prioritized learning**: PER helped focus on important experiences, accelerating learning
- **Multi-agent complexity**: Two agents must learn to coordinate actions simultaneously
- **Sparse rewards**: Tennis environment provides limited positive feedback initially
- **Continuous action space**: More challenging than discrete actions
- **Cooperative requirement**: Both agents must perform well for sustained rallies

The use of Prioritized Experience Replay likely contributed to the relatively efficient learning (1,260 episodes), as it allows the agent to replay the most informative experiences more frequently. The final breakthrough at episode 1300 demonstrates the sudden emergence of cooperative behavior typical in multi-agent reinforcement learning.

## File Structure

```
├── maddpg_agent.py                # Main agent implementation with PER
├── model.py                       # Neural network architectures
├── Tennis.ipynb                   # Training notebook
├── checkpoint_actor.pth           # Latest actor weights
├── checkpoint_critic.pth          # Latest critic weights
├── solved_checkpoint_actor.pth    # Solved actor weights
├── solved_checkpoint_critic.pth   # Solved critic weights
├── img.png                        # Training results plot
└── README.md                      # This file
```

## Usage

### Training Implementation
```python
from maddpg_agent import Agent

# Initialize agent
agent = Agent(state_size=state_size, action_size=action_size, 
              num_agents=num_agents, random_seed=1)

def maddpg(n_episodes=6000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    mean_scores = []   
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        
        for t in range(max_t):
            # Get actions from both agents
            action1 = agent.act(state[0])
            action2 = agent.act(state[1])
            action = np.concatenate((action1, action2), axis=0)
            action = np.clip(action, -1, 1)
            
            # Environment step
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            
            # Agent learning step
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if np.any(done):
                break
        
        # Track scores and save checkpoints
        scores_deque.append(np.max(score))
        scores.append(np.max(score))    
        mean_scores.append(np.mean(scores_deque))
        
        # Save checkpoints every episode
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        
        # Print progress
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(
            i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(
                i_episode, np.mean(scores_deque)))
        
        # Check if solved
        if np.mean(scores_deque) > 0.5:
            torch.save(agent.actor_local.state_dict(), 'solved_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'solved_checkpoint_critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(
                i_episode-100, np.mean(scores_deque)))
            break
    
    return scores, mean_scores

# Run training
scores, mean_scores = maddpg()
```

### Testing
```python
# Initialize agent and load trained weights
agent = Agent(state_size=state_size, action_size=action_size, 
              num_agents=num_agents, random_seed=1)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

# Reset environment for testing
env_info = env.reset(train_mode=False)[brain_name]     
states = env_info.vector_observations                  
scores = np.zeros(num_agents)                          

while True:
    # Get actions from both agents (no exploration noise)
    action1 = agent.act(states[0])
    action2 = agent.act(states[1])
    action = np.concatenate((action1, action2), axis=0)
    action = np.clip(action, -1, 1)             
    
    # Environment step
    env_info = env.step(action)[brain_name]           
    next_states = env_info.vector_observations         
    rewards = env_info.rewards                         
    dones = env_info.local_done                        
    
    # Update scores and states
    scores += env_info.rewards                         
    states = next_states                               
    
    if np.any(dones):                                  
        break

print('Score (max over agents) from this episode: {}'.format(np.max(scores)))
# Output: Score (max over agents) from this episode: 2.600000038743019
```

## Key Features

### Enhanced Implementation
- **Type Hints**: Full type annotations for better code clarity
- **Configuration Management**: Centralized hyperparameter control
- **Error Handling**: Robust CUDA tensor management
- **Modular Design**: Clean separation of concerns
- **Checkpointing**: Save/load model states for reproducibility

### Algorithm Improvements
- **Prioritized Experience Replay (PER)**: Focuses learning on important experiences with high TD errors
- **Proper Weight Initialization**: Xavier initialization for stable training
- **Gradient Clipping**: Prevents exploding gradients
- **Batch Normalization**: Improved training stability
- **Exploration Decay**: Noise reduction over time for better convergence

## Future Work

### Potential Improvements
1. **Prioritized Experience Replay (PER)**: Focus learning on important experiences
2. **Multi-Agent PPO (MAPPO)**: Alternative policy gradient approach
3. **Hyperparameter Tuning**: Systematic optimization of learning parameters
4. **Network Architecture**: Experiment with deeper networks or attention mechanisms
5. **Curriculum Learning**: Progressive difficulty increase during training

### Research Directions
- **Communication Protocols**: Explicit agent-to-agent communication
- **Hierarchical Multi-Agent RL**: Decompose complex tasks into subtasks
- **Transfer Learning**: Apply learned policies to new multi-agent environments
- **Robustness Testing**: Evaluate performance under various conditions

## Dependencies

- Python 3.6+
- PyTorch 1.0+
- NumPy
- Matplotlib
- Unity ML-Agents

## References

1. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
2. [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
3. [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)

---

*This implementation demonstrates the power of multi-agent reinforcement learning in continuous control tasks, showing how independent agents can learn to cooperate through shared experience and decentralized execution.*
