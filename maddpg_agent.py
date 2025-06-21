import numpy as np
import copy
from collections import namedtuple, deque
import random
from typing import Tuple, List, Optional
from dataclasses import dataclass
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

@dataclass
class AgentConfig:
    """Configuration class for MADDPG Agent hyperparameters."""
    buffer_size: int = int(1e5)
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 1e-3
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    weight_decay: float = 0
    update_every: int = 1
    num_updates: int = 1
    noise_decay: float = 0.999
    noise_min: float = 0.01
    grad_clip_norm: float = 1.0
    prioritized_replay: bool = True
    alpha: float = 0.6  # PER alpha
    beta: float = 0.4   # PER beta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Enhanced MADDPG Agent with multiple improvements."""
    
    def __init__(self, state_size: int, action_size: int, num_agents: int, 
                 random_seed: int, config: Optional[AgentConfig] = None):
        """Initialize an Agent object with enhanced features.
        
        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
            num_agents: Number of agents in the environment
            random_seed: Random seed for reproducibility
            config: Configuration object with hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.config = config or AgentConfig()
        self.seed = np.random.seed(random_seed)
        
        # Initialize networks
        self._init_networks(random_seed)
        
        # Noise process with decay
        self.noise = OUNoise(action_size, random_seed)
        self.noise_scale = 1.0
        
        # Replay memory
        if self.config.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                action_size, self.config.buffer_size, 
                self.config.batch_size, random_seed,
                alpha=self.config.alpha, beta=self.config.beta
            )
        else:
            self.memory = ReplayBuffer(
                action_size, self.config.buffer_size,
                self.config.batch_size, random_seed
            )
        
        # Learning control
        self.t_step = 0
        
        # Metrics tracking
        self.actor_losses = []
        self.critic_losses = []
        
    def _init_networks(self, random_seed: int) -> None:
        """Initialize actor and critic networks."""
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), 
            lr=self.config.lr_actor
        )
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), 
            lr=self.config.lr_critic, 
            weight_decay=self.config.weight_decay
        )
        
        # Initialize target networks
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)
        
    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
             next_states: np.ndarray, dones: np.ndarray) -> None:
        """Save experience and learn if enough samples available."""
        # Save experiences for all agents
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0 and len(self.memory) > self.config.batch_size:
            for _ in range(self.config.num_updates):
                if self.config.prioritized_replay:
                    experiences, indices, weights = self.memory.sample()
                    td_errors = self.learn(experiences, self.config.gamma, weights)
                    self.memory.update_priorities(indices, td_errors)
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences, self.config.gamma)
    
    def act(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Return actions for given states as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            noise = self.noise.sample() * self.noise_scale
            actions += noise
            # Decay noise over time
            self.noise_scale = max(
                self.config.noise_min,
                self.noise_scale * self.config.noise_decay
            )
        
        return np.clip(actions, -1, 1)
    
    def reset(self) -> None:
        """Reset noise process."""
        self.noise.reset()
    
    def learn(self, experiences: Tuple, gamma: float, 
              importance_weights: Optional[torch.Tensor] = None) -> Optional[np.ndarray]:
        """Update policy and value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        if importance_weights is not None:
            # Prioritized Experience Replay weighted loss
            td_errors = Q_targets - Q_expected
            critic_loss = (importance_weights * td_errors.pow(2)).mean()
            # Store TD errors for priority update (detach and move to CPU)
            td_errors_np = td_errors.detach().cpu().numpy()
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            td_errors_np = None
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.critic_local.parameters(), 
                self.config.grad_clip_norm
            )
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_local.parameters(), 
                self.config.grad_clip_norm
            )
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)
        
        # Store losses for monitoring
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        if importance_weights is not None:
            return np.abs(td_errors_np.squeeze())
        return None
    
    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float) -> None:
        """Soft update model parameters using Polyak averaging."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
        """Hard update (copy) model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoints."""
        checkpoint = {
            'actor_local_state_dict': self.actor_local.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_local_state_dict': self.critic_local.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'noise_scale': self.noise_scale
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor_local.load_state_dict(checkpoint['actor_local_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_local_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'noise_scale' in checkpoint:
            self.noise_scale = checkpoint['noise_scale']


class OUNoise:
    """Enhanced Ornstein-Uhlenbeck process with better parameter control."""

    def __init__(self, size: int, seed: int, mu: float = 0.0, 
                 theta: float = 0.15, sigma: float = 0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        np.random.seed(seed)
        self.reset()

    def reset(self) -> None:
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Enhanced replay buffer with better memory management."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                   field_names=["state", "action", "reward", "next_state", "done"])
        np.random.seed(seed)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)
        
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer implementation."""
    
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, 
                 seed: int, alpha: float = 0.6, beta: float = 0.4):
        """Initialize prioritized replay buffer."""
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.experience = namedtuple("Experience", 
                                   field_names=["state", "action", "reward", "next_state", "done"])
        
        # Initialize memory and priority arrays
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        
        np.random.seed(seed)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add experience with maximum priority."""
        e = self.experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self) -> Tuple[Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
        """Sample batch according to priorities."""
        if len(self.memory) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # Calculate sampling probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)
        
        # Convert to tensors
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences])
        ).float().to(device)
        
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences])
        ).float().to(device)
        
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])
        ).float().to(device)
        
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences])
        ).float().to(device)
        
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)
        ).float().to(device)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current size of memory."""
        return len(self.memory)