"""
MuZero Trainer — Active Learning Loop
Part of Sovereign Stack V3.0 — Sprint 6

Handles the training loop for the MuZero model:
- Sampling from ReplayBuffer
- Unrolling the network (BPTT)
- Calculating Policy, Value, and Reward losses
- Optimization step
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import List

from eva_lab.muzero.agent import MuZeroAgent, GameHistory
from eva_lab.muzero.config import MuZeroConfigV3

logger = logging.getLogger(__name__)


class MuZeroTrainer:
    """
    Trainer for MuZero active learning.
    """

    def __init__(self, agent: MuZeroAgent):
        self.agent = agent
        self.config = agent.config
        self.optimizer = optim.Adam(
            self.agent.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.steps = 0

    def train_step(self, batch_size: int = None) -> dict:
        """
        Execute one training step (batch sample -> loss -> optimize).
        """
        if self.agent.replay_buffer.size < self.config.batch_size:
            return {"status": "waiting_for_data", "buffer": self.agent.replay_buffer.size}

        batch_size = batch_size or self.config.batch_size
        batch = self._sample_batch(batch_size)
        
        # Unroll and calculate loss
        total_loss, metrics = self._compute_loss(batch)

        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.steps += 1
        return metrics

    def _sample_batch(self, batch_size: int) -> List[tuple]:
        """
        Sample a batch of (game, start_index) tuples.
        """
        games = self.agent.replay_buffer.games
        game_indices = np.random.choice(len(games), batch_size)
        batch = []
        for g_idx in game_indices:
            game = games[g_idx]
            # Valid start index: essential to have enough steps to unroll?
            # Or we pad? For simplicity V3, we assume we pick a random start.
            # If game is shorter than unroll steps, we might need padding or simpler logic.
            # Here we pick a random position in the game.
            
            # We need at least 1 step to train? 
            # Let's say we pick a start index from 0 to len(game)-1
            start_idx = np.random.randint(0, len(game))
            batch.append((game, start_idx))
        return batch

    def _compute_loss(self, batch):
        """
        Compute the MuZero loss: Value + Reward + Policy + L2
        Unrolled for `num_unroll_steps`.
        """
        device = self.agent.device
        
        # Prepare inputs
        observations = []
        actions = []      # Actions to take at each step k
        target_values = []
        target_rewards = []
        target_policies = []
        
        num_unroll = self.config.num_unroll_steps
        
        # We need to construct tensors for the initial step (image) 
        # and then lists of targets for each unroll step.
        
        for game, start_idx in batch:
            # Observation at start_idx
            obs = torch.tensor(game.observations[start_idx], dtype=torch.float32)
            observations.append(obs)
            
            # For K steps
            game_actions = []
            game_vals = []
            game_rews = []
            game_pols = []
            
            for k in range(num_unroll + 1):
                curr_idx = start_idx + k
                
                # Targets (Value, Reward, Policy)
                if curr_idx < len(game):
                    # Value: bootstrapping or Monte Carlo return?
                    # MuZero uses n-step return or reanalyzed value. 
                    # Here we use the stored value (from MCTS or simple return).
                    # Simple MVP: use stored MCTS value.
                    game_vals.append(game.values[curr_idx])
                    game_pols.append(game.policies[curr_idx])
                    
                    if k > 0: # Reward is from step k-1 to k
                        game_rews.append(game.rewards[curr_idx])
                        game_actions.append(game.actions[curr_idx]) 
                    else:
                        # Step 0: No reward prediction needed for "current state" in dynamics? 
                        # Dynamics predicts r, s_next given s, a.
                        # So for step 0, we predict reward for action a_0?
                        # Wait, dynamics(s_0, a_0) -> s_1, r_1.
                        # So we need a_0 to rollout.
                        pass
                        
                else:
                    # Pad / terminal state
                    game_vals.append(0.0)
                    game_pols.append([1/5]*5) # Uniform generic
                    if k > 0:
                        game_rews.append(0.0)
                        game_actions.append(np.random.randint(0, 5)) # Dummy action
            
            # Actions needed for dynamics: a_0, a_1, ... a_{k-1}
            # If we are at step 0, we use action 0 to get to step 1.
            # The bootstrap unroll loop:
            # s0 = repr(o0)
            # v0, p0 = pred(s0) (Loss vs target_v0, target_p0)
            # s1, r1 = dyn(s0, a0) (Action taken at step 0)
            # v1, p1 = pred(s1) (Loss vs target_v1, target_p1, r1 vs target_r1)
            
            # So we need actions[start_idx ... start_idx + unroll - 1]
            act_seq = []
            for k in range(num_unroll):
                idx = start_idx + k
                if idx < len(game):
                    act_seq.append(game.actions[idx])
                else:
                    act_seq.append(np.random.randint(0, 5)) # Dummy
            
            actions.append(act_seq)
            target_values.append(game_vals)     # Length K+1
            target_policies.append(game_pols)   # Length K+1
            
            # Rewards shift: r1 corresponds to transition 0->1
            # We need rewards targets for steps 1..K
            # In game history, rewards[i] is reward received AFTER action[i] (transition to i+1)?
            # Or reward received AT step i? 
            # Convention: step(a) -> obs', reward, done.
            # So rewards[i] is reward for action[i].
            # Dynamics(s_i, a_i) -> r_i, s_{i+1}
            # So target reward for step k (0->1) is game.rewards[start_idx + k]
            rew_seq = []
            for k in range(num_unroll):
                idx = start_idx + k
                if idx < len(game):
                    rew_seq.append(game.rewards[idx])
                else:
                    rew_seq.append(0.0)
            target_rewards.append(rew_seq)

        # Convert to tensors
        observations = torch.stack(observations).to(device) # [B, Obs]
        actions = torch.tensor(actions, dtype=torch.long).to(device) # [B, K]
        target_values = torch.tensor(target_values, dtype=torch.float32).to(device) # [B, K+1]
        target_rewards = torch.tensor(target_rewards, dtype=torch.float32).to(device) # [B, K]
        target_policies = torch.tensor(target_policies, dtype=torch.float32).to(device) # [B, K+1, ActionDim]

        # ── Unroll Loop ──
        
        loss_val = 0
        loss_rew = 0
        loss_pol = 0
        
        # Initial step
        hidden_state = self.agent.network.representation(observations)
        policy_pred, value_pred = self.agent.network.prediction(hidden_state)
        
        # Step 0 Losses
        loss_val += F.mse_loss(value_pred.squeeze(-1), target_values[:, 0])
        loss_pol += self._policy_loss(policy_pred, target_policies[:, 0])
        
        gradient_scale = 1.0 / num_unroll
        
        for k in range(num_unroll):
            # Dynamics
            # Action: need one-hot
            action_indices = actions[:, k] # [B]
            # Convert to one-hot [B, ActionDim]
            action_onehot = F.one_hot(action_indices, num_classes=self.config.action_space_size).float()
            
            hidden_state, reward_pred = self.agent.network.dynamics(hidden_state, action_onehot)
            policy_pred, value_pred = self.agent.network.prediction(hidden_state)
            
            # Scale gradient for dynamics (hook)
            hidden_state.register_hook(lambda grad: grad * 0.5) 

            # Losses
            loss_rew += F.mse_loss(reward_pred.squeeze(-1), target_rewards[:, k])
            loss_val += F.mse_loss(value_pred.squeeze(-1), target_values[:, k+1])
            loss_pol += self._policy_loss(policy_pred, target_policies[:, k+1])

        # Total Loss
        total_loss = (loss_val + loss_rew + loss_pol)
        
        metrics = {
            "loss_total": total_loss.item(),
            "loss_val": loss_val.item(),
            "loss_rew": loss_rew.item(),
            "loss_pol": loss_pol.item()
        }
        return total_loss, metrics

    def _policy_loss(self, pred, target):
        """Cross Entropy between predicted softmax and target distribution."""
        # pred is already softmaxed in network output?
        # PredictionNetwork.policy_head uses Softmax(dim=1).
        # So we should use BCELoss or similar? 
        # Typically: sum(-target * log(pred))
        return -torch.sum(target * torch.log(pred + 1e-8), dim=1).mean()
