import numpy as np
import matplotlib.pyplot as plt

# The mean and variance graph shows the average and spread of rewards over multiple runs, 
# while the learning curve tracks how the algorithm's performance improves over time during training

# Rewards Obtained for Q-learning and Policy Gradient
q_learning_rewards = [
    [0, 16, 0, 14, 8, 14, 12, 24, 0, 18, 20, 22, 24, 32, 34, 20, 24, 20, 30, 24, 24, 24, 26, 32, 30, 34, 26, 26, 28, 34],  # Q-learning rewards
    [8, 0, 0, 2, 12, 0, 16, 18, 12, 24, 20, 20, 20, 21, 26, 20, 14, 20, 27, 16, 22, 28, 20, 30, 30, 30, 24, 26, 28, 30],       # Q-learning rewards
    [0, 0, 0, 2, 12, 0, 12, 18, 0, 24, 20, 20, 20, 21, 28, 20, 14, 20, 27, 16, 22, 26, 18, 30, 32, 30, 24, 26, 28, 30]
]

policy_gradient_rewards = [
    [9, 17, 9, 12, 11, 15, 14, 9, 19, 19, 24, 30, 30, 28, 43, 34, 38, 41, 41, 54, 48, 52, 46, 48, 52, 48, 54, 54, 50, 54],
    [9, 0, 11, 9, 0, 0, 11, 11, 14, 18, 24, 19, 24, 30, 41, 38, 36, 41, 38, 48, 52, 50, 46, 46, 46, 48, 48, 54, 50, 54],
    [0, 9, 9, 12, 0, 15, 11, 11, 19, 19, 24, 30, 24, 28, 43, 32, 34, 40, 41, 50, 54, 48, 52, 46, 48, 52, 54, 54, 50, 54]
]

# Graph of rewards over episodes with mean and variance

# Calculate mean and variance for Q-learning rewards
q_learning_mean = np.mean(q_learning_rewards, axis=0)
q_learning_variance = np.var(q_learning_rewards, axis=0)

# Calculate mean and variance for Policy Gradient rewards
policy_gradient_mean = np.mean(policy_gradient_rewards, axis=0)
policy_gradient_variance = np.var(policy_gradient_rewards, axis=0)

# Plot Q-learning rewards
plt.plot(q_learning_mean, label='Q-learning', color='blue')
plt.fill_between(range(len(q_learning_mean)),
                 q_learning_mean - q_learning_variance,
                 q_learning_mean + q_learning_variance,
                 color='blue', alpha=0.3)

# Plot Policy Gradient rewards
plt.plot(policy_gradient_mean, label='Policy Gradient', color='red')
plt.fill_between(range(len(policy_gradient_mean)),
                 policy_gradient_mean - policy_gradient_variance,
                 policy_gradient_mean + policy_gradient_variance,
                 color='red', alpha=0.3)

# Add labels and legend
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episode (Q-learning vs Policy Gradient)')
plt.legend()



# Histogram to show which method generally has higher rewards

# # Flatten the rewards list for histogram
# q_learning_rewards_flat = np.concatenate(q_learning_rewards)
# policy_gradient_rewards_flat = np.concatenate(policy_gradient_rewards)

# # Plot histograms
# plt.hist(q_learning_rewards_flat, bins=20, alpha=0.5, label='Q-learning', color='blue')
# plt.hist(policy_gradient_rewards_flat, bins=20, alpha=0.5, label='Policy Gradient', color='red')

# # Add labels and legend
# plt.xlabel('Reward')
# plt.ylabel('Frequency')
# plt.title('Reward Distribution (Q-learning vs Policy Gradient)')
# plt.legend()


# Learning curve

# # Calculate cumulative rewards
# q_learning_cumulative = np.cumsum(q_learning_rewards, axis=1)
# policy_gradient_cumulative = np.cumsum(policy_gradient_rewards, axis=1)

# # Calculate mean cumulative rewards
# q_learning_mean = np.mean(q_learning_cumulative, axis=0)
# policy_gradient_mean = np.mean(policy_gradient_cumulative, axis=0)

# # Plot learning curve
# plt.plot(range(1, 21), q_learning_mean, label='Q-learning', color='blue')
# plt.plot(range(1, 21), policy_gradient_mean, label='Policy Gradient', color='red')

# # Add labels and legend
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Reward')
# plt.title('Learning Curve (Q-learning vs Policy Gradient)')
# plt.legend()

# Show plot
plt.grid(True)
plt.show()
