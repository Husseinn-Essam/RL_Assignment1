import numpy as np
import time
from gymnasium.wrappers import RecordVideo
from GridMazeEnv import GridMazeEnv

def get_env_dynamics(env):
    """
    Computes the transition probabilities and rewards for the given environment.
    P[s][a] = [(prob, next_state, reward, terminated), ...]
    This version computes next states analytically (no calls to step) so the
    70/15/15 stochasticity is applied exactly once per outcome.
    """
    # Unwrap env if it's wrapped (e.g., by RecordVideo)
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    size = base_env.size
    P = {s: {a: [] for a in range(base_env.action_space.n)} for s in range(base_env.observation_space.n)}

    # Precompute action->direction mapping from base_env
    action_to_direction = base_env._action_to_direction

    for s in range(base_env.observation_space.n):
        r = s // size
        c = s % size
        agent = np.array([r, c])
        for a in range(base_env.action_space.n):
            outcomes = {}
            # enumerate the three possible intended/perpendicular outcomes with their probs
            for prob, action_outcome in [(0.7, a), (0.15, (a + 1) % 4), (0.15, (a - 1 + 4) % 4)]:
                direction = action_to_direction[action_outcome]
                next_loc = np.clip(agent + direction, 0, size - 1)
                next_s = int(next_loc[0] * size + next_loc[1])

                # Determine termination and reward using the same logic as step()
                terminated = np.array_equal(next_loc, base_env._goal_location)
                is_bad_location = any(np.array_equal(next_loc, bl) for bl in base_env._bad_locations)
                if is_bad_location:
                    terminated = True

                if terminated and not is_bad_location:
                    reward = 10.0
                elif terminated and is_bad_location:
                    reward = -10.0
                else:
                    reward = -0.1

                if next_s in outcomes:
                    # This can happen if two different moves lead to the same state (e.g., hitting a wall)
                    # We add the probabilities together.
                    existing_prob, _, _, _ = outcomes[next_s]
                    outcomes[next_s] = (existing_prob + prob, next_s, reward, terminated)
                else:
                    outcomes[next_s] = (prob, next_s, reward, terminated)

            P[s][a] = list(outcomes.values())

    return P

def policy_evaluation(policy, V, P, gamma=0.99, theta=1e-6):
    """
    Evaluates a policy by iteratively updating the value function.
    Terminal transitions do not include future value.
    """
    while True:
        delta = 0
        for s in range(len(V)):
            v = V[s]
            new_v = 0.0
            a = policy[s]
            for prob, next_state, reward, terminated in P[s][a]:
                if terminated:
                    new_v += prob * reward
                else:
                    new_v += prob * (reward + gamma * V[next_state])
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(policy, V, P, gamma=0.99):
    """
    Improves the policy based on the current value function.
    Terminal transitions are treated correctly (no future value after termination).
    """
    policy_stable = True
    for s in range(len(policy)):
        old_action = policy[s]
        action_values = np.zeros(len(P[s]))
        for a in range(len(P[s])):
            for prob, next_state, reward, terminated in P[s][a]:
                if terminated:
                    action_values[a] += prob * reward
                else:
                    action_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = int(np.argmax(action_values))
        if old_action != policy[s]:
            policy_stable = False
    return policy, policy_stable

def policy_iteration(env, gamma=0.99):
    """
    The main Policy Iteration algorithm. 
    """
    print("1. Computing environment dynamics (P)...")
    P = get_env_dynamics(env)
    print("   ...Dynamics computed.")
    
    # Initialize a random policy and a zero value function
    policy = np.random.randint(0, env.action_space.n, size=env.observation_space.n)
    V = np.zeros(env.observation_space.n)
    
    policy_stable = False
    iterations = 0
    while not policy_stable:
        iterations += 1
        print(f"\nPolicy Iteration: Starting iteration {iterations}...")
        
        print(f"  - Running Policy Evaluation...")
        V = policy_evaluation(policy, V, P, gamma)
        print(f"  - Policy Evaluation complete.")
        
        print(f"  - Running Policy Improvement...")
        policy, policy_stable = policy_improvement(policy, V, P, gamma)
        print(f"  - Policy Improvement complete. Stable: {policy_stable}")
    
    print(f"\nPolicy converged after {iterations} iterations.")
    return policy, V, iterations

if __name__ == "__main__":
    # --- 1. Create the environment ---
    env = GridMazeEnv(render_mode="rgb_array")
    env = RecordVideo(env, "./videos", episode_trigger=lambda x: x > 0)
    # For human visualization, use this line
    # env = GridMazeEnv(render_mode="human")

    # --- 2. Run Policy Iteration ---
    # For Policy Iteration, the environment must be fixed.
    # We reset it once to generate the maze layout (G and X locations).
    env.reset()

    # Pass the base environment to policy_iteration
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    optimal_policy, V, num_iterations = policy_iteration(base_env)

    print("\nOptimal policy found. Displaying agent behavior.")
    print("Number of iterations to converge:", num_iterations)

    # --- 3. Test the trained agent ---
    for episode in range(3):
        obs, info = env.reset()
        # Recompute policy for each new layout
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        optimal_policy, V, num_iterations = policy_iteration(base_env)

        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated and step_count < 100:
            env.render()
            action = optimal_policy[obs]
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward
            step_count += 1
            time.sleep(0.2) # Slow down for visualization
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
