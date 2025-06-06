from satellite_env import SatelliteEnv
import numpy as np

env  = SatelliteEnv()
env.reset()

for agent in env.agent_iter():
    obs = env.observe(agent)
    act_space = env.action_space(agent)
    action = act_space.sample()  # Sample a random action from the action space
    reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()  # Reset the environment if done
        print(f"Agent {agent} has finished its episode.")
    else:
        print(f"Agent {agent} took action {action}, received reward {reward}, and is {'done' if done else 'not done'}.")

env.close()  # Close the environment when done
print("Environment closed.")