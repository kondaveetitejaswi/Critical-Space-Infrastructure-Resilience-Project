from satellite_env import SatelliteDefenseEnv
import numpy as np

def test_satellite_defense_env():
    env = SatelliteDefenseEnv()
    episode_count = 0
    max_episodes = 1

    print("\n --------Starting Satellite Defense Environment Test ----------\n")

    while episode_count < max_episodes:
        env.reset(seed=42)
        done = False

        for agent in env.agent_iter():
            obs = env.observe(agent)
            if obs is None:
                continue

            action = np.array([np.random.uniform(0, 1)], dtype=np.float32)
            env.step(action)

            env.render()

            if any(env.dones.values()):
                report = env.get_env_report()
                print("\n✅ Cumulative Rewards (from env):")
                for agent, total_reward in report["cumulative_rewards"].items():
                    print(f"{agent}: {total_reward:.3f}")
                episode_count += 1
                break

    env.close()
    print("\n --------Ending Satellite Defense Environment Test ----------\n")
    print(f"Completed {episode_count} episodes")

if __name__ == "__main__":
    test_satellite_defense_env()
