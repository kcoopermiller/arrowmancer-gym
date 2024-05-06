from src import ArrowmancerEnv, Agent
import torch
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    units = [
        {'name': 'Capricorn', 'level': '4A'}, 
        {'name': 'Aquarius', 'level': '5'}, 
        {'name': 'Pisces', 'level': 'Alt'}, 
    ] 
    env = ArrowmancerEnv(units)
    agent = Agent(env, units)

    agent.policy_net.load_state_dict(torch.load('model.pth'))
    agent.policy_net.to(device)

    # Set the agent to evaluation mode
    agent.policy_net.eval()

    # Run the agent in the environment
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(agent.get_state(state), dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(device)
        with torch.no_grad():
            action = agent.policy_net(state_tensor).max(1)[1].view(1, 1)
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        env.render()
        time.sleep(0.35)

if __name__ == "__main__":
    main()