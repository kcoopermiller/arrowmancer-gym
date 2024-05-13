from src import ArrowmancerEnv, Agent
import torch
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set training mode')
    parser.add_argument('--model', type=str, default='dqn', help='model to use; available options: [dqn]')
    parser.add_argument('--device', type=str, default='cuda', help='device to use; available options: [cpu, cuda]')
    parser.add_argument('--witches', nargs=3, type=str, default=['Celine', 'Kepler', 'Zorn'], metavar=('W1', 'W2', 'W3'), help='names of the witches (standard banner only)')
    # TODO: add hyperparameter arguments
    
    args = parser.parse_args()
    units = [
        {'name': args.witches[0], 'banner': 'standard'}, 
        {'name': args.witches[1], 'banner': 'standard'}, 
        {'name': args.witches[2], 'banner': 'standard'}, 
    ]
    env = ArrowmancerEnv(units)
    agent = Agent(env, algorithm=args.model, device=args.device)

    if args.train:
        agent.train(num_episodes=1000)
    else: 
        agent.policy_model.load_state_dict(torch.load(f'Models/{args.model}.pth'))
        agent.policy_model.to(args.device)

        # Set the agent to evaluation mode
        agent.policy_model.eval()

        # Run the agent in the environment
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(agent.get_state(state), dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.to(args.device)
            with torch.no_grad():
                action = agent.policy_model(state_tensor).max(1)[1].view(1, 1)
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            env.render()
            time.sleep(0.3)

if __name__ == "__main__":
    main()