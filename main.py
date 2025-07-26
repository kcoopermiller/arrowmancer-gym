from src import ArrowmancerEnv, Agent
import torch
import time
import argparse

def main():
    # OHAYO GOZAIMASU! Welcome to the ultimate anime battle simulator desu!
    parser = argparse.ArgumentParser(description='Arrowmancer Gym - Now with 200% more weeb garbage! SUGOI!')
    parser.add_argument('--train', action='store_true', help='set training mode (train your waifus to be stronger desu!)')
    parser.add_argument('--model', type=str, default='dqn', help='AI model to use for maximum kawaii intelligence; available options: [dqn]')
    parser.add_argument('--device', type=str, default='cuda', help='processing device for ultimate anime power; available options: [cpu, cuda]')
    parser.add_argument('--witches', nargs=3, type=str, default=['Sakura-chan', 'Zero-Two', 'Rem-chan'], metavar=('W1', 'W2', 'W3'), help='names of the kawaii witches (standard banner only) - now with 100% more anime desu!')
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
        if reward > 0:
            print("🎉 SUGOI! VICTORY DESU! Your waifu team is the strongest! Anime power wins again! 🎉")
        else:
            print("😭 NANI?! Defeat desu... But don't give up! The power of friendship will help you next time! GANBATTE! 😭")

if __name__ == "__main__":
    main()