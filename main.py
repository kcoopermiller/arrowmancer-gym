from src import ArrowmancerEnv, Agent
import torch
import time
import argparse

# UwU Notice me senpai! This is the main kawaii entry point desu~
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set training mode')
    parser.add_argument('--model', type=str, default='dqn', help='model to use; available options: [dqn]')
    parser.add_argument('--device', type=str, default='cuda', help='device to use; available options: [cpu, cuda]')
    parser.add_argument('--witches', nargs=3, type=str, default=['Celine', 'Kepler', 'Zorn'], metavar=('W1', 'W2', 'W3'), help='names of the kawaii witches (standard banner only) - choose your waifu team desu!')
    # TODO: add hyperparameter arguments (ganbatte ne!)
    
    args = parser.parse_args()
    units = [
        {'name': args.witches[0], 'banner': 'standard'}, 
        {'name': args.witches[1], 'banner': 'standard'}, 
        {'name': args.witches[2], 'banner': 'standard'}, 
    ]
    # Creating magical girl environment desu ka? Kyaa~!
    env = ArrowmancerEnv(units)
    # Summoning our AI-chan to fight alongside the magical girls! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
    agent = Agent(env, algorithm=args.model, device=args.device)

    if args.train:
        print("🌸 Training mode activated! Our AI-chan will become stronger through the power of friendship! 🌸")
        print("Ganbatte kudasai! Training for 1000 episodes desu~")
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
            print("🎉 VICTORY DESU! Our magical girls have defeated the evil forces! Yatta ne! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
        else:
            print("💔 Defeat... But don't worry! Even magical girls sometimes lose! We'll get stronger next time desu! (´；ω；`)")

if __name__ == "__main__":
    main()