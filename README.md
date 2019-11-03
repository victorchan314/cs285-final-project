    # Code to run vanilla Actor-Critic
    python run_hw3_actor_critic.py --env_name CartPole-v0 --exp_name CartPole

    # Run policy distillation with MLP student
    python run_hw3_actor_critic.py --env_name CartPole-v0 --exp_name CartPole --distill MLP
