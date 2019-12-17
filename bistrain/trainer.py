# In case of being imported on notebook
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def train_a2c(mp_envs, agent, episodes=2000, n_step=5, print_every=10, max_steps=300):
    """
    Train the given agent on the parallel environments provided.

    Parameters
    ----------
    mp_envs: SubprocVecEnv
        Parallels environments
    agent: Agent
        Agent exploring the environments
    episodes: int
        Number of episodes to train
    print_every: int
        Frequency to display training metrics
    max_steps: int
        Maximum number of steps
    """
    # Saving metrics
    avg_scores_deque = deque(maxlen=print_every)
    avg_scores = []
    scores_envs = []
    # Keep track of progress
    pbar = tqdm(range(1, episodes + 1), ncols=800)
    for i_episode in pbar:
        # Reset env
        initial_states = mp_envs.reset()
        # Reset agent noise (exploration)
        agent.reset()
        score = []
        gamma = GAMMA
        for i in range(max_steps):
            # Collect trajectories
            S, A, R, Sp, dones = n_step_boostrap(mp_envs, agent,
                                                 initial_states,
                                                 n_step)
            agent.learn(S, A, R, Sp, dones, gamma)
            # Start from the next state
            initial_states = Sp[:, 0, :]
            # Collect scores from all parallel envs
            score.append(R[:, 0])
            # Update initial gamma
            gamma *= GAMMA
            if dones[:, -1].any():
                break
        # Save scores
        episode_score = np.asarray(score).sum(axis=0)
        mean_score = episode_score.mean()
        avg_scores_deque.append(mean_score)
        avg_scores.append(mean_score)
        scores_envs.append(episode_score)
        # Display some progress
        if (i_episode) % print_every == 0:
            text = '\rEpisode {}/{}\tAverage Scores: {:.2f}'.format(i_episode,
                                                                    episodes,
                                                                    np.mean(avg_scores_deque))
            pbar.set_description(text)

        # Solve environment
        # if mean(avg_scores_deque) >=

    return np.asarray(scores_envs)

