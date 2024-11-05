pendulum = Pendulum(is_root=True)
data_buffer = DataBuffer(
    pendulum,
    paths_to_remember=["pendulum_state", "reinforce/reinforce_control"],
    buffer_size=200,
)
is_done = IsDone(
    pendulum,
    paths_to_observe=["pendulum_state"],
)
reinforce = Reinforce(
    pendulum,
    data_buffer,
    is_done,
    update_freq=1,
    step_size=0.1,
)

graph = Graph(
    [pendulum, reinforce, data_buffer, is_done],
    states_to_log=["pendulum_state", "reinforce/reinforce_control", "is_done"],
)

# Training loop
n_episodes = 1000
episode_count = 0

while episode_count < n_episodes:
    graph.step()
    if is_done.state.data[0]:
        episode_count += 1
        pendulum.reset()
        is_done.reset()
        data_buffer.reset()

# Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(reinforce.returns)
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("REINFORCE Learning Curve")
plt.grid(True)
plt.show()
