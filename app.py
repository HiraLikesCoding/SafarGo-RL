import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SafarGo RL", page_icon="🚕", layout="wide")

st.title("🚕 SafarGo — RL Vehicle Repositioning")
st.write(
    "A simulated ride-hailing system using Q-Learning "
    "to reposition idle vehicles across 40 city zones."
)

st.warning(
    "This is a synthetic simulation for an academic MVP, "
    "not a live dispatch system. Results are illustrative."
)

# Configuration
ZONES = 40
ACTIONS = [-1, 0, 1]  # previous zone, stay, next zone
EPISODES = st.slider("Training episodes", 100, 1000, 300, 100)
VEHICLES = st.slider("Vehicles in simulated pilot", 10, 30, 20, 5)
STEPS = 48  # 48 simulated intervals per episode

ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.05

# Fixed synthetic demand pattern for repeatable experiments
rng = np.random.default_rng(42)
zone_popularity = rng.uniform(0.5, 2.0, ZONES)


def demand_at(zone, step, random_gen):
    """Generate synthetic demand based on zone and time."""
    rush = 1.5 if 12 <= step % 24 <= 17 else 1.0
    rate = zone_popularity[zone] * rush * 0.7
    return random_gen.poisson(rate)


def choose_action(q_table, state, epsilon, random_gen):
    if random_gen.random() < epsilon:
        return random_gen.integers(len(ACTIONS))
    return int(np.argmax(q_table[state]))


def run_episode(q_table, training, epsilon, random_gen):
    positions = random_gen.integers(0, ZONES, size=VEHICLES)
    total_reward = 0
    total_rides = 0
    total_wait = 0
    wait_samples = 0
    fuel = 0

    for step in range(STEPS):
        demand = np.array([
            demand_at(z, step, random_gen) for z in range(ZONES)
        ])

        for i in range(VEHICLES):
            zone = int(positions[i])
            demand_level = min(int(demand[zone]), 2)
            time_bucket = (step % 24) // 6

            # State = zone + time of day + local demand level
            state = (zone * 4 + time_bucket) * 3 + demand_level

            action_index = choose_action(
                q_table, state, epsilon if training else 0, random_gen
            )
            movement = ACTIONS[action_index]
            next_zone = (zone + movement) % ZONES

            # Simple synthetic movement cost
            if movement != 0:
                fuel += 1

            # Serve a request if demand exists at the destination
            available_demand = int(demand[next_zone])
            if available_demand > 0:
                rides = 1
                demand[next_zone] -= 1
                wait = random_gen.uniform(2, 10)
            else:
                rides = 0
                wait = 0

            # Reward balances rides, movement, and passenger waiting
            reward = (10 * rides) - (1.5 if movement != 0 else 0)
            if rides:
                reward -= max(0, wait - 7)

            next_demand_level = min(int(demand[next_zone]), 2)
            next_state = (
                (next_zone * 4 + time_bucket) * 3
                + next_demand_level
            )

            if training:
                best_future = np.max(q_table[next_state])
                old_value = q_table[state, action_index]
                q_table[state, action_index] = old_value + ALPHA * (
                    reward + GAMMA * best_future - old_value
                )

            positions[i] = next_zone
            total_reward += reward
            total_rides += rides

            if rides:
                total_wait += wait
                wait_samples += 1

    average_wait = total_wait / max(wait_samples, 1)
    return total_reward, average_wait, total_rides, fuel


def train_agent(episodes):
    # 40 zones x 4 time buckets x 3 demand levels
    q_table = np.zeros((ZONES * 4 * 3, len(ACTIONS)))
    history = []
    random_gen = np.random.default_rng(7)

    for ep in range(episodes):
        epsilon = max(
            EPSILON_MIN,
            EPSILON_START * (1 - ep / episodes)
        )
        result = run_episode(q_table, True, epsilon, random_gen)
        history.append(result[0])

    return q_table, history


def evaluate_q_agent(q_table, runs=20):
    random_gen = np.random.default_rng(100)
    results = [
        run_episode(q_table, False, 0, random_gen)
        for _ in range(runs)
    ]
    return np.mean(results, axis=0)


def evaluate_baseline(runs=20):
    random_gen = np.random.default_rng(200)
    results = []

    for _ in range(runs):
        positions = random_gen.integers(0, ZONES, size=VEHICLES)
        total_reward = 0
        total_rides = 0
        total_wait = 0
        wait_samples = 0
        fuel = 0

        for step in range(STEPS):
            demand = np.array([
                demand_at(z, step, random_gen) for z in range(ZONES)
            ])

            for i in range(VEHICLES):
                zone = int(positions[i])
                busiest = int(np.argmax(demand))

                # Move one zone toward the busiest zone
                clockwise = (busiest - zone) % ZONES
                if clockwise == 0:
                    movement = 0
                elif clockwise <= ZONES // 2:
                    movement = 1
                else:
                    movement = -1

                next_zone = (zone + movement) % ZONES
                if movement != 0:
                    fuel += 1

                if demand[next_zone] > 0:
                    demand[next_zone] -= 1
                    rides = 1
                    wait = random_gen.uniform(2, 10)
                    total_wait += wait
                    wait_samples += 1
                else:
                    rides = 0
                    wait = 0

                reward = 10 * rides - (1.5 if movement != 0 else 0)
                if rides:
                    reward -= max(0, wait - 7)

                total_reward += reward
                total_rides += rides
                positions[i] = next_zone

        results.append((
            total_reward,
            total_wait / max(wait_samples, 1),
            total_rides,
            fuel
        ))

    return np.mean(results, axis=0)


if st.button("Train and evaluate SafarGo", type="primary"):
    with st.spinner("Training the agent and running simulations..."):
        q_table, reward_history = train_agent(EPISODES)
        rl_results = evaluate_q_agent(q_table)
        baseline_results = evaluate_baseline()

    st.subheader("Performance comparison")
    comparison = pd.DataFrame({
        "Metric": [
            "Average reward per episode",
            "Average passenger wait (minutes)",
            "Completed rides",
            "Estimated fuel-use units"
        ],
        "Q-Learning": [
            round(rl_results[0], 2),
            round(rl_results[1], 2),
            round(rl_results[2], 2),
            round(rl_results[3], 2)
        ],
        "Busiest-zone baseline": [
            round(baseline_results[0], 2),
            round(baseline_results[1], 2),
            round(baseline_results[2], 2),
            round(baseline_results[3], 2)
        ]
    })
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Reward curve")
    fig, ax = plt.subplots()
    ax.plot(reward_history)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Q-Learning reward during training")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Fuel-use units represent repositioning movements, not measured litres. "
        "Waiting time is generated synthetically and is not a validated traffic model."
    )

    st.subheader("Interpretation")
    st.write(
        "Compare the two policies using the same metrics. A higher reward alone "
        "does not prove the RL agent is better. Check rides, waiting time, fuel "
        "and fairness before drawing conclusions."
    )