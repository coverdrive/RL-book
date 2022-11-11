# American Options Pricing

The American Options Pricing are different strategies we use to maximize the profit when exercising the options where the price varies with time. The goal of this project is to verify that RL based techniques perform well in such prediction use cases. We'll be conducting our horserace b/w Dynamic Programming, Backward Induction, Longstaff Shwarz, LSPI & DQNs here.

## File Structure

- `algo_wrapper.py`: This is the interface class that defines the training, prediction, etc. that every model would need to implement.
- `price_simulator.py`: This is used to generate test data for the options pricing at different points in time.
    - **Usage**: `python3 price_simulator.py --folder_path="~/RL-book/rl/simulated/" <flags>`. The following are the flags available.
    - **Flags**:
        - `--folder_path`: Folder to Store the Generated Runs. **Required**, no default.
        - `--expiry`: Time to expiry. Default: 1.0.
        - `--rate`: Risk Free Rate. Default: 0.05.
        - `--vol`: Volatility. Default: 0.25.
        - `--spot_price`: Spot Price at T=0. Default: 100.0.
        - `--num_paths`: Number of Simulation Paths to Run. Default: 100.
        - `--spot_price_frac`: Variation of Spot Price around args.spot at T=0. Default: 0.3.
        - `--seed`: Random Seed. Default: 0.
        - `--overwrite`: Overwrite Folder if already exists?. Default: False.
