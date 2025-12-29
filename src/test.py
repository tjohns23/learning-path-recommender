from simulator.items import generate_items
from simulator.users import generate_users
from simulator.simulate import run_simulation
from features.interaction_features import extract_interaction_features

items = generate_items(5, 5)
users = generate_users(3, 5)
logs = run_simulation(users, items, max_steps=10)

df_interaction_features = extract_interaction_features(logs, users, items)
print(df_interaction_features.head())
