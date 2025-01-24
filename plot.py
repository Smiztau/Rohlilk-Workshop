import xgboost as xgb
import matplotlib.pyplot as plt
import os
import pandas as pd

# Directory containing the models
model_dir = "models"
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

# List all model files
model_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]

# Dictionary to aggregate feature importance scores
combined_importance = {}

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)

    # Load the model
    booster = xgb.Booster()
    booster.load_model(model_path)

    # Get feature importance scores
    importance_scores = booster.get_score(importance_type='gain')

    # Aggregate the scores across models
    for feature, score in importance_scores.items():
        if feature not in combined_importance:
            combined_importance[feature] = 0
        combined_importance[feature] += score

# Convert aggregated importance to a DataFrame for easy plotting
importance_df = pd.DataFrame(
    list(combined_importance.items()), columns=['Feature', 'Importance']
).sort_values(by='Importance', ascending=False)

# Plot the combined feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Total Gain Across Models')
plt.ylabel('Features')
plt.title('Combined Feature Importance Across All Models')
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.tight_layout()

# # Save the plot
# output_plot_path = os.path.join(output_dir, 'photos/combined_feature_importance.png')
# plt.savefig(output_plot_path)
plt.show()

# print(f"Combined feature importance plot saved to: {output_plot_path}")
