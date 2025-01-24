import xgboost as xgb
import matplotlib.pyplot as plt

booster = xgb.Booster()
booster.load_model("models/xgboost_model_test.json")

# 2. (Optional) If you need feature names and they are not in the model,
#    you can manually supply them here:
# booster.feature_names = ["feature1", "feature2", ...]

# # 3. Plot feature importances
xgb.plot_importance(booster, importance_type='gain')
plt.show()

# # 4. Plot a single tree (e.g., tree index 0)
# xgb.plot_tree(booster, num_trees=0)
# plt.show()

# 5. Dump the model to text to see each tree’s splits
booster.dump_model('model_dump.txt', with_stats=True)