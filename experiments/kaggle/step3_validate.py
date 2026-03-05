# ── Step 3: Validate ensemble on held-out X_valid/y_valid ──
X_val_ens = X_valid[cat_cols_ens + cont_cols_ens]
y_val_bin = (y_valid == 'Presence').astype(int)

# Each base learner's val prediction = average of its K fold models
val_preds_per_model = np.zeros((len(X_val_ens), n_models))
for m_idx, name in enumerate(model_names):
    fold_preds = np.zeros((len(X_val_ens), N_FOLDS))
    for k, fitted_model in enumerate(fold_models[name]):
        fold_preds[:, k] = fitted_model.predict_proba(X_val_ens)[:, 1]
    val_preds_per_model[:, m_idx] = fold_preds.mean(axis=1)

# Weighted ensemble prediction
val_ensemble_preds = val_preds_per_model @ optimal_weights

print("Validation AUC per base learner:")
print("-" * 40)
for m_idx, name in enumerate(model_names):
    auc = roc_auc_score(y_val_bin, val_preds_per_model[:, m_idx])
    print(f"  {name:8s}: {auc:.4f}")

print(f"\nVal AUC (uniform avg):    {roc_auc_score(y_val_bin, val_preds_per_model.mean(axis=1)):.4f}")
print(f"Val AUC (AUC-optimized):  {roc_auc_score(y_val_bin, val_ensemble_preds):.4f}")
print(f"Baseline LR Val AUC:      0.9537")
