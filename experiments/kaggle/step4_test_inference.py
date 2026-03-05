# ── Step 4: Test inference ──
tst_df = pd.read_csv(path/'test.csv')
X_tst_ens = tst_df[cat_cols_ens + cont_cols_ens]

# Each base learner's test prediction = average of its K fold models
tst_preds_per_model = np.zeros((len(X_tst_ens), n_models))
for m_idx, name in enumerate(model_names):
    fold_preds = np.zeros((len(X_tst_ens), N_FOLDS))
    for k, fitted_model in enumerate(fold_models[name]):
        fold_preds[:, k] = fitted_model.predict_proba(X_tst_ens)[:, 1]
    tst_preds_per_model[:, m_idx] = fold_preds.mean(axis=1)

# Weighted ensemble
tst_ensemble_preds = tst_preds_per_model @ optimal_weights

# Save submission
sub_df = pd.DataFrame({'id': tst_df['id'], 'Heart Disease': tst_ensemble_preds})
sub_df.to_csv('submission_auc_ensemble_ledell.csv', index=False)
print(f"Submission shape: {sub_df.shape}")
print(sub_df.head())
