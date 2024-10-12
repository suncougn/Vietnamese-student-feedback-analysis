from sklearn.model_selection import RandomizedSearchCV

def perform_random_search_pytorch(model_name, model, param_dist, X_train, y_train, n_iter=3, cv=None):
	randomizedSearch=RandomizedSearchCV(
		estimator=model,
		param_distributions=param_dist,
		n_iter=n_iter,
		cv=cv
		)
	randomizedSearch.fit(X_train, y_train)
	print(f"\n{model_name} model results after HyperParameters Tuning:\n")
	cv_results=randomizedSearch.cv_results_
	for params, mean_test_score, rank in zip(cv_results['params'], cv_results['mean_test_score'], cv_results['rank_test_score']):
		print(f"Params: {params} - Mean test score: {mean_test_score:.2f} - Rank: {rank}")
	best_parameters = randomizedSearch.best_params_
	print(f"\nBests parameters for {model_name} model: {best_parameters}\n")

	best_model=randomizedSearch.best_estimator_
	return best_parameters, best_model

if __name__=='__main__':
	pass