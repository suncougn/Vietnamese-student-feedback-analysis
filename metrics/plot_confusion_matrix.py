import matplotlib.pyplot as plt
import seaborn as sns

def plot_cfs_matrix(confusion_matrix, class_name) -> None:
	plt.figure(figsize=(3.2,3))
	sns.heatmap(confusion_matrix, fmt='g', annot=True, cmap='Blues', cbar=True)
	plt.title('Confusion Matrix', fontsize=16)
	plt.xlabel('Predicted Labels', fontsize=14)
	plt.ylabel('True Labels', fontsize=14)
	plt.xticks(ticks=[0.5, 1.5, 2.5], labels=class_name)
	plt.yticks(ticks=[0.5, 1.5, 2.5], labels=class_name)
	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	pass