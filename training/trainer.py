from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd


class trainer():
	def __init__(self):
		pass
	def train(self, model, dataloader, epoch, epochs, writer, criterion, optimizer, device):
		progress_bar=tqdm(dataloader, colour='#800080', ncols=120)
		total_loss=0
		total_samples=0
		errors=0
		model = model.to(device)
		model.train()
		for iteration, batch in enumerate(progress_bar):
			try:
				X, y = batch['numericalized_encode'].to(device), batch['labels'].to(device)
				y_pred = model(X)
				loss=criterion(y_pred, y)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss+=loss.item()
				total_samples+=X.size(0)
				progress_bar.set_description(f"TRAIN | Epoch: {epoch+1}/{epochs} | Iter: {iteration+1}/{len(dataloader)} | Error: {errors}/{len(dataloader)} | Loss: {(total_loss/total_samples):.4f}")
			except Exception as e:
				errors+=1
				continue
		writer.add_scalar('Train/Loss', total_loss/total_samples, epoch+1)

	def validation(self, model, dataloader, criterion, device):
		model.eval()
		total_loss, total_samples, total_corrects=0,0,0
		with torch.no_grad():
			for batch in dataloader:
				outputs_logits=model(batch['numericalized_encode'].to(device))
				loss=criterion(outputs_logits, batch['labels'].to(device))
				_, predicted = torch.max(outputs_logits.data, 1)
				total_samples+=batch['labels'].to(device).size(0)
				total_corrects+=(predicted==batch['labels'].to(device)).sum().item()
				total_loss+=loss
		return total_loss/total_samples, total_corrects/total_samples
	def evaluate(self, model, test_loader, device):
		model.eval()
		predicted=[]
		true_labels=[]
		with torch.no_grad():
			for batch in test_loader:
				outputs_logits=model(batch['numericalized_encode'].to(device))
				_, predict=torch.max(outputs_logits, 1)
				predicted.extend(predict.cpu().numpy())
				true_labels.extend(batch['labels'].to(device).cpu().numpy())
		result = classification_report(predicted, true_labels, output_dict=True)
		result_df = pd.DataFrame(result).transpose().reset_index()
		return {
			'accuracy': accuracy_score(predicted, true_labels),
			'precision': result_df[result_df['index']=="weighted avg"]['precision'].values[0],
			'recall': result_df[result_df['index']=="weighted avg"]['recall'].values[0],
			'f1-score': result_df[result_df['index']=="weighted avg"]['f1-score'].values[0],
			'confusion_matrix': confusion_matrix(true_labels, predicted)
		}
if __name__=="__main__":
	pass