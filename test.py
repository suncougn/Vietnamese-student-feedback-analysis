from build_model.build_gru import *
import pickle

with open('model_architecture/GRU.pkl','rb') as f:
	model=pickle.load(f)

state_dict=torch.load('D:\\IT\\Major\\NLP\\Student feedback\\model\\GRU\\best.pt')
model.load_state_dict(state_dict['model_state_dict'])

X=torch.ones((16, 24), dtype=torch.long)
print(model(X))