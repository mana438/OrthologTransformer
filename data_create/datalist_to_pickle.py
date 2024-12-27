from read import OrthologDataset
import time
import glob
import pickle


OMA_species="/home/4/ux03574/workplace/data/OMA_database/prokaryotes_group.txt"
ortholog_files_train="/home/4/ux03574/workplace/data/OMA_database/BS_IS/train_fasta/*"
pickle_path="/home/4/ux03574/workplace/data/OMA_database/BS_IS/train_dataset.pkl"

reverse=False
calm=True

# OrthologDataset オブジェクトを作成する
dataset = OrthologDataset(OMA_species,"./vocab_OMA.json")
ortholog_files_train = glob.glob(ortholog_files_train)
train_dataset = dataset.load_data(ortholog_files_train, reverse, calm)
with open(pickle_path, "wb") as f:
    pickle.dump(train_dataset, f)