import glob
import pandas as pd
import tqdm

class Process_data():
    def __init__(self, lbs = None, is_train = True, is_save = True):
        self.id2lbs = lbs 
        self.lbs2id = {val:index for index, val in enumerate(lbs)}    
        
        if is_train == True:
            self.root = 'Train_Full'
        else: 
            self.root = 'Test_Full'
        
        self.all_files_name = glob.glob(self.root + '/*/*.txt')
        self.df = self.process_all_file()
        
        if is_save == True:
            self.save()

    
    def process_all_file(self):
        all_labels = []
        all_files_name = []
        all_text = []
        all_labels2id = []
        for lb in self.id2lbs:
            label2id = self.lbs2id[lb]
            label_files_name = glob.glob(self.root+ '/' + lb + '/*.txt')
            
            labels = [lb]*len(label_files_name)
            labels2id = [label2id]*len(label_files_name)
            
            all_labels += labels 
            all_files_name += label_files_name
            all_labels2id += labels2id
            
            text_of_label = []
            for file in tqdm(label_files_name):
                with open(file, 'r', encoding='utf-16', errors='ignore') as f:
                    lines = f.readlines()
                    lines = [line.replace('\n', '') for line in lines]
                    lines = " ".join(lines)
                    sent = self.process_sent(lines)
                    text_of_label.append(sent)
            all_text += text_of_label

        assert len(all_labels) == len(all_files_name) 
        assert len(all_labels) == len(all_text) 
        assert len(all_labels) == len(all_labels2id)
        df = pd.DataFrame()
        df['labels'] = all_labels
        df['labels2id'] = all_labels2id
        df['text'] = all_text
        df['file_name'] = all_files_name
        df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def process_sent(self, sentence):
        sentence=sentence.strip(' ')
        sentence=sentence.strip('.')
        sentence=sentence.strip(';')
        sentence=sentence.strip(',')
        sentence=sentence.replace('%','')
        sentence=sentence.replace('=',' ')
        sentence=sentence.replace(',',', ')
        sentence=sentence.replace(';',' ')
        sentence=sentence.replace(', ',' ')
        sentence=sentence.replace('.',' ')
        sentence=sentence.replace('/',' ')
        sentence=sentence.replace('/ ',' ')
        sentence=sentence.replace('-',' ')
        sentence=sentence.replace('(',' ')
        sentence=sentence.replace(')',' ')
        sentence=sentence.replace(':',' ')
        sentence=sentence.replace('   ',' ')
        sentence=sentence.replace('  ',' ')
        sentence=sentence.lower()
        sentence=sentence.split()
        
        return sentence
    def save(self):
        self.df.to_csv(self.root + '/.csv', index = False)
