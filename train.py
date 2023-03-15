import json
import numpy as np
import random
import math

###Read Files###
#author id -> author names
a_authors = json.load(open("aozora_authors.json",encoding="utf-8")) 

#content id -> contents
a_contents = json.load(open("aozora_contents.json",encoding="utf-8"))

#title id -> titles
a_titles = json.load(open("aozora_titles.json",encoding="utf-8"))

#title id -> content id
test_aozora_data = json.load(open("test_aozora_data.json",encoding="utf-8"))
train_aozora_data = json.load(open("train_aozora_data.json",encoding="utf-8"))

#read train/test data
train_q = json.load(open("train_q.json",encoding="utf-8"))
test_q = json.load(open("test_q.json",encoding="utf-8"))

print(len(train_q),len(test_q))

#Cleaning the Contents#
for key in a_contents:
    
    a_contents[key] = a_contents[key].strip("\n")
    a_contents[key] = a_contents[key].strip("\r")
    a_contents[key] = a_contents[key].strip("\u3000")
    #print(a_contents[key])

answer_dict = {}
title_contents_dict = {}

for items in train_q:
    answer_dict[items["title_id"]]=items["answer"]

for items in train_aozora_data:
    for novels in items["novels"]:
        title_contents_dict[novels["title_id"]] = novels["content_id"]


title_contents_dict.update(test_aozora_data)



### First Glance ###
'''input -> title_id + candidates[author id] + answer

   output -> title_id + sorted_candidates[author id]

   basic acc : 0.266

   num of author : 870
   num of titles : 14488
   num of contents : 14488

   train : 12991
   test : 548



   ***process***

   input -> titles + candidates[author id] 
   
         -> contents + answer_author (candidates) 
         -> model 
         -> contents + sorted_candidates[author id] 
         -> title_id + sorted_candidates[author id]
                                                       -> output

    steps1 tranfer words/sentences into vectors.
    steps2                                                          
'''



#attempt1 if test data already exists in train data, simply gives the output.
#result: No data are the same. acc=0.266
'''
answer_dict = {}

for items in train_q:
    answer_dict[items["title_id"]]=items["answer"]


for items in test_q:
    if items['title_id'] in answer_dict:
        print("True")
        answer = answer_dict[items['title_id']]
        ans_index = items["candidates"].index(answer)
        if ans_index != 0 :
            temp = items["candidates"][0]
            items["candidates"][0] = items["candidates"][ans_index]
            items["candidates"][ans_index] = temp

with open("suggestion.json", "w") as outfile:
    jsonString = json.dumps(test_q)
    outfile.write(jsonString)
'''

#attempt2 use BERT to tokenize the contents
#         calculate the similarity between vectors to compare if cotents from novels are similar
#         when input, compare the content between the other novels from that author
#         choose the most similar as the predicted author
#         this method doesn't require Model Training/Finetuning. 
#result:  0.574 without weights added
#         0.791 with weights
#         0.810 with exp weights



#from transformers import BertJapaneseTokenizer, BertModel
import torch

####turn all contents into vectors
#### use bert to translate the contents into a 1*768 vector.
#### save the content_id - vector into vec_contents.json
#### time-costing so this part is commented.
###################this only uses once######################
'''
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #device = "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
model = SentenceBertJapanese(MODEL_NAME)

vec_content = {}
count = 0 
for keys in a_contents:
    count += 1
    if count%100 == 0:
        print(count)
    sentence = a_contents[keys]
    #print(len(sentence))
    sentence_embeddings = model.encode([sentence], batch_size=8)
    #print("Sentence embeddings:", sentence_embeddings.shape)
    vec_content[keys] = sentence_embeddings.detach().numpy()


from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
        
with open("vec_contents.json", "w") as outfile:
    jsonString = json.dumps(vec_content, cls=NumpyArrayEncoder)
    outfile.write(jsonString)
'''
###########################################################

#### use cosine_similarity to evaluate the similarity between the contents of the title given and contents written by each author.
#### sort the author list based on the evaluation.


vec_content = json.load(open("vec_contents.json",encoding="utf-8")) 




from sklearn.metrics.pairwise import cosine_similarity

for items in test_q:
    novel = vec_content[title_contents_dict[items["title_id"]]]
    #print(novel)
    evaluate = []
    #print(items["candidates"])
    for ids in items["candidates"]:
        content_list = []
        for keys in answer_dict:
            if answer_dict[keys] == ids:
                content_list.append(vec_content[title_contents_dict[keys]][0])
                #print("novel_sim",vec_content[title_contents_dict[keys]].shape)
        content_list.append(novel[0])
        #print(content_list)
        if len(content_list)>1:
            sim2 = cosine_similarity([content_list[-1]],content_list[0:-2])[0].ravel()
            sim = sim2
            print(len(sim))
            l = len(sim)
            sim.sort()
            sum_sim = 0
            divider = 0
            ### add weights for each novel, the higher the similarity is, the bigger the weight is.
            for k in range(0,l):
                sum_sim += float(sim[k]) * math.exp((l-k)/l)
                divider += math.exp((l-k)/l)
            sum_sim /= divider

        else:
            sum_sim = 0
        evaluate.append(sum_sim)
    for i in range(0,len(evaluate)):
        for j in range(0,len(evaluate)-i-1):
            if evaluate[j] < evaluate[j+1]:
                (evaluate[j],evaluate[j+1]) = (evaluate[j+1],evaluate[j])
                (items["candidates"][j],items["candidates"][j+1]) = (items["candidates"][j+1],items["candidates"][j])
    
    print(items["candidates"])


with open("suggestion.json", "w") as outfile:
    jsonString = json.dumps(test_q)
    outfile.write(jsonString)





#attempt3 use BERT to tokenize the contents
#         use pairs to train a network to predict the similarity (instead the cosine_similarity)
#         the pair contains two contents from two different novels
#         if the authors are the same, label is 1
#         if the authors are different, label is 0
#result:  0.523 with linear model
#





#### generate Training Set####
###################this only uses once######################
'''
trainset = []
vec_content = json.load(open("vec_contents.json",encoding="utf-8")) 

count = 0
while(count < 100000):
    title_pair = random.sample(a_titles.keys(),2)
    ## samples which label = 1 ##
    if title_pair[0] in answer_dict and title_pair[1] in answer_dict and answer_dict[title_pair[0]] == answer_dict[title_pair[1]]:
        author_id_pair = [answer_dict[title_pair[0]],answer_dict[title_pair[1]]]
        content_id_pair = [title_contents_dict[title_pair[0]], title_contents_dict[title_pair[1]]]

        content_pair = [vec_content[content_id_pair[0]][0],vec_content[content_id_pair[1]][0]]
        label = 1
        train_data = {"label":[1,0],"text":content_pair}
        trainset.append(train_data)
        count += 1
        if count%1000 == 0:
            print(count)

count = 0
while(count < 100000):
    title_pair = random.sample(a_titles.keys(),2)
    ## samples which label = 0 ##
    if title_pair[0] in answer_dict and title_pair[1] in answer_dict and answer_dict[title_pair[0]] == answer_dict[title_pair[1]]:
        author_id_pair = [answer_dict[title_pair[0]],answer_dict[title_pair[1]]]
        content_id_pair = [title_contents_dict[title_pair[0]], title_contents_dict[title_pair[1]]]

        content_pair = [vec_content[content_id_pair[0]][0],vec_content[content_id_pair[1]][0]]
        label = 0
        train_data = {"label":[0,1],"text":content_pair}
        trainset.append(train_data)
        count += 1
        if count%1000 == 0:
            print(count)

with open("testset.json", "w") as outfile:
    jsonString = json.dumps(trainset)
    outfile.write(jsonString)
'''
############################################################


#### input trainset ####


'''
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class PairwiseNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(768,96)

        self.linear2 = torch.nn.Linear(96,2)
        self.relu= torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x0,x1):

        x00 = x0.mul(x1)
        x01 = self.relu(self.linear1(x00))
        x02 = self.relu(self.linear2(x01))
        out = self.sigmoid(x02)

        return out 
'''
###################this only uses once######################
'''
model = PairwiseNetwork().to(device)



dataset = json.load(open("trainset.json",encoding="utf-8")) 
#testset = json.load(open("testset.json",encoding="utf-8")) 

Data_Size = len(dataset)
#Test_Size = len(testset)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
loss = torch.nn.CrossEntropyLoss()
loss.cuda()

print(model)

num_epochs = 100
batch = 512

x0 = []
x1 = []
labels = []
for items in dataset:
    x0.append(items["text"][0])
    x1.append(items["text"][1])
    labels.append(items["label"])

tensor_x0 = torch.Tensor(x0).to("cuda:0") # transform to torch tensor
tensor_x1 = torch.Tensor(x1).to("cuda:0") # transform to torch tensor
tensor_y = torch.Tensor(labels).to("cuda:0")


my_dataset = TensorDataset(tensor_x0,tensor_x1,tensor_y) # create your datset
dataloader = DataLoader(my_dataset,batch_size = batch, shuffle=True) # create your dataloader



for epoch in range(num_epochs):  # loop over the dataset multiple times

    random.shuffle(dataset)
    total_train_loss = 0
    total_test_loss = 0
    model.train()


    for i,  (x0,x1, labels) in enumerate(dataloader):
        
        optimizer.zero_grad()

        # forward + backward + optimize
   
        outputs = model(x0,x1)
        #print(x0.size(),x1.size())
        #print(outputs.size(),labels.size())
        l = abs((outputs[0]-labels[0]).sum())
    
        #print(float(l))
        l.backward()
        optimizer.step()

        total_train_loss += l


    total_train_loss/= (Data_Size/batch)
    t1 = round(float(total_train_loss),3)
              
    

    #total_test_loss/= Test_Size
    #t2 = round(float(total_test_loss),3)
    #print("trainloss:",t1," ","testloss:",t2)                 
    print("trainloss:",t1)

    if epoch % 2 == 0:    
        #torch.save(model, str(epoch)+"_train"+str(t1)+"_test"+str(t2)+"_"+"save.pt")
        torch.save(model, str(epoch)+"_train"+str(t1)+"_"+"save.pt")
    total_train_loss = 0 

print('Finished Training')
'''
############################################################

####Test to make predictions with model####

'''
vec_content = json.load(open("vec_contents.json",encoding="utf-8")) 
model = PairwiseNetwork().to(device)
model = torch.load("16_train0.0_save.pt")
model.eval()
for items in test_q:
    novel = vec_content[title_contents_dict[items["title_id"]]]
    #print(novel)
    evaluate = []
    #print(items["candidates"])
    for ids in items["candidates"]:
        content_list = []
        for keys in answer_dict:
            if answer_dict[keys] == ids:
                content_list.append(vec_content[title_contents_dict[keys]][0])
                #print("novel_sim",vec_content[title_contents_dict[keys]].shape)
        content_list.append(novel[0])
        #print(content_list)
        if len(content_list)>1:

            #sim2 = cosine_similarity([content_list[-1]],content_list[0:-2])[0].ravel()
            sim2 = []
            for con in content_list[0:-2]:
                pred = model(torch.cuda.FloatTensor(content_list[-1]),torch.cuda.FloatTensor(con),)
                sim2.append(pred[0])
            sim = sim2
            #print(len(sim))
            l = len(sim)
            sim.sort()
            sum_sim = 0
            divider = 0
            ### add weights for each novel, the higher the similarity is, the bigger the weight is.
            for k in range(0,l):
                sum_sim += float(sim[k]) * (l-k)
                divider += l - k
            sum_sim /= divider

        else:
            sum_sim = 0
        evaluate.append(sum_sim)
    for i in range(0,len(evaluate)):
        for j in range(0,len(evaluate)-i-1):
            if evaluate[j] < evaluate[j+1]:
                (evaluate[j],evaluate[j+1]) = (evaluate[j+1],evaluate[j])
                (items["candidates"][j],items["candidates"][j+1]) = (items["candidates"][j+1],items["candidates"][j])
    
    print(items["candidates"])


with open("suggestion.json", "w") as outfile:
    jsonString = json.dumps(test_q)
    outfile.write(jsonString)



'''









