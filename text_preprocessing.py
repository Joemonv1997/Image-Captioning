import os
import pickle
import shutil
import nltk
mapping=dict()
with open("Flickr8k.token.txt") as token:
    file=token.read()
    token.close()
try:
    for doc in file.split("\n"):
        print(doc)
        img_id,*img_text=doc.split()
        print(img_id,"\n",img_text)
        img_id=img_id.split(".")[0]
        img_text=" ".join(img_text)
        if img_id not in mapping:
            mapping[img_id]=list()
        mapping[img_id].append(img_text)
except :
    pass
with open("text.pkl","wb") as text:
    pickle.dump(mapping,text)