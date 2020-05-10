# COSC572 Final Report Grammar Correction

**The glove.6B 100d file is required.**

Because this file exceeds the limit of single file size(100MB). It cannot be downloaded with the whole project through webpage or by using commands git clone/git pull.

Please use [Large File Storage](https://git-lfs.github.com/)

or 

Download this [file](https://github.com/gnaWeinrE/NLP_FinalProject/blob/master/glove.6B.100d.txt) from the web page alone.

## Commands

To preprocess the original data, use command:

**python data.py**

To train the model, use command:

**python train.py**

To predict a sentence, use command:

**python predict.py p He went in school in the morning**

To correct a sentence, use command:

**python predict.py c He went in school in the morning**
