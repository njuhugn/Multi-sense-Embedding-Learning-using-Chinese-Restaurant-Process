# Do Multi-Sense Embeddings Improve Natural Language Understanding

Implementations of multi-sense learning algorithm using Chinese Restaurant Process in "Do Multi-Sense Embeddings Improve Natural Language Understanding" by Jiwei Li and Dan Jurafsky, EMNLP 2015
Other algorithms involved in the paper will be released soon.


## Input Files
train_file.txt: input files. The algorithm treats documents as basic units. Each line corresponds to a sentence within a document. Documents are signified by an empty line with "\n". Each index in train_file.txt corresponds to a specific word token.  If your input files do not have explicit document boundaries, you can treat a chunk of any random number of sentences as a document.

frequency.txt: word occuring probability for each token found in train_file.txt. The first line in frequency.txt corresponds to the occuring probability for word indexed by 0, the second line  to the occuring probability of word 1 and so forth


## Output Files
save_vect: each line corresponds to the learned embedding for an indexed word, e.g., the first line corresponds to embedding for word indexed by 0, second line to word 1, and so forth.
save_vect_sense: 
First line starts with: "word 0 sense0 0.9866423962091394", meaning that sense 0 for 0th word has 0.986 occuring probability, followed by the corresponding embedding for current sense.

## Preprocessing
in directory Preprocessing, text.txt is a small sample of txt (a massively larger dataset is needed to train meaningful representations). Run:
python WordIndexNumDic.py vocabsize output_dictionary_file output_frequency_file output_index_file input_text_file, for example:
python WordIndexNumDic.py 20000 ../dictionary.txt ../frequency.txt ../train_file.txt text.txt


For any question, feel free to contact jiweil@stanford.edu



```latex
@article{li2015hierarchical,
    title={Do Multi-Sense Embeddings Improve Natural Language Understanding?},
    author={Li, Jiwei and Jurafsky, Dan},
    journal={EMNLP 2015},
    year={2015}
}
```
