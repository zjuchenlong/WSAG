YouwikiHow dataset can be downloaded from https://github.com/zjuchenlong/YouwikiHow.


### Dataset layout
```
YouwikiHow/
|   +-- annotations/
|       +-- wikihow_data.pkl
|       +-- crosstask_test.pkl
|   +-- features/
|       +-- train.csv
|       +-- test.csv
|       +-- train_s3d_features_from_ht100m
|       +-- test_s3d_features_fr16_sz256_nf16
```

### Training Set
The training set `wikihow_data.pkl` is a dictionary, **key** is the **wikiHow ID**, and `value` are:

- **task_full_name**: wikiHow task full name.
- **task_text**: wikihow articles. 
    - **simplified_headline_list**: List of all high-level summary sentences.
    - **simplified_article_list**:  list of all low-level articles.
    - **head2sent**: The sentence IDs mapping from *high-level* (key) sentences to *low-level* (value) sentences.
    - **sent2head**: The sentence IDs mapping from *low-level* (key) sentences to *high-level* (value) sentences.
- **task_video**: The list of all YouTube videos correspoinding to this wikiHow task.
- **video_duration**: The total duration of all YouTube videos.

### Test Set
The test set `crosstask_test.pkl` is same format as the training set, with two more keys:
- **step2headline**: The mannually mapping between the CrossTask steps (in original datasets) and the wikiHow articles (i.e., headlines).
- **crosstask_annotations**: The ground-truth for evaluation. *Keys* are the video ids, and *Values* are all possible ground-truth annotations propagated from CrossTask.

### Download Visual Features
[Google drive](https://drive.google.com/drive/folders/1Ril5qUilNGw8gBvCCjhnJo9SqetrKT_K?usp=sharing)
- **train_s3d_features_from_ht100m**
- **test_s3d_features_fr16_sz256_nf16** 
