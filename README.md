### Code Setup and Requirements
You can install all the required packages using the following command:
```
    $ pip install -r requirements.txt
```

### Train Random Forest

To train Random Forest Classifier and get Random Forest proximity, use the following command. This will save the proximity matrix for the graph in `proximities/` directory and the Random Forest model in `models/rf/` directory.
```
   $ python train_rf.py
```

### Train GCN

To train GCN on a dataset, use the following command. This will save the results in `results/` directory and the best GCN model in `models/gnn/` directory.
```
   $ python train_gcn.py --dataset <dataset_id> --gpu <gpu_index>
```

`--dataset_id` is the ID of OpenML dataset.
`--gpu_index` is the index of GPU which is set to 0 by default.


