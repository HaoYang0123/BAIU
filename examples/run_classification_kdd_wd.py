# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys
sys.path.insert(0, '..')
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
#from deepctr_torch.models.deepfm import DeepFM
from deepctr_torch.models.wdl import WDL

if __name__ == "__main__":
    input_path = sys.argv[1]
    sparse_features = ['C' + str(i) for i in range(1, 13)]

    int2str_dict = {}
    for name in sparse_features:
        int2str_dict[name] = str

    data = pd.read_csv(input_path, converters=int2str_dict)


    data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        #print("ori", feat, data[feat])
        data[feat] = lbe.fit_transform(data[feat])
        #print("fff", feat, data[feat])
    #mms = MinMaxScaler(feature_range=(0, 1))
    #data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 10, embedding_dim=4)
                              for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    print("dnn feature", dnn_feature_columns)
    print("linear features", linear_feature_columns)

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    print("feature names", feature_names)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    #print("train_model_input", train_model_input)
    #print("test_model_input", test_model_input)

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    #model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #               task='binary',
    #               l2_reg_embedding=1e-5, device=device)
    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
                    task='binary', 
                    l2_reg_embedding=1e-5, device=device)

    print("Model**********", model)
    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    print("start fit======")
    history = model.fit(train_model_input, train[target].values, batch_size=40960, epochs=10, verbose=2,
                        validation_split=0.1, test_x=test_model_input, test_y=test[target].values)
    print("start predict")
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
