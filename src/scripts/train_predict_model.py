import argparse
from src.common.load_data import load_data
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from src.common.score import scorePredict
from src.model.model_stance import StanceModel


def main(parser):
    args = parser.parse_args()
    training_set = args.training_set
    use_cuda = args.use_cuda
    is_cross_validation = args.is_cross_validation
    is_evaluate = args.is_evaluate
    model_dir = args.model_dir
    model_type = args.model_type
    model_name = args.model_name
    wandb_project = args.wandb_project
    is_sweeping = args.is_sweeping
    best_result_config = args.best_result_config
    exec_model(model_type, model_name, model_dir, training_set, is_cross_validation, is_evaluate, use_cuda,
               wandb_project, is_sweeping, best_result_config)
    

def exec_model(model_type, model_name, model_dir, training_set, is_cross_validation, is_evaluate, use_cuda,
               wandb_project, is_sweeping, best_result_config):

    df = load_data(os.getcwd() + training_set, True)
    df = df[0:150]
    labels = list(df['labels'].unique())

    results = None
    f1s = None
    if model_dir == '':
        model = StanceModel(model_type, model_name, use_cuda, len(labels), wandb_project,
                            is_sweeping, is_evaluate, best_result_config, True)
        if is_cross_validation:
            n = 5
            kf = KFold(n_splits=n, random_state=3, shuffle=True)
            results = []
            f1s = []
            for train_index, val_index in kf.split(df):
                train_df = df.iloc[train_index]
                val_df = df.iloc[val_index]
                acc, f1, model_outputs_test = model.train_predict_model(train_df, val_df)
                results.append(acc)
                f1s.append(f1)
            print(np.mean(results))
            print(np.mean(f1s))
        else:
            df_train, df_test = train_test_split(df, test_size=0.20, random_state=1)
            model.train_predict_model(df_train)

    else:
        model = StanceModel(model_type, os.getcwd() + model_dir, use_cuda)
    if not results:
        y_predict = model.predict_task(df_test)
        labels_test = pd.Series(df_test['labels']).to_numpy()
        labels = list(df_test['labels'].unique())
        labels.sort()
        result, f1 = scorePredict(y_predict, labels_test, labels)
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_set",
                        default="/data/Dataset_VIL.json",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--is_cross_validation",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cross-validation is a requirement.")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--is_evaluate",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if you want to split train in train and dev.")

    parser.add_argument("--model_dir",
                        default='',
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--model_type",
                        default="roberta",
                        type=str,
                        help="This parameter is the relative type of model to trian and predict.")

    parser.add_argument("--model_name",
                        default="PlanTL-GOB-ES/roberta-base-bne",
                        type=str,
                        help="This parameter is the relative name of model to trian and predict.")

    parser.add_argument("--wandb_project",
                        default="",
                        type=str,
                        help="This parameter is the name of wandb project.")

    parser.add_argument("--is_sweeping",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you use sweep search.")

    parser.add_argument("--best_result_config",
                        default="",
                        type=str,
                        help="This parameter is the file with best hyperparameters configuration.")

    main(parser)