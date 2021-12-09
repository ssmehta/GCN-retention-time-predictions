import argparse
import numpy as np
import pandas as pd
import glob
import sys

sys.path.append('../src/')

from utils import metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset')
    args = parser.parse_args()

    models = ['gcn', 'rgcn', 'mlp', 'rf', 'svm', 'gb', 'ab']
    df = pd.DataFrame(index=[m.upper() for m in models])

    for model in models:
        files = glob.glob(f'../output/predictions/{args.dataset}/{model}/*')

        if files:
            assert files[0].split('/')[-1].split('.')[0] == 'train'
            assert files[1].split('/')[-1].split('.')[0] == 'valid'
            assert files[2].split('/')[-1].split('.')[0] == 'test' or files[2].split('/')[-1].split('.')[0] == 'test_1'

            if len(files) == 4:
                assert files[3].split('/')[-1].split('.')[0] == 'test_2'

            for file in files:
                data = np.load(file)
                mae = metrics.get('mae')(data[0], data[1])
                mre = metrics.get('mre')(data[0], data[1])
                rmse = metrics.get('rmse')(data[0], data[1])
                r2  = metrics.get('r2')(data[0], data[1])

                subset = file.split('/')[-1].split('.')[0]
                df.loc[model.upper(), subset +'_mae'] = mae
                df.loc[model.upper(), subset +'_mre'] = mre
                df.loc[model.upper(), subset +'_rmse'] = rmse
                df.loc[model.upper(), subset +'_r2'] = r2

    df.to_csv(f'results_{args.dataset}.csv')
