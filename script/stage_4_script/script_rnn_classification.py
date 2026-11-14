import os, sys

# local_code があるフォルダを自動で探してパスに追加
cur_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = cur_dir
while PROJECT_ROOT != os.path.dirname(PROJECT_ROOT):
    if os.path.isdir(os.path.join(PROJECT_ROOT, 'local_code')):
        break
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Plot_Learning_Curve import plot_learning_curve

import numpy as np
import torch

if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_4_data') + os.sep
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result') + os.sep
    os.makedirs(RESULT_DIR, exist_ok=True)

    MODEL_TYPE = 'rnn'   # 'rnn', 'lstm', 'gru' に変えて3回実行する！
    EXPERIMENT_NAME = f'classification_{MODEL_TYPE}'
    # ------------------------------------------------------

    # ---- object initialization section ---------------
    data_obj = Dataset_Loader('text_classification', '')
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_source_file_name = 'text_classification'

    method_obj = Method_RNN_Classification(f'{MODEL_TYPE} classification', '')
    method_obj.rnn_type = MODEL_TYPE
    method_obj.hidden_dim = 128
    method_obj.num_layers = 1
    method_obj.bidirectional = False
    method_obj.dropout = 0.3
    method_obj.loss_name = 'cross_entropy'
    method_obj.optimizer_name = 'adam'
    method_obj.learning_rate = 1e-3
    method_obj.batch_size = 64
    method_obj.max_epoch = 10
    method_obj.device_name = 'auto'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR + EXPERIMENT_NAME + '_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')
    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, F1 score', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()

    plot_learning_curve(
        method_obj.history,
        RESULT_DIR + EXPERIMENT_NAME + '_learning_curve.png',
        title=f'{MODEL_TYPE.upper()} (hidden={method_obj.hidden_dim}), {method_obj.loss_name}, {method_obj.optimizer_name}'
    )

    print(f'************ Overall Performance: Text Classification ({MODEL_TYPE.upper()}) ************')
    for name, value in scores.items():
        print('%-20s %.4f' % (name, value))
    print('************ Finish ************')

    with open(RESULT_DIR + EXPERIMENT_NAME + '_metrics.txt', 'w') as f:
        for name, value in scores.items():
            f.write('%-20s %.4f\n' % (name, value))