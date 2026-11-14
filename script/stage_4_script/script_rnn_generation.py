import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
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

    MODEL_TYPE = 'rnn'   # 'rnn', 'lstm', 'gru'
    EXPERIMENT_NAME = f'generation_{MODEL_TYPE}'
    # ------------------------------------------------------

    # ---- object initialization section ---------------
    data_obj = Dataset_Loader_Generation('text_generation', '')
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_source_file_name = 'text_generation'

    method_obj = Method_RNN_Generation(f'{MODEL_TYPE} generation', '')
    method_obj.rnn_type = MODEL_TYPE
    method_obj.hidden_dim = 128
    method_obj.num_layers = 1
    method_obj.dropout = 0.2
    method_obj.loss_name = 'cross_entropy'
    method_obj.optimizer_name = 'adam'
    method_obj.learning_rate = 3e-3
    method_obj.batch_size = 32
    method_obj.max_epoch = 25
    method_obj.device_name = 'auto'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR + EXPERIMENT_NAME + '_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train/test split', '')
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
        title=f'Generation {MODEL_TYPE.upper()} (hidden={method_obj.hidden_dim}), {method_obj.loss_name}, {method_obj.optimizer_name}'
    )

    prompts = [
        'what did the',
        'why did the chicken',
        'two guys walked into',
        'how many'
    ]
    print('************ Sample Generated Jokes ************')
    sample_lines = []
    for p in prompts:
        gen = method_obj.generate_text(p, max_new_tokens=20, temperature=0.8)
        line = f'Prompt: "{p}" -> {gen}'
        print(line)
        sample_lines.append(line)

    with open(RESULT_DIR + EXPERIMENT_NAME + '_samples.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_lines) + '\n')

    print(f'************ Overall Performance: Text Generation ({MODEL_TYPE.upper()}) ************')
    for name, value in scores.items():
        print('%-20s %.4f' % (name, value))
    print('************ Finish ************')

    with open(RESULT_DIR + EXPERIMENT_NAME + '_metrics.txt', 'w', encoding='utf-8') as f:
        for name, value in scores.items():
            f.write('%-20s %.4f\n' % (name, value))