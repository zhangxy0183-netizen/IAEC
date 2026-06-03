import argparse
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_all.tools.utils import model_test, load_config
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')
dataset = config['dataset']
if dataset != 'AVEC2014':
    base_root = config[dataset]['base_root']
else:
    base_root = None
test_path = config[dataset]['test_path']
mode = config['mode']
stage = config['stage']
audio_feature_type = config['audio_feature_type']
# 定义超参数
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--name', type=str, help="测试文件")
parser.add_argument('--batch_size', default=4, type=int, help='批处理大小')
parser.add_argument('--device', default=1, type=int, help='使用的GPU设备编号')
parser.add_argument('--video_feature_dim', default=128, type=int, help='视频特征维度')
parser.add_argument('--audio_feature_dim', default=128, type=int, help='音频特征维度')
parser.add_argument('--lstm_hidden_dim', default=128, type=int, help='lstm的隐藏层维度')
parser.add_argument('--output_dim', default=1, type=int, help='输出维度')
parser.add_argument('--temperature', default=1.0, type=float, help='情感对齐模块的温度参数')
parser.add_argument('--lambda_similarity', default=1.0, type=float, help='')
parser.add_argument('--dropout', default=0, type=float, help='dropout')
parser.add_argument('--log_dir', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/testlog',
                     type=str, help='日志保存路径')
if dataset != 'AVEC2014':
    parser.add_argument('--base_root', default=base_root, type=str, help='base_root')
parser.add_argument('--test_path', default=test_path, type=str, help='test_path')
parser.add_argument('--dataset', default=dataset, type=str, help='数据集版本 2014 2017 2019')
parser.add_argument('--mode', default=mode, type=str, help='mode')
parser.add_argument('--stage', default=stage, type=int, help='mode')
parser.add_argument('--eval_audio_noise', default=0.0, type=float,help='测试阶段音频高斯噪声标准差')
parser.add_argument('--eval_video_occlusion', default=0.0, type=float, 
                    help='测试阶段视频遮挡比例，例如 0.2 表示遮挡 20% 面积')
parser.add_argument('--eval_video_mode', default='none', type=str, choices=['none', 'center', 'random'],
                    help='测试阶段视频遮挡模式')
parser.add_argument('--audio_feature_type', default=audio_feature_type, type=str, help='audio_npy, wav2vec2, hubert')
args = parser.parse_args()

testmode = ['w_all', 'w_o_SE', 'w_o_ID', 'w_o_SIM', 'w_o_Video_Guide']
# testmode = ['w_all']
for data in testmode:
    args.test_mode = data
    model_test(args=args)
    


