import argparse
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_k.tools.utils import run_test

# 定义超参数
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--name', type=str, help="测试文件")
parser.add_argument('--batch_size', default=4, type=int, help='批处理大小')
parser.add_argument('--device', default=2, type=int, help='使用的GPU设备编号')
parser.add_argument('--video_feature_dim', default=128, type=int, help='视频特征维度')
parser.add_argument('--audio_feature_dim', default=128, type=int, help='音频特征维度')
parser.add_argument('--lstm_hidden_dim', default=128, type=int, help='lstm的隐藏层维度')
parser.add_argument('--output_dim', default=1, type=int, help='输出维度')
parser.add_argument('--temperature', default=1.0, type=float, help='情感对齐模块的温度参数')
parser.add_argument('--lambda_similarity', default=1.0, type=float, help='情感对齐损失权重')
parser.add_argument('--frame_num', default=15, type=int, help='帧数')
parser.add_argument('--dropout', default=0, type=float, help='dropout')
parser.add_argument('--log_dir', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/testlog',
                     type=str, help='日志保存路径')
parser.add_argument('--mode', default="cam", type=str, help='cam/fea')
parser.add_argument('--test_mode', default="wo_se_model", type=str, help='w_all_model wo_id_model wo_sim_model wo_se_model')
args = parser.parse_args()

testmode = ['w_all_model', 'wo_id_model', 'wo_sim_model', 'wo_se_model']
for data in testmode:
    args.test_mode = data
    args.mode = 'fea'
    run_test(args=args)
    args.mode = 'cam'
    run_test(args=args)


    