
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='train')
parser.add_argument('--alias', type=str, required=False, default='scannet')
parser.add_argument('--gpu', type=str, required=False, default='0')
parser.add_argument('--port', type=str, required=False, default='10001')
parser.add_argument('--ckpt', type=str, required=False, default='\'\'')
args = parser.parse_args()


def execute_command(cmds):
  cmd = ' '.join(cmds)
  print('Execute: \n' + cmd + '\n')
  os.system(cmd)


def train():
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.alias  {}'.format(args.alias),
      'SOLVER.dist_url tcp://localhost:{}'.format(args.port),]
  execute_command(cmds)


  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_pred logs/scannet/lkpm_test_{}'.format(args.alias),
      '--path_out logs/scannet/lkpm_test_seg_{}'.format(args.alias),
      '--filelist  data/scannet.npz/scannetv2_test.txt', ]
  execute_command(cmds)


def validate():
  # get the predicted probabilities for each point
  ckpt = ('logs/scannet/lkpm_{}/best_model.pth'.format(args.alias)
          if args.ckpt == '\'\'' else args.ckpt)   # use args.ckpt if provided
  print('alias:', args.alias)
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'LOSS.mask -255',       # to keep all points
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.run evaluate',
      'SOLVER.eval_epoch 120',  # voting with 120 predictions
      'SOLVER.alias val_{}'.format(args.alias),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.test.batch_size 1',
      'DATA.test.distort True',]
  execute_command(cmds)

  # map the probabilities to labels
  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_pred logs/scannet/lkpm_val_{}'.format(args.alias),
      '--path_out  logs/scannet/lkpm_val_seg_{}'.format(args.alias),
      '--filelist  data/scannet.npz/scannetv2_val_npz.txt', ]
  execute_command(cmds)
  # calculate the mIoU
  cmds = [
      'python tools/seg_scannet.py',
      '--run calc_iou',
      '--path_in data/scannet.npz/train',
      '--path_pred logs/scannet/pointmamba_val_seg_{}'.format(args.alias), ]
  execute_command(cmds)

def calc_iou():
  cmds = [
      'python tools/seg_scannet.py',
      '--run calc_iou',
      '--path_in data/scannet.npz/train',
      '--path_pred logs/scannet/pointmamba_val_seg_{}'.format(args.alias), ]
  execute_command(cmds)

if __name__ == '__main__':
  eval('%s()' % args.run)
