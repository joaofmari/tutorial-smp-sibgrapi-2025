import os
import time, datetime
import argparse
import shutil

# =====================================================================
# Run several experiments in batch mode.
# =====================================================================
# Enables to run several experiments in batch mode.
# Each experiment runs the script train-test.py with a specific set of parameters.
# The user can define lists of hyperparameters, architectures, and encoders.
# ---------------------------------------------------------------------

# Argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='deepglobe', help='Dataset name.')
args = parser.parse_args()

# Create main experiment folder.
EXP_PATH_MAIN = f'exp_{args.ds}' 

# Dataset-specific parameters.
h_size_dict = {'38-cloud' : 384,
               'deepglobe' : 512,
               'FUSAR-Map' : 512}
w_size_dict = {'38-cloud' : 384,
               'deepglobe' : 512,
               'FUSAR-Map' : 512}
in_channels_dict = {'38-cloud' : 4,
                    'deepglobe' : 3,
                    'FUSAR-Map' : 1}
n_classes_dict = {'38-cloud' : 2,
                  'deepglobe' : 7,
                  'FUSAR-Map' : 5}

# Hyperparameters are defined as lists to run several experiments.
bs_list = [8] # [8, 16, 24]
lr_list = [0.0001] # [0.001, 0.0001, 0.00001]
loss_list = ['crossentropy', 'dice'] # ["jaccard", "dice", "tversky", "focal", "lavosz", "crossentropy"]
da_train_list = ['none', 'moderate'] # ['none', 'mild', 'moderate', 'strong']
scheduler_list = ['plateau'] # ['cosine', 'plateau', 'step']

# Architecture list
model_list_ = [
    'Unet', 
    'FPN',
    # Insert other models here.
]

# Encoder list
backbone_list_ = [
    'resnet50',            
    'efficientnet-b2',     
    # Insert other encoders here! 
]

max_epochs = 400 # 200, 400 # 1000
save_images_str = '--save_images' # --no-save_images
segmap_mode = 'darker' # ['simple', 'gray', 'darker']

# Inicia contagem de tempo deste época
time_start = time.time()

# Experiment counter.
ec = 0 

# Create the command string and run it.
for model in model_list_:
    for backbone in backbone_list_:
        for bs in bs_list:
            for lr in lr_list:
                for scheduler in scheduler_list:
                    for da_train in da_train_list:
                        for loss in loss_list:

                            ### for smp_reduction in smp_reduction_list:
                            cmd_str = f'nohup python train-test.py --dataset_name {args.ds} --n_classes {n_classes_dict[args.ds]} ' + \
                                      f'--in_channels {in_channels_dict[args.ds]} --h_size {h_size_dict[args.ds]} --w_size {w_size_dict[args.ds]} ' + \
                                      f'--model {model} --backbone {backbone} --loss {loss} --da_train {da_train} --max_epochs {max_epochs} ' + \
                                      f'--batch_size {bs} --lr {lr} --scheduler {scheduler} {save_images_str} --segmap_mode {segmap_mode} ' + \
                                      f' --ec {ec}'

                            ec = ec + 1

                            print(cmd_str)
                            os.system(cmd_str)

# Count total time of the experiment.
time_exp = time.time() - time_start
time_exp_hms = str(datetime.timedelta(seconds = time_exp))
print(f'Time exp.: {time_exp} sec ({time_exp_hms})')

print('\nFinish! (run-batch)')