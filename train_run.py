import sys 
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='no-rep', help='no-rep=gpt2gen, no-zipfs, has-rep=regular, rm-window-rep')
    parser.add_argument('--task', type=str, default='wp', help='wp, wikitext')

    parser.add_argument('--rand_idx', type=str, default='no',
                        help='no or yes')

    parser.add_argument('--pretrained_model', type=str, default='gpt2', help='')
    parser.add_argument('--model_type', type=str, default='gpt2', help='')

    parser.add_argument('--dataset_name', type=str, default='wikitext', help='')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1', help='')
    parser.add_argument('--train_file', type=str, default='wikitext', help='')
    parser.add_argument('--validation_file', type=str, default='wikitext', help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--block_size', type=int, default=100, help='')

    # training parameters.
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--epoch', type=int, default=5, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--temperature', type=float, default=1., help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--percent', type=float, default=1.0, help='')

    parser.add_argument('--submit', type=str, default='no', help='')
    parser.add_argument('--use_big', type=str, default='no', help='')

    parser.add_argument('--app', type=str, default='', help='')


    args = parser.parse_args()

    folder_name = "classifier_models"


    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    if args.experiment == 'e2e-tgt' or  args.experiment == 'e2e-tgt-pos' or args.experiment == 'e2e-tgt-tree' or \
            args.experiment == 'e2e-tgt-gen-tree' or  args.experiment == 'e2e-tgt-gen-pos' or args.experiment == 'e2e-back-gen' \
            or args.experiment == 'e2e-tgt-gen-length' or args.experiment == 'e2e-tgt-gen-spans' \
            or args.experiment == 'e2e-back' \
            or args.experiment == 'simple-wiki' or args.experiment == 'roc':

        if args.dataset_name == 'none':
            Model_FILE = args.experiment + \
                         '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch, args.bsz * args.gradient_accumulation_steps,
                                                     args.pretrained_model, os.path.basename(args.train_file), args.seed,
                                                           args.task)
            Model_FILE = Model_FILE + f'_{args.notes}'
            logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
            Model_FILE = os.path.join(folder_name, Model_FILE)
            app = f" --train_file={args.train_file} --validation_file {args.validation_file} " \
                  f" --task {args.task}"
            app += " " + args.app


        else:
            Model_FILE = args.experiment + \
                         '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch, args.bsz * args.gradient_accumulation_steps,
                                                     args.pretrained_model, args.dataset_config_name, args.seed, args.task)
            Model_FILE = Model_FILE + f'_{args.notes}'
            logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
            Model_FILE = os.path.join(folder_name, Model_FILE)
            app = f" --dataset_name={args.dataset_name} " \
                  f"--dataset_config_name {args.dataset_config_name} --task {args.task}"
            app += " " + args.app




    COMMANDLINE = f"python transformers/examples/pytorch/language-modeling/run_clm.py \
            --output_dir={Model_FILE} \
            --model_name_or_path={args.pretrained_model} \
            --tokenizer_name={args.pretrained_model} \
            --per_device_train_batch_size {args.bsz} \
            --per_device_eval_batch_size {args.bsz} \
            --save_steps 50000 \
            --num_train_epochs {args.epoch} \
            --do_train --eval_steps 10000 --evaluation_strategy steps \
            --do_eval --dataloader_num_workers 4 \
            --save_total_limit 1 \
            --overwrite_output_dir  \
            --logging_dir {logging_dir} \
            --block_size {args.block_size}  \
            --disable_tqdm True --model_type {args.model_type} \
            --gradient_accumulation_steps {args.gradient_accumulation_steps} " \
                  f"--experiment {args.experiment} --seed {args.seed}"


    COMMANDLINE += app

    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    if args.submit == 'no':
        os.system(COMMANDLINE)  # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
