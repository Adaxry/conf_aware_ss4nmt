# conf_aware_ss4nmt
Codes of "Confidence-Aware Scheduled Sampling for Neural Machine Translation". The implementation is based on [THUMT](https://github.com/thumt/THUMT).

## requirements
tensorflow-gpu 1.12+
python 3.5+

## datasets 
We obtain the [WMT14 En-De](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh) and [WMT14 En-Fr](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh) datasets following the [script] from [fairseq](https://github.com/pytorch/fairseq). We preprocess the WMT19 Zh-En dataset released on the official WMT [website](http://www.statmt.org/wmt20/translation-task.html) and make the preprocessed datasets available at [here](https://drive.google.com/file/d/1LvUPsIZ_xRwuB1vHlvi1COeZEOxfbYy0/view?usp=sharing).

## pre-train
We firstly pre-train the Transformer for 100k steps. An example script for Transformer$_{base}$ is as bellow:
```
code_dir=./nmt/THUMT
data_dir=path2yourdata
work_dir=./nmt
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export PYTHONPATH=$work_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=pretrained_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=100000,batch_size=4096,max_length=128,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True

```

## finetune
Finetuning the Transformer with scheduled sampling:
```
code_dir=./nmt/THUMT-confidence
work_dir=./nmt
data_dir=path2yourdata

export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=finetune_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --checkpoint $work_dir/train/pretrained_model_name
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=300000,batch_size=4096,max_length=128,layer_preprocess=none,layer_postprocess=layer_norm,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=10000,keep_checkpoint_max=200,bridge_input_scale=sqrt_depth,use_bridge_dropout=True,select_random_trg_prob=0.95,select_golden_trg_prob=0.9,happen_prob=1.0,replace_prob=1.0,mle_rate=1,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=False,bridge_rate=1.0,zero_step=True

```


