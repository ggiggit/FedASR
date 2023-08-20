# FedASR
This project is based on Pytorch-Kaldi and utilizes the ease of use of Pytorch to easily integrate federated learning with Pytorch.  
Just use it like the way using Pytorch-Kaldi.
## TIMIT tutorial
In the following, we provide a short tutorial of the PyTorch-Kaldi toolkit based on the popular TIMIT dataset.

1. Make sure you have the TIMIT dataset. If not, it can be downloaded from the LDC website (https://catalog.ldc.upenn.edu/LDC93S1).

2. Make sure Kaldi and PyTorch installations are fine. Make also sure that your KALDI paths are currently working (you should add the Kaldi paths into the .bashrc as reported in the section "Prerequisites"). For instance, type "copy-feats" and "hmm-info" and make sure no errors appear. 

3. Run the Kaldi s5 baseline of TIMIT. This step is necessary to compute features and labels later used to train the PyTorch neural network. We recommend running the full timit s5 recipe (including the DNN training): 

```
cd kaldi/egs/timit/s5
./run.sh
./local/nnet/run_dnn.sh
```

This way all the necessary files are created and the user can directly compare the results obtained by Kaldi with that achieved with our toolkit.

4. Compute the alignments (i.e, the phone-state labels) for test and dev data with the following commands (go into $KALDI_ROOT/egs/timit/s5). If you want to use tri3 alignments, type:
```
steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
```

If you want to use dnn alignments (as suggested), type:
```
steps/nnet/align.sh --nj 4 data-fmllr-tri3/train data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali

steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
```

5. We start this tutorial with a very simple MLP network trained on mfcc features.  Before launching the experiment, take a look at the configuration file  *cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg*. See the [Description of the configuration files](#description-of-the-configuration-files) for a detailed description of all its fields. 

6. Change the config file according to your paths. In particular:
- Set “fea_lst” with the path of your mfcc training list (that should be in $KALDI_ROOT/egs/timit/s5/data/train/feats.scp)
- Add your path (e.g., $KALDI_ROOT/egs/timit/s5/data/train/utt2spk) into “--utt2spk=ark:”
- Add your CMVN transformation e.g.,$KALDI_ROOT/egs/timit/s5/mfcc/cmvn_train.ark
- Add the folder where labels are stored (e.g.,$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali for training and ,$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev for dev data).

To avoid errors make sure that all the paths in the cfg file exist. **Please, avoid using paths containing bash variables since paths are read literally and are not automatically expanded** (e.g., use /home/mirco/kaldi-trunk/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali instead of $KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali)

7. Run the ASR experiment in the way of federated learning:
```
python run_fed_exp.py cfg/TIMIT_baselines/TIMIT_fed.cfg
```

This script starts a full ASR experiment and performs training, validation, forward, and decoding steps.  A progress bar shows the evolution of all the aforementioned phases. The script *run_exp.py* progressively creates the following files in the output directory:

- *res.res*: a file that summarizes training and validation performance across various validation epochs.
- *log.log*: a file that contains possible errors and warnings.
- *conf.cfg*: a copy of the configuration file.
- *model.svg* is a picture that shows the considered model and how the various neural networks are connected. This is really useful to debug models that are more complex than this one (e.g, models based on multiple neural networks).
- The folder *exp_files* contains several files that summarize the evolution of training and validation over the various epochs. For instance, files *.info report chunk-specific information such as the chunk_loss and error and the training time. The *.cfg files are the chunk-specific configuration files (see general architecture for more details), while files *.lst report the list of features used to train each specific chunk.
- At the end of training, a directory called *generated outputs* containing plots of loss and errors during the various training epochs is created.



