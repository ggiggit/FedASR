# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"
#export mkgraph_cmd="queue.pl -l arch=*64,ram_free=4G,mem_free=4G"
#export cuda_cmd=run.pl


#if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
#  export train_cmd="queue.pl -l arch=*64*"
#  export decode_cmd="queue.pl -l arch=*64* --mem 3G"
#  export mkgraph_cmd="queue.pl -l arch=*64* --mem 4G"
#  export cuda_cmd="queue.pl -l gpu=1"
#elif [[ $(hostname -f) == *.fit.vutbr.cz ]]; then
#  #b) BUT cluster options
#  queue="all.q@@blade,all.q@@speech,all.q@dellgpu*,all.q@supergpu*"
#  export train_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,matylda5=0.5"
#  export decode_cmd="queue.pl -q $queue -l ram_free=3000M,mem_free=3000M,matylda5=0.1"
#  export mkgraph_cmd="queue.pl -q $queue -l ram_free=4G,mem_free=4G,matylda5=3"
#  export cuda_cmd="queue.pl -q long.q@pcspeech-gpu,long.q@dellgpu1,long.q@pcgpu*,long.q@supergpu1 -l gpu=1" 
#else
#  echo "$0: you need to define options for your cluster."
#  exit 1;
#fi

#c) run locally...
export train_cmd=utils/run.pl
export decode_cmd=utils/run.pl
export cuda_cmd=utils/run.pl
export mkgraph_cmd=utils/run.pl
