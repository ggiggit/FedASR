##########################################################
# FedASR v.0.1
# WanZixiang, Beijing University of Posts and Telecommunications
# August 2023
##########################################################


from __future__ import print_function

import os
import sys
import glob
import configparser
import numpy as np
from fedutils import (
    check_cfg,
    fed_create_lists,
    create_configs,
    compute_avg_performance,
    read_args_command_line,
    run_shell,
    compute_n_chunks,
    fed_compute_n_chunks,
    get_all_archs,
    cfg_item2sec,
    fed_dump_round_results,
    fed_create_curves,
    change_lr_cfg,
    expand_str_rd,
    fed_get_val_cfg_file_path,
    fed_get_val_info_file_path,
    average_weights,
    optimizer_init,
    save_globa_model,
    global_dict_fea_lab_arch
)
from data_io import read_lab_fea_refac01 as read_lab_fea
from shutil import copyfile
from core import read_next_chunk_into_shared_list_with_subprocess, extract_data_from_shared_list, convert_numpy_to_torch
import re
from distutils.util import strtobool
import importlib
import math
import multiprocessing
import threading
import torch
import copy

def _run_forwarding_in_subprocesses(config):
    use_cuda = strtobool(config["exp"]["use_cuda"])
    if use_cuda:
        return False
    else:
        return True


def _is_first_validation(rd):
    if rd>0:
        return False
    return True


def _max_nr_of_parallel_forwarding_processes(config):
    if "max_nr_of_parallel_forwarding_processes" in config["forward"]:
        return int(config["forward"]["max_nr_of_parallel_forwarding_processes"])
    return -1

def create_nns(config_file):
    def _read_lab_fea(config, shared_list):
        [fea_dict, lab_dict, arch_dict] = global_dict_fea_lab_arch(config)
        shared_list.append(fea_dict)
        shared_list.append(lab_dict)
        shared_list.append(arch_dict)
    
    nns={}
    # Reading config parameters for local model
    config = configparser.ConfigParser()
    config.read(config_file)
    model = config["model"]["model"].split("\n")
    output_folder = config["exp"]["out_folder"]
    is_production = strtobool(config["exp"]["production"])
    shared_list = []
    p = threading.Thread(target=_read_lab_fea, args=(config, shared_list))
    p.start()
    p.join()
    inp_out_dict = shared_list[0]
    arch_dict = shared_list[2]
    pattern = "(.*)=(.*)\((.*),(.*)\)"
    for line in model:
        [out_name, operation, inp1, inp2] = list(re.findall(pattern, line)[0])
        if operation == "compute":
            # computing input dim
            inp_dim = inp_out_dict[inp2][-1]
            module = importlib.import_module(config[arch_dict[inp1][0]]["arch_library"])
            nn_class = getattr(module, config[arch_dict[inp1][0]]["arch_class"])
            # Here may report some errors of lacking some key
            # It's OK. Just give the config some key-value. 
            # Because the nns is used to store weights of models, not for running.
            cls = ["GRU","LSTM","logMelFb","channel_averaging","liGRU","minimalGRU","RNN"]
            if config[arch_dict[inp1][0]]["arch_class"] in cls:
                config[arch_dict[inp1][0]]["use_cuda"]="True"
                config[arch_dict[inp1][0]]["to_do"]="Train"
            net = nn_class(config[arch_dict[inp1][0]], inp_dim)
            nns[arch_dict[inp1][1]] = net
            out_dim = net.out_dim

            # updating output dim
            inp_out_dict[out_name] = [out_dim]

        if operation == "concatenate":

            inp_dim1 = inp_out_dict[inp1][-1]
            inp_dim2 = inp_out_dict[inp2][-1]

            inp_out_dict[out_name] = [inp_dim1 + inp_dim2]

        if operation == "cost_nll":
            inp_out_dict[out_name] = [1]

        if operation == "cost_err":
            inp_out_dict[out_name] = [1]

        if (
            operation == "mult"
            or operation == "sum"
            or operation == "mult_constant"
            or operation == "sum_constant"
            or operation == "avg"
            or operation == "mse"
        ):
            inp_out_dict[out_name] = inp_out_dict[inp1]
    return [nns, arch_dict]


# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)


# Reading and parsing optional arguments from command line (e.g.,--optimization,lr=0.002) 好像没用，没试过
[section_args, field_args, value_args] = read_args_command_line(sys.argv, config)


# Output folder creation
out_folder = config["exp"]["out_folder"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder + "/exp_files")

# Log file path
log_file = config["exp"]["out_folder"] + "/log.log"


# Read, parse, and check the config file
cfg_file_proto = config["cfg_proto"]["cfg_proto"]
[config, name_data, name_arch] = check_cfg(cfg_file, config, cfg_file_proto)


# Read cfg file options

is_production = strtobool(config["exp"]["production"])
cfg_file_proto_chunk = config["cfg_proto"]["cfg_proto_chunk"]

cmd = config["exp"]["cmd"]
N_rd = int(config["federated"]["round"])
N_rd_str_format = "0" + str(max(math.ceil(np.log10(N_rd)), 1)) + "d"
N_ep = int(config["federated"]["epoch"])
N_ep_str_format = "0" + str(max(math.ceil(np.log10(N_ep)), 1)) + "d"
N_clt = int(config["federated"]["client_num"])
N_clt_str_format = "0" + str(max(math.ceil(np.log10(N_clt)), 1)) + "d"
frac = min(float(config["federated"]["client_fraction"]),1)
tr_data_lst = config["data_use"]["train_with"].split(",")
valid_data_lst = config["data_use"]["valid_with"].split(",")
forward_data_lst = config["data_use"]["forward_with"].split(",")
max_seq_length_train = config["batches"]["max_seq_length_train"]
forward_save_files = list(map(strtobool, config["forward"]["save_out_file"].split(",")))


print("- Reading config file......OK!")


# Copy the global cfg file into the output folder
cfg_file = out_folder + "/conf.cfg"
with open(cfg_file, "w") as configfile:
    config.write(configfile)


# Load the run_nn function from core libriary
# The run_nn is a function that process a single chunk of data
run_nn_script = config["exp"]["run_nn_script"].split(".py")[0]
module = importlib.import_module("core")
run_nn = getattr(module, run_nn_script)
nns = {}
optimizers = {}
use_cuda = strtobool(config["exp"]["use_cuda"])
multi_gpu = strtobool(config["exp"]["multi_gpu"])


# Splitting data into chunks (see out_folder/additional_files)
# 划分训练，验证和测试集，并存储起来
# create_lists(config)
fed_create_lists(config)

# Writing the config files
create_configs(config)

print("- Chunk creation......OK!\n")
# create res_file
global_res_file_path = out_folder + "/global"+"_res.res"
global_res_file = open(global_res_file_path,"w")
global_res_file.close()

# Learning rates and architecture-specific optimization parameters
arch_lst = get_all_archs(config)
lr = {}
auto_lr_annealing = {}
improvement_threshold = {}
halving_factor = {}
pt_files = {}


for arch in arch_lst:
    lr[arch] = expand_str_rd(config[arch]["arch_lr"], "float",N_rd, "|", "*")
    if len(config[arch]["arch_lr"].split("|")) > 1:
        auto_lr_annealing[arch] = False
    else:
        auto_lr_annealing[arch] = True
    improvement_threshold[arch] = float(config[arch]["arch_improvement_threshold"])
    halving_factor[arch] = float(config[arch]["arch_halving_factor"])
    for clt in range(N_clt):
        pt_files.setdefault(clt, {})
        pt_files[clt][arch] = config[arch]["arch_pretrain_file"]


# If production, skip training and forward directly from last saved models
if is_production:
    rd = N_rd - 1
    ep = N_ep - 1
    N_ep = 0
    N_rd = 0
    model_files = {}
    global_model_files = {}
    for clt in range(N_clt):
        for arch in pt_files[clt].keys():
            model_files.setdefault(clt, {})
            model_files[clt][arch] = out_folder + "/exp_files/final_" + "clt" + format(clt,N_clt_str_format)+"_"+arch + ".pkl"
    for arch in pt_files[clt].keys():
        global_model_files[arch] = out_folder + "/exp_files/final_" + "global_"+arch + ".pkl"


op_counter ={clt: 1 for clt in range(N_clt)}   # used to dected the next configuration file from the list_chunks.txt
global_op_counter = 1

# Reading the ordered list of config file to process
cfg_file_list = dict()
for clt in range(N_clt):
    cfg_file_list[clt] = [line.rstrip("\n") for line in open(out_folder + "/exp_files/"+"clt"+format(clt, N_clt_str_format)+"_list_chunks.txt")]
    cfg_file_list[clt].append(cfg_file_list[clt][-1])
global_cfg_file_list =  [line.rstrip("\n") for line in open(out_folder + "/exp_files/"+"global"+"_list_chunks.txt")]
global_cfg_file_list.append(global_cfg_file_list[-1])


# A variable that tells if the current chunk is the first one that is being processed:
processed_first = {clt: True for clt in range(N_clt)} 
global_processed_first = True

data_name = {clt: [] for clt in range(N_clt)} 
data_set = {clt: [] for clt in range(N_clt)} 
data_end_index = {clt: [] for clt in range(N_clt)} 
fea_dict = {clt: [] for clt in range(N_clt)} 
lab_dict = {clt: [] for clt in range(N_clt)} 
arch_dict = {clt: [] for clt in range(N_clt)} 
global_data_name = []
global_data_set = []
global_data_end_index = []
global_fea_dict = []
global_lab_dict = []
global_arch_dict = []


# 创建全局模型
[global_model, global_arch_dict] = create_nns(cfg_file)
optimizers = optimizer_init(global_model, config, global_arch_dict)
# --------TRAINING LOOP--------#
for rd in range(N_rd):
    print(
                "------------------------------ Round %s / %s ------------------------------"
                % (format(rd, N_rd_str_format), format(N_rd - 1, N_rd_str_format))
            )
    
    local_weights, local_losses, local_errors, local_times = {}, [], [], []
    m = max(int(frac * N_clt), 1)
    idxs_users = np.random.choice(range(N_clt), m, replace=False)
    # 本地训练
    for clt in idxs_users:
        
        for ep in range(N_ep):

            print(
                "------------------------------ Epoch %s / %s ------------------------------"
                % (format(ep, N_ep_str_format), format(N_ep - 1, N_ep_str_format))
            )
        
            for tr_data in tr_data_lst:
                
                # Compute the total number of chunks for each training epoch
                #N_ck_tr = compute_n_chunks(out_folder, tr_data, ep, N_ep_str_format, "train")
                N_ck_tr = fed_compute_n_chunks(out_folder, tr_data,rd,N_rd_str_format, ep, N_ep_str_format,clt, N_clt_str_format, "train")
                N_ck_str_format = "0" + str(max(math.ceil(np.log10(N_ck_tr)), 1)) + "d"

                # ***Epoch training***
                for ck in range(N_ck_tr):
                    # 分发全局模型到本地
                    info_file = (
                    out_folder
                    + "/exp_files/train_"
                    + tr_data
                    + "_rd"
                    + format(rd, N_rd_str_format)
                    + "_ep"
                    + format(ep, N_ep_str_format)
                    + "_clt"
                    + format(clt, N_clt_str_format)
                    + "_ck"
                    + format(ck, N_ck_str_format)
                    + ".info"
                    )
                    save_globa_model(global_model, multi_gpu, optimizers, info_file, global_arch_dict)

                    
                    # paths of the output files (info,model,chunk_specific cfg file)
                    info_file = (
                        out_folder
                        + "/exp_files/train_"
                        + tr_data
                        + "_rd"
                        + format(rd, N_rd_str_format)
                        + "_ep"
                        + format(ep, N_ep_str_format)
                        + "_clt"
                        + format(clt, N_clt_str_format)
                        + "_ck"
                        + format(ck, N_ck_str_format)
                        + ".info"
                    )

                    if rd + ep + ck == 0:
                        model_files_past = {}
                        model_files_past.setdefault(clt, {})
                    else:
                        model_files_past.setdefault(clt, {})
                        model_files.setdefault(clt, {})
                        model_files_past[clt] = model_files[clt]

                    model_files = {}
                    model_files.setdefault(clt, {})
                    
                
                    for arch in pt_files[clt].keys():
                        model_files[clt][arch] = info_file.replace(".info", "_" + arch + ".pkl")
                        

                    config_chunk_file = (
                        out_folder
                        + "/exp_files/train_"
                        + tr_data
                        + "_rd"
                        + format(rd, N_rd_str_format)
                        + "_ep"
                        + format(ep, N_ep_str_format)
                        + "_clt"
                        + format(clt, N_clt_str_format)
                        + "_ck"
                        + format(ck, N_ck_str_format)
                        + ".cfg"
                    )

                    # update learning rate in the cfg file (if needed)
                    change_lr_cfg(config_chunk_file, lr, rd)

                    # if this chunk has not already been processed, do training...
                    if not (os.path.exists(info_file)):

                        print("Client = %i Training %s chunk = %i / %i" % (clt, tr_data, ck + 1, N_ck_tr))
                        

                        # getting the next chunk
                        next_config_file = cfg_file_list[clt][op_counter[clt]]
                        # run chunk processing
                        [data_name[clt], data_set[clt], data_end_index[clt], fea_dict[clt], lab_dict[clt], arch_dict[clt]] = run_nn(
                            data_name[clt],
                            data_set[clt],
                            data_end_index[clt],
                            fea_dict[clt],
                            lab_dict[clt],
                            arch_dict[clt],
                            config_chunk_file,
                            processed_first[clt],
                            next_config_file,
                        )

                        # update the first_processed variable
                        processed_first[clt] = False

                        if not (os.path.exists(info_file)):
                            sys.stderr.write(
                                "ERROR: training epoch %i,client %i, chunk %i not done! File %s does not exist.\nSee %s \n"
                                % (ep, clt, ck, info_file, log_file)
                            )
                            sys.exit(0)

                    # update the operation counter
                    op_counter[clt] += 1

                    # update pt_file (used to initialized the DNN for the next chunk)
                    for pt_arch in pt_files[clt].keys():
                        pt_files[clt][pt_arch] = (
                            out_folder
                            + "/exp_files/train_"
                            + tr_data
                            + "_rd"
                            + format(rd, N_rd_str_format)
                            + "_ep"
                            + format(ep, N_ep_str_format)
                            + "_clt"
                            + format(clt, N_clt_str_format)
                            + "_ck"
                            + format(ck, N_ck_str_format)
                            + "_"
                            + pt_arch
                            + ".pkl"
                        )
                    # remove previous pkl files
                    if len(model_files_past[clt].keys()) > 0:
                        for pt_arch in pt_files[clt].keys():
                            if os.path.exists(model_files_past[clt][pt_arch]):
                                os.remove(model_files_past[clt][pt_arch])
                    
        
        
        # 累计clt的权重，损失和时间
        for tr_data in tr_data_lst:
        # Training Loss and Error out of ck loop
            tr_info_lst = sorted(
                glob.glob(out_folder + "/exp_files/train_" + tr_data +"_rd"+format(rd, N_rd_str_format)+  "*_clt"
                    + format(clt, N_clt_str_format) + "*.info")
            )
            [tr_loss, tr_error, tr_time] = compute_avg_performance(tr_info_lst)
            # 读本地模型权重
            for net in global_model.keys():
                pt_file_arch = (
                            out_folder
                            + "/exp_files/train_"
                            + tr_data
                            + "_rd"
                            + format(rd, N_rd_str_format)
                            + "_ep"
                            + format(ep, N_ep_str_format)
                            + "_clt"
                            + format(clt, N_clt_str_format)
                            + "_ck"
                            + format(ck, N_ck_str_format)
                            + "_"
                            + global_arch_dict[net][0]
                            + ".pkl"
                        )
                if use_cuda:
                    checkpoint_load = torch.load(pt_file_arch)
                else:
                    checkpoint_load = torch.load(pt_file_arch, map_location="cpu")
                if tr_data not in local_weights:
                    local_weights[tr_data] = {}
                if net not in local_weights[tr_data]:
                    local_weights[tr_data][net] = []
                local_weights[tr_data][net].append(copy.deepcopy(checkpoint_load["model_par"]))
            local_losses.append(copy.deepcopy(tr_loss))
            local_errors.append(copy.deepcopy(tr_error))
            local_times.append(copy.deepcopy(tr_time))
            

        
    
    # 保存全局模型
    global_weights = dict()
    val_time_tot = 0
    for tr_data in tr_data_lst:
        if tr_data not in global_weights:
            global_weights[tr_data] = {}
        for net in global_model.keys():
            if net not in global_weights[tr_data]:
                global_weights[net] = []
            global_weights[tr_data][net] = average_weights(local_weights[tr_data][net])
            global_model[net].load_state_dict(global_weights[tr_data][net])
        info_file = (
                    out_folder
                    + "/exp_files/train_"
                    + tr_data
                    + "_rd"
                    + format(rd, N_rd_str_format)
                    + "_ep"
                    + format(ep, N_ep_str_format)
                    + "_global"
                    + "_ck"
                    + format(ck, N_ck_str_format)
                    + ".info"
                )
        save_globa_model(global_model, multi_gpu, optimizers, info_file, global_arch_dict)
        global_model_files = dict()
        for arch in global_model.keys():
            global_model_files[arch] = info_file.replace(".info", "_" + arch + ".pkl")
        
    # 全局模型测试
    if not _is_first_validation(rd):
        valid_peformance_dict_prev = valid_peformance_dict
    valid_peformance_dict = {}
    for valid_data in valid_data_lst:
            N_ck_valid = compute_n_chunks(out_folder, valid_data,  rd, N_rd_str_format, "valid")
            N_ck_str_format_val = "0" + str(max(math.ceil(np.log10(N_ck_valid)), 1)) + "d"
            for ck_val in range(N_ck_valid):
                info_file = fed_get_val_info_file_path(
                    out_folder,
                    valid_data,
                    rd,
                    ck_val,
                    N_rd_str_format,
                    N_ck_str_format_val,
                )
                config_chunk_file = fed_get_val_cfg_file_path(
                    out_folder,
                    valid_data,
                    rd,
                    ck_val,
                    N_rd_str_format,
                    N_ck_str_format_val,
                )

                if not (os.path.exists(info_file)):
                    print("Global Validating %s chunk = %i / %i" % (valid_data, ck_val + 1, N_ck_valid))
                    next_config_file = global_cfg_file_list[global_op_counter]
                
                    global_data_name, global_data_set, global_data_end_index, global_fea_dict, global_lab_dict, global_arch_dict = run_nn(
                        global_data_name,
                        global_data_set,
                        global_data_end_index,
                        global_fea_dict,
                        global_lab_dict,
                        global_arch_dict,
                        config_chunk_file,
                        global_processed_first,
                        next_config_file,
                    )
                    global_processed_first = False
                    if not (os.path.exists(info_file)):
                        sys.stderr.write(
                            "ERROR: Global validation on epoch %i, chunk %i, valid chunk %i of dataset %s not done! File %s does not exist.\nSee %s \n"
                            % (ep, ck, ck_val, valid_data, info_file, log_file)
                        )
                        sys.exit(0)
                global_op_counter += 1
            valid_info_lst = sorted(
                glob.glob(
                    fed_get_val_info_file_path(
                        out_folder,
                        valid_data,
                        rd,
                        None,
                        N_rd_str_format,
                        N_ck_str_format_val,
                    )
                )
            )
            valid_loss, valid_error, valid_time = compute_avg_performance(valid_info_lst)
            valid_peformance_dict[valid_data] = [valid_loss, valid_error, valid_time]
            val_time_tot += valid_time
    if not _is_first_validation(rd):
        err_valid_mean = np.mean(np.asarray(list(valid_peformance_dict.values()))[:, 1])
        err_valid_mean_prev = np.mean(np.asarray(list(valid_peformance_dict_prev.values()))[:, 1])
        for lr_arch in lr.keys():
            if rd < N_rd - 1 and auto_lr_annealing[lr_arch]:
                if ((err_valid_mean_prev - err_valid_mean) / err_valid_mean) < improvement_threshold[lr_arch]:
                    new_lr_value = float(lr[lr_arch][rd]) * halving_factor[lr_arch]
                    for i in range(rd + 1, N_rd):
                        lr[lr_arch][i] = str(new_lr_value)
    # 保存一个round结果
    fed_dump_round_results(
            global_res_file_path,
            rd,
            tr_data_lst,
            sum(local_losses)/len(local_losses),
            sum(local_errors)/len(local_errors),
            sum(local_times)/len(local_times) +val_time_tot,
            valid_data_lst,
            valid_peformance_dict,
            lr,
            N_rd,
        )



for pt_arch in global_model_files.keys():
    if os.path.exists(global_model_files[pt_arch]) and not os.path.exists(out_folder + "/exp_files/final_" +  "global_"+pt_arch +  ".pkl"):
        copyfile(global_model_files[pt_arch], out_folder + "/exp_files/final_" + "global_"+pt_arch  + ".pkl")

# --------FORWARD--------#
for forward_data in forward_data_lst:

    # Compute the number of chunks
    N_ck_forward = compute_n_chunks(out_folder, forward_data, rd, N_rd_str_format, "forward")
    #N_ck_forward = fed_compute_n_chunks(out_folder, forward_data, ep, N_ep_str_format,clt, N_clt_str_format, "forward")
    N_ck_str_format = "0" + str(max(math.ceil(np.log10(N_ck_forward)), 1)) + "d"

    processes = list()
    info_files = list()
    for ck in range(N_ck_forward):
        if not is_production:
            print("Global Testing %s chunk = %i / %i" % (forward_data, ck + 1, N_ck_forward))
        else:
            print("Global Forwarding %s chunk = %i / %i" % (forward_data, ck + 1, N_ck_forward))

        # output file
        info_file = (
            out_folder
            + "/exp_files/forward_"
            + forward_data
            + "_rd"
            + format(rd, N_rd_str_format)
            + "_global"
            + "_ck"
            + format(ck, N_ck_str_format)
            + ".info"
        )
        config_chunk_file = (
            out_folder
            + "/exp_files/forward_"
            + forward_data
            + "_rd"
            + format(rd, N_rd_str_format)
            + "_global"
            + "_ck"
            + format(ck, N_ck_str_format)
            + ".cfg"
        )

        # Do forward if the chunk was not already processed
        if not (os.path.exists(info_file)):

            # Doing forward

            # getting the next chunk
            next_config_file = global_cfg_file_list[global_op_counter]

            # run chunk processing
            if _run_forwarding_in_subprocesses(config):
                shared_list = list()
                output_folder = config["exp"]["out_folder"]
                save_gpumem = strtobool(config["exp"]["save_gpumem"])
                use_cuda = strtobool(config["exp"]["use_cuda"])
                p = read_next_chunk_into_shared_list_with_subprocess(
                    read_lab_fea, shared_list, config_chunk_file, is_production, output_folder, wait_for_process=True
                )
                global_data_name, data_end_index_fea, data_end_index_lab, global_fea_dict, global_lab_dict, global_arch_dict, data_set_dict = extract_data_from_shared_list(
                    shared_list
                )
                data_set_inp, data_set_ref = convert_numpy_to_torch(data_set_dict, save_gpumem, use_cuda)
                data_set = {"input": data_set_inp, "ref": data_set_ref}
                global_data_end_index = {"fea": data_end_index_fea, "lab": data_end_index_lab}
                p = multiprocessing.Process(
                    target=run_nn,
                    kwargs={
                        "data_name": global_data_name,
                        "data_set": global_data_set,
                        "data_end_index": global_data_end_index,
                        "fea_dict": global_fea_dict,
                        "lab_dict": global_lab_dict,
                        "arch_dict": global_arch_dict,
                        "cfg_file": config_chunk_file,
                        "processed_first": False,
                        "next_config_file": None,
                    },
                )
                processes.append(p)
                if _max_nr_of_parallel_forwarding_processes(config) != -1 and len(
                    processes
                ) > _max_nr_of_parallel_forwarding_processes(config):
                    processes[0].join()
                    del processes[0]
                p.start()
            else:
                [global_data_name, global_data_set, global_data_end_index, global_fea_dict, global_lab_dict, global_arch_dict] = run_nn(
                    global_data_name,
                    global_data_set,
                    global_data_end_index,
                    global_fea_dict,
                    global_lab_dict,
                    global_arch_dict,
                    config_chunk_file,
                    global_processed_first,
                    next_config_file,
                )
                global_processed_first = False
                if not (os.path.exists(info_file)):
                    sys.stderr.write(
                        "ERROR: Global forward chunk %i of dataset %s not done! File %s does not exist.\nSee %s \n"
                        % (ck, forward_data, info_file, log_file)
                    )
                    sys.exit(0)

            info_files.append(info_file)

        # update the operation counter
        global_op_counter += 1



# --------DECODING--------#
dec_lst = glob.glob(out_folder + "/exp_files/"+"*_global"+"*_to_decode.ark")

forward_data_lst = config["data_use"]["forward_with"].split(",")
forward_outs = config["forward"]["forward_out"].split(",")
forward_dec_outs = list(map(strtobool, config["forward"]["require_decoding"].split(",")))
for data in forward_data_lst:
    for k in range(len(forward_outs)):
        if forward_dec_outs[k]:

            print("Global :Decoding %s output %s" % (data, forward_outs[k]))

            info_file = out_folder + "/exp_files/decoding_" + data +"_global_" + forward_outs[k] + ".info"

            # create decode config file
            config_dec_file = out_folder + "/decoding_" + data +"_global_" + forward_outs[k] + ".conf"
            config_dec = configparser.ConfigParser()
            config_dec.add_section("decoding")

            for dec_key in config["decoding"].keys():
                config_dec.set("decoding", dec_key, config["decoding"][dec_key])

            # add graph_dir, datadir, alidir
            lab_field = config[cfg_item2sec(config, "data_name", data)]["lab"]

            # Production case, we don't have labels
            if not is_production:
                pattern = "lab_folder=(.*)\nlab_opts=(.*)\nlab_count_file=(.*)\nlab_data_folder=(.*)\nlab_graph=(.*)"
                alidir = re.findall(pattern, lab_field)[0][0]
                config_dec.set("decoding", "alidir", os.path.abspath(alidir))

                datadir = re.findall(pattern, lab_field)[0][3]
                config_dec.set("decoding", "data", os.path.abspath(datadir))

                graphdir = re.findall(pattern, lab_field)[0][4]
                config_dec.set("decoding", "graphdir", os.path.abspath(graphdir))
            else:
                pattern = "lab_data_folder=(.*)\nlab_graph=(.*)"
                datadir = re.findall(pattern, lab_field)[0][0]
                config_dec.set("decoding", "data", os.path.abspath(datadir))

                graphdir = re.findall(pattern, lab_field)[0][1]
                config_dec.set("decoding", "graphdir", os.path.abspath(graphdir))

                # The ali dir is supposed to be in exp/model/ which is one level ahead of graphdir
                alidir = graphdir.split("/")[0 : len(graphdir.split("/")) - 1]
                alidir = "/".join(alidir)
                config_dec.set("decoding", "alidir", os.path.abspath(alidir))

            with open(config_dec_file, "w") as configfile:
                config_dec.write(configfile)

            out_folder = os.path.abspath(out_folder)
            files_dec = out_folder + "/exp_files/forward_" + data +"*_global*" + forward_outs[k] + "_to_decode.ark"
            out_dec_folder = out_folder + "/decode_" + data +"_global_" + forward_outs[k]

            if not (os.path.exists(info_file)):

                # Run the decoder
                cmd_decode = (
                    cmd
                    + config["decoding"]["decoding_script_folder"]
                    + "/"
                    + config["decoding"]["decoding_script"]
                    + " "
                    + os.path.abspath(config_dec_file)
                    + " "
                    + out_dec_folder
                    + ' "'
                    + files_dec
                    + '"'
                )
                run_shell(cmd_decode, log_file)

                # remove ark files if needed
                if not forward_save_files[k]:
                    list_rem = glob.glob(files_dec)
                    for rem_ark in list_rem:
                        os.remove(rem_ark)

            # Print WER results and write info file
            cmd_res = "./check_res_dec.sh " + out_dec_folder
            wers = run_shell(cmd_res, log_file).decode("utf-8")
            res_file = open(global_res_file_path, "a")
            res_file.write("Global : %s\n" % (wers))
            print(wers)

# Saving Loss and Err as .txt and plotting curves
if not is_production:
    #create_curves(out_folder, N_ep, valid_data_lst)
    fed_create_curves(out_folder, N_rd,-1, N_clt_str_format, valid_data_lst)
