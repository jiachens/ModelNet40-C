import os
import statistics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
from config import result_dir
from config import def_ranges

DATA_DIR = result_dir
DATA_FILE = "data.csv"
markers = ['o', 'v']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'darkorange', 'seagreen', 'navy', 'purple', 'skyblue', 'olive', 'salmon', 'saddlebrown', 'slategray']


def format_data():
    data = []

    for filename in os.listdir(DATA_DIR):
        if len(filename) < 4 or filename[-4:] != ".txt":
            continue
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r') as f:
            log_str = f.readlines()
            if len(log_str) == 0:
                print("Error no content", filename)
                continue

            log_str = log_str[-1]
            if log_str[-1] == "\n":
                log_str = log_str[:-1]

        one_data = []
        tokens = log_str.split(' ')
        next_value = False
        for token in tokens:
            if not next_value and token[-1] == ":":
                next_value = True
            elif next_value:
                if token.isnumeric():
                    one_data.append(int(token))
                elif token.replace('.','',1).isdigit():
                    one_data.append(float(token))
                else:
                    if token == "nan":
                        print("Error nan", filename)
                        token = 0
                    one_data.append(token)
                next_value = False

        tokens = filename.split('.')[-2].split('_')
        if tokens[-1] == "clean":
            one_data[1] = "none"
            train_mode = '_'.join(tokens[1:len(tokens)-1])
        else:
            for index in range(len(tokens)):
                if one_data[1].split('_')[0] == tokens[index]:
                    break
            train_mode = '_'.join(tokens[1:index])
        one_data.insert(1, train_mode if train_mode != "" else "none")
        one_data.insert(5, 100 - one_data[4] * 100)
        one_data.insert(7, 100 - one_data[6] * 100)
        
        if one_data[0] not in def_ranges["model"]:
            continue
        if one_data[1] not in def_ranges["train_mode"]:
            continue
        if one_data[2] not in def_ranges["corruption"]:
            continue

        data.append(one_data)

    df_data = pd.DataFrame(data, columns=['model', 'train_mode', 'corruption', 'severity', 'acc', 'err', 'class_acc', 'class_err'])
    df_data = df_data.sort_values(['model', 'train_mode', 'corruption', 'severity'])
    df_data.to_csv(DATA_FILE, index=False)


def update_font(font):
    params = {'legend.fontsize': font,
             'axes.labelsize': font,
             'axes.titlesize': font,
             'xtick.labelsize': font,
             'ytick.labelsize': font}
    pylab.rcParams.update(params)


def load_data():
    return pd.read_csv(DATA_FILE)


def draw_train_mode_comparison(df=None, figure_path=None):
    update_font('x-small')
    if df is None:
        df = load_data()

    corruption_list = list(def_ranges["corruption"].keys())[:-1]
    model_list = list(def_ranges["model"].keys())
    train_mode_list = ["none"] + list(def_ranges["train_mode"].keys())[:4]
    metric = "err"

    dim0, dim1 = len(corruption_list), len(model_list)
    fig, ax = plt.subplots(dim0, dim1, figsize=(dim1 * 3, dim0 * 3))

    for corruption_id, corruption in enumerate(corruption_list):
        for model_id, model in enumerate(model_list):
            for train_mode_id, train_mode in enumerate(train_mode_list):
                selected_data = df[(df["model"] == model) & (df["train_mode"] == train_mode) & (df["corruption"] == corruption)].sort_values("severity")
                label = "{}".format(def_ranges["train_mode"][train_mode])
                if selected_data[metric].shape != 5:
                    print(model, train_mode, corruption)
                    continue
                ax[corruption_id, model_id].plot(def_ranges["severity"], selected_data[metric], "{}{}-".format(colors[train_mode_id], markers[0]), label=label)
            
            ax[corruption_id, model_id].set_title("Model {} and Corruption {}".format(def_ranges["model"][model], def_ranges["corruption"][corruption]))
            # ax[corruption_id, model_id].set_xlabel("Severity")
            # ax[corruption_id, model_id].set_ylabel("Accuracy")
            ax[corruption_id, model_id].legend()
            ax[corruption_id, model_id].set_xticks([1,2,3,4,5])
            # ax[corruption_id, model_id].set_ylim([0,100])

    plt.tight_layout(pad=0, w_pad=-2)
    if figure_path is None:
        plt.savefig("figures/train_mode_comparison.pdf")
    else:
        plt.savefig(figure_path)


def draw_model_comparison(df=None, figure_path=None, mode="train", metric="err", split=0):
    update_font('x-small')
    if df is None:
        df = load_data()

    corruption_list = list(def_ranges["corruption"].keys())[:-1]
    if split == 1:
        corruption_list = corruption_list[:8]
    elif split == 2:
         corruption_list = corruption_list[8:]
    model_list = list(def_ranges["model"].keys())
    if mode == "train":
        train_mode_list = ["none"] + list(def_ranges["train_mode"].keys())[:4]
    else:
        train_mode_list = ["none"] + list(def_ranges["train_mode"].keys())[4:6]
    dim0, dim1 = len(corruption_list), len(train_mode_list)
    fig, ax = plt.subplots(dim0, dim1, figsize=(dim1 * 3, dim0 * 2.3))

    for corruption_id, corruption in enumerate(corruption_list):
        for train_mode_id, train_mode in enumerate(train_mode_list):
            for model_id, model in enumerate(model_list):
                selected_data = df[(df["model"] == model) & (df["train_mode"] == train_mode) & (df["corruption"] == corruption)].sort_values("severity")
                label = "{}".format(def_ranges["model"][model])
                if selected_data[metric].shape[0] != 5:
                    print(model, train_mode, corruption)
                    continue
                ax[corruption_id, train_mode_id].plot(def_ranges["severity"], selected_data[metric], color=colors[model_id], marker=markers[0], linestyle='-', label=label)
            
            ax[corruption_id, train_mode_id].set_title("{} {} and {} corruption".format(def_ranges["train_mode"][train_mode], "training" if mode == "train" else "testing", def_ranges["corruption"][corruption]))
            # ax[corruption_id, train_mode_id].set_xlabel("Severity")
            # ax[corruption_id, train_mode_id].set_ylabel("Accuracy")
            ax[corruption_id, train_mode_id].legend()
            ax[corruption_id, train_mode_id].set_xticks([1,2,3,4,5])
    if mode == "train" and split == 1:
        plt.tight_layout(pad=2, w_pad=-3)
    elif mode == "train" and split == 2:
        plt.tight_layout(pad=2, w_pad=-1)
    elif mode == "test" and split == 1:
        plt.tight_layout(pad=2, w_pad=0)
    else:
        plt.tight_layout(pad=2, w_pad=1)

    if figure_path is None:
        plt.savefig("figures/model_comparison_{}_{}_{}.pdf".format(mode, metric, split))
    else:
        plt.savefig(figure_path)
    plt.clf()


def draw_corruption_comparison(df=None, figure_path=None):
    if df is None:
        df = load_data()

    dim0, dim1 = len(def_ranges["model"]), len(def_ranges["train_mode"])
    fig, ax = plt.subplots(dim0, dim1, figsize=(dim1 * 4, dim0 * 4))

    for model_id, model in enumerate(list(def_ranges["model"].keys())):
        for train_mode_id, train_mode in enumerate(list(def_ranges["train_mode"].keys())):
            for corruption_id, corruption in enumerate(list(def_ranges["corruption"].keys())):
                if corruption == "none":
                    continue
                selected_data = df[(df["corruption"] == corruption) & (df["train_mode"] == train_mode) & (df["model"] == model)].sort_values("severity")
                for metric_id, metric in enumerate(def_ranges["metric"]):
                    label = "{}-{}".format(corruption, metric)
                    ax[model_id, train_mode_id].plot(def_ranges["severity"], selected_data[metric], marker=markers[metric_id], color=colors[corruption_id], label=label if metric_id == 0 else None)
            
            ax[model_id, train_mode_id].set_title("Corruptions with Model {} and TrainMode {}".format(model, train_mode))
            # ax[model_id, train_mode_id].set_xlabel("Severity")
            # ax[model_id, train_mode_id].set_ylabel("Accuracy")
            ax[model_id, train_mode_id].legend()
            ax[model_id, train_mode_id].set_xticks([1,2,3,4,5])

    if figure_path is None:
        plt.savefig("figures/corruption_comparison.pdf")
    else:
        plt.savefig(figure_path)


def get_best_model(df=None):
    if df is None:
        df = load_data()

    best_model = {}

    for metric_id, metric in enumerate(["acc", "class_acc"]):
        best_model[metric] = {}
        for corruption_id, corruption in enumerate(list(def_ranges["corruption"].keys())):
            best_model[metric][corruption] = []
            for model_id, model in enumerate(list(def_ranges["model"].keys())):
                best_model[metric][corruption].append([])
                for train_mode_id, train_mode in enumerate(list(def_ranges["train_mode"].keys())):
                    selected_data = df[(df["corruption"] == corruption) & (df["train_mode"] == train_mode) & (df["model"] == model)][metric]
                    best_model[metric][corruption][model_id].append(selected_data.sum()/5)
                best_model[metric][corruption][model_id] = float(np.mean(np.array(best_model[metric][corruption][model_id])))
            best_model[metric][corruption] = list(def_ranges["model"].keys())[np.argmax(np.array(best_model[metric][corruption]))]
    
    data = []
    for metric in best_model:
        for corruption in best_model[metric]:
            data.append([metric, corruption, best_model[metric][corruption]])

    df_data = pd.DataFrame(data, columns=['metric', 'corruption', 'best_model'])
    df_data.to_csv("best_model.csv", index=False)


def get_best_train_mode(df=None):
    if df is None:
        df = load_data()

    best_train_mode = {}

    for metric_id, metric in enumerate(["acc", "class_acc"]):
        best_train_mode[metric] = {}
        for corruption_id, corruption in enumerate(list(def_ranges["corruption"].keys())):
            best_train_mode[metric][corruption] = []
            for train_mode_id, train_mode in enumerate(list(def_ranges["train_mode"].keys())):
                if train_mode in ["bn", "tent", "none", "pgd"]:
                    continue
                best_train_mode[metric][corruption].append([])
                for model_id, model in enumerate(list(def_ranges["model"].keys())):
                    selected_data = df[(df["corruption"] == corruption) & (df["model"] == model) & (df["train_mode"] == train_mode)][metric]
                    best_train_mode[metric][corruption][train_mode_id].append(selected_data.sum()/5)
                best_train_mode[metric][corruption][train_mode_id] = float(np.mean(np.array(best_train_mode[metric][corruption][train_mode_id])))
            best_train_mode[metric][corruption] = list(def_ranges["train_mode"].keys())[np.argmax(np.array(best_train_mode[metric][corruption]))]
    
    data = []
    for metric in best_train_mode:
        for corruption in best_train_mode[metric]:
            data.append([metric, corruption, best_train_mode[metric][corruption]])

    df_data = pd.DataFrame(data, columns=['metric', 'corruption', 'best_train_mode'])
    print(df_data)
    df_data.to_csv("best_train_mode.csv", index=False)
                

def get_corruption_tables(df=None, metric="acc"):
    if df is None:
        df = load_data()

    all_tables = ""
    for corruption_id, corruption in enumerate(list(def_ranges["corruption"].keys())):
        all_tables += "{}\n".format(corruption)
        data = []
        for model_id, model in enumerate(list(def_ranges["model"].keys())):
            data.append([])
            for train_mode_id, train_mode in enumerate(list(def_ranges["train_mode"].keys())):
                selected_data = df[(df["corruption"] == corruption) & (df["model"] == model) & (df["train_mode"] == train_mode)][metric].sum()/5
                data[-1].append(selected_data)
            data[-1].append(sum(data[-1])/len(data[-1]))
            data[-1] = [model] + data[-1]

        data.append(["average"])
        data[-1] += [statistics.mean([data[model_id][train_mode_id+1] for model_id, model in enumerate(def_ranges["model"])]) for train_mode_id, train_mode in enumerate(list(def_ranges["train_mode"].keys()))]

        df_data = pd.DataFrame(data, columns=["model"]+list(def_ranges["train_mode"].keys())+["average"])
        df_data.to_csv("tables/{}_table.csv".format(corruption), index=False)

        with open("tables/{}_table.csv".format(corruption), 'r') as f:
            all_tables += f.read()
        all_tables += "\n\n"

    with open("corruption_tables.csv", 'w') as f:
        f.write(all_tables)


def draw_teaser(df=None):
    update_font("xx-small")
    if df is None:
        df = load_data()

    model_dim = len(def_ranges["model"])
    fig, ax = plt.subplots(figsize=(6, 1.2))

    bar_width = 0.4
    bar_params = {
        "edgecolor": None,
    }
    for bar_id, bar in enumerate(["Clean Inputs", "Corrupted Inputs"]):
        bar_mean = []
        if bar == "Clean Inputs":
            bar_std = [0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3]
        else:
            bar_std = [0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3]
        for model_id, model in enumerate(list(def_ranges["model"].keys())):
            if bar == "Clean Inputs":
                one_bar_mean = (100 - df[(df["corruption"] == "none") & (df["model"] == model) & (df["train_mode"] == "none")]["acc"].mean() * 100)
                color = 'royalblue'
            else:
                one_bar_mean = (100 - df[(df["corruption"] != "none") & (df["model"] == model) & (df["train_mode"] == "none")]["acc"].mean() * 100)
                color = '#F15757'
            bar_mean.append(one_bar_mean)
        ax.bar(np.arange(model_dim) + (bar_id - 0.5) * bar_width, bar_mean, bar_width, color=color, label=bar, **bar_params)
    
    ax.set_ylabel("Error Rate (%)")
    ax.set_ylim([0, 40])
    ax.set_xticks(np.arange(model_dim))
    ax.set_xticklabels(list(def_ranges["model"].values()))
    ax.legend(ncol=2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    plt.tight_layout(pad=0.2)

    plt.savefig("figures/teaser.png")


def get_table_1(df=None):
    if df is None:
        df = load_data()

    train_mode_list = ["cutmix_r", "cutmix_k", "mixup", "rsmix", "pgd"]
    format_string =  " & ".join(["{: <10}"] + ["{:4.1f}" for i in range (len(train_mode_list))]) + " \\\\"
    print(" & ".join([""] + train_mode_list))
    for model, model_name in def_ranges["model"].items():
        row = [model_name]
        for train_mode in train_mode_list:
            data = (100 - df[(df["corruption"] == "none") & (df["model"] == model) & (df["train_mode"] == train_mode)]["acc"].mean() * 100)
            row.append(data)
        print(format_string.format(*row))


def get_table_2():
    df = load_data()

    format_string =  " & ".join(["{: <18}"] + ["{:4.1f}" for i in range (len(def_ranges["corruption"]))]) + " \\\\"
    print(" & ".join(["", "average"] + list(def_ranges["corruption"].values())[:-1]))
    rows = []
    for model, model_name in def_ranges["model"].items():
        row = [model_name]
        for corruption, corruption_name in def_ranges["corruption"].items():
            if corruption == "none":
                continue
            data = df[(df["corruption"] == corruption) & (df["model"] == model) & (df["train_mode"] == "none")]["err"].mean()
            row.append(data)
        average = sum(row[1:]) / len(row[1:])
        row.insert(1, average)
        print(format_string.format(*row))
        rows.append(row)
    row = ["Average"]
    for col_id in range(1, len(def_ranges["corruption"])+1):
        row.append(np.mean([r[col_id] for r in rows]))
    print(format_string.format(*row))
    

def get_table_3():
    df = load_data()

    train_mode_list = ["cutmix_r", "cutmix_k", "mixup", "rsmix", "pgd"]
    corruption_list = list(def_ranges["corruption"].keys())
    corruption_groups = [corruption_list[:5], corruption_list[5:10], corruption_list[10:15]]
    format_string =  " & ".join(["{: <10}"] + ["{:4.1f}" for i in range (20)]) + " \\\\"
    print(" & ".join(["", "average"] + ["Average", "density", "noise", "trans"] * 5))
    rows = []
    for model, model_name in def_ranges["model"].items():
        if model_name == "PointMLP-Elite":
            continue
        row = [model_name]
        x = df[(df["model"] == model) & (df["train_mode"] == "none")]["err"].mean()
        row.append(x)
        for train_mode in train_mode_list:
            group_data = []
            for corruption_group in corruption_groups:
                data = []
                for corruption in corruption_group:
                    x = df[(df["corruption"] == corruption) & (df["model"] == model) & (df["train_mode"] == train_mode)]["err"].mean()
                    data.append(x)
                group_data.append(np.mean(data))
            row.append(np.mean(group_data))
            row += group_data
        print(format_string.format(*row))
        rows.append(row)
    row = ["Average"]
    for col_id in range(1, 22):
        row.append(np.mean([r[col_id] for r in rows]))
    print(format_string.format(*row))


def get_table_4():
    df = load_data()

    # train_mode_list = ["bn", "tent"]
    train_mode_list = ["megamerger"]
    corruption_list = list(def_ranges["corruption"].keys())
    corruption_groups = [corruption_list[:5], corruption_list[5:10], corruption_list[10:15]]
    format_string =  " & ".join(["{: <10}"] + ["{:6.3f}" for i in range (4 * len(train_mode_list))]) + " \\\\"
    print(" & ".join([""] + ["Average", "density", "noise", "trans"] * len(train_mode_list)))
    rows = []
    for model, model_name in def_ranges["model"].items():
        if model_name == "PointMLP-Elite":
            continue
        row = [model_name]
        for train_mode in train_mode_list:
            group_data = []
            for corruption_group in corruption_groups:
                data = []
                for corruption in corruption_group:
                    x = df[(df["corruption"] == corruption) & (df["model"] == model) & (df["train_mode"] == train_mode)]["err"].mean()
                    data.append(x)
                group_data.append(np.mean(data))
            row.append(np.mean(group_data))
            row += group_data
        print(format_string.format(*row))
        rows.append(row)
    row = ["Average"]
    for col_id in range(1, 4 * len(train_mode_list)+1):
        row.append(np.mean([r[col_id] for r in rows]))
    print(format_string.format(*row))


def draw_severity_comparison():
    df = load_data()

    model_list = list(def_ranges["model"].keys())
    model_list.remove("pointMLP2")
    corruption_list = list(def_ranges["corruption"].keys())
    corruption_list.remove("none")
    corruption_list.remove("lidar")
    corruption_list.remove("occlusion")
    train_mode_list = list(def_ranges["train_mode"].keys())
    train_mode_list.remove("none")
    train_mode_list.remove("bn")
    train_mode_list.remove("tent")
    train_mode_list.remove("pgd")
    train_mode_list.remove("megamerger")
    train_mode_list = ["none"] + train_mode_list

    figure_dim = len(model_list)+1
    bar_dim = len(train_mode_list)
    # fig, axes = plt.subplots(figure_dim, 1, figsize=(8, figure_dim * 4))
    fig, ax = plt.subplots(figsize=(6.0, 2.4))

    bar_width = 0.18
    bar_params = {
        "edgecolor": None,
    }

    figure_data_all = []
    for model_id, model in enumerate(model_list):
        # ax = axes[model_id]
        figure_data = []
        for i, severity in enumerate(def_ranges["severity"]):
            bar_data = []
            for train_mode_id, train_mode in enumerate(train_mode_list):
                x = df[(df["corruption"] != "none") & (df["model"] == model) & (df["train_mode"] == train_mode) & (df["severity"] == severity)]["err"].mean()
                bar_data.append(x)
            figure_data.append(bar_data)
        figure_data_all.append(figure_data)
    figure_data_all = np.array(figure_data_all)
    print(figure_data_all)

    for i, severity in enumerate(def_ranges["severity"]):
        bar_mean = np.mean(figure_data_all[:,i,:], axis=0)
        bar_std = np.std(figure_data_all[:,i,:], axis=0)
        ax.bar(np.arange(bar_dim) + (i - 2) * bar_width, bar_mean, bar_width, label="Severity-"+str(severity), **bar_params)
        ax.errorbar(np.arange(bar_dim) + (i - 2) * bar_width, bar_mean, yerr=bar_std, color="k", fmt='o', markersize=2, capsize=4)

    ax.set_ylim([0, 44])
    ax.set_ylabel("Average Error Rate (%)")
    ax.set_xticks(np.arange(bar_dim))
    ax.set_xticklabels([def_ranges["train_mode"][train_mode] for train_mode in train_mode_list])
    # ax.set_title("All")
    ax.legend(ncol=3)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    
    plt.tight_layout(pad=0.2)

    plt.savefig("figures/draw_severity_comparison.pdf")


def draw_test_adaptation():
    import copy
    from collections import OrderedDict
    df = load_data()
    fig, ax = plt.subplots(figsize=(6.0, 2.4))

    best_train = OrderedDict([("occlusion", "cutmix_k"), ("lidar", "rsmix"), ("rotation", "mixup")])
    test = ["tent", "bn"]
    data = {"train": {}, "bn": {}, "tent": {}}
    for severity in [1, 2, 3, 4, 5]:
        for key in data:
            data[key][severity] = []
        for corruption, train_mode in best_train.items():
            data["train"][severity].append(df[(df["corruption"] == corruption) & (df["train_mode"] == train_mode) & (df["severity"] == severity)]["err"].mean())
            for test_mode in test:
                data[test_mode][severity].append(df[(df["corruption"] == corruption) & (df["train_mode"] == test_mode) & (df["severity"] == severity)]["err"].mean())
    for mode in ["train", "tent", "bn"]:
        data[mode][0] = [0,0,0]

    bar_width = 0.06
    default_bar_params = {
        "edgecolor": 'black',
        'linewidth': 0.5,
    }
    for mode_id, mode in enumerate(["train", "tent", "bn"]):
        for severity in [1, 2, 3, 4, 5]:
            print(mode, severity, data[mode][severity])
            color = [0.9 - 0.18 * severity] * 3
            color[mode_id] = 1
            bar_params = copy.deepcopy(default_bar_params)
            if severity == 5:
                bar_params["label"] = "Best augmentation" if mode == "train" else def_ranges["train_mode"][mode]
            ax.bar([i+(severity-0.5)*bar_width+5*mode_id*bar_width for i in range(3)], 
                    data[mode][severity],
                    bar_width,
                    color=tuple(color),
                    **bar_params
                    )
    ax.set_xticks([i + 7.5 * bar_width for i in range(3)])
    ax.set_xticklabels([def_ranges["corruption"][corruption] for corruption, train_mode in best_train.items()])
    ax.legend()
    plt.tight_layout(pad=0)
    plt.savefig("figures/draw_test_adaptation.pdf")
    


# Sets the font of matplotlib.
update_font('xx-small')

# Builds the csv data table once.
if not os.path.isfile(DATA_FILE):
    format_data()

# Figure 9-16.
draw_model_comparison()
draw_train_mode_comparison()
draw_corruption_comparison()

get_best_model()
get_best_train_mode()
get_corruption_tables()

# Figure 1.
draw_teaser()

get_table_1()
get_table_2()
get_table_3()
get_table_4()

# Figure 4.
draw_severity_comparison()
draw_test_adaptation()
