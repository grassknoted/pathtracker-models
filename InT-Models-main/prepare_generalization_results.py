import os
from glob import glob
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib import patches


glob_dirs = [

]
exclude_dirs = ['backup_models', "measure_acc.py"]
rename_model = {
    "hgru_TEST": "complete_circuit",
    "hgru_wider_32": "complete_circuit",
    "r2plus": "r2plus1",
    "imagenet_mc3": "imagenet_m3d",
    "mc3": "m3d",
    "in_timesformer_facebook": "timesformer_facebook_in",
    "hgru_soft": "softmax_circuit",
    "ffhgru_soft": "softmax_circuit",
}


exp_list = ["64_1_14", "32_1_14", "_64_1_14", "_32_1_14"]
keep_cols = ["64_1_0", "64_1_14", "64_1_25","32_1_0", "32_1_14", "32_1_25"]  # , "of_64_1_0", "of_64_1_14", "of_64_1_25", "of_32_1_0", "of_32_1_14", "of_32_1_25"]
keep_models = [
    "complete_circuit",
    "softmax_circuit",
    "imagenet_mc3",
    "imagenet_r2plus1",
    "imagenet_r3d",
    "mc3",
    "r3d",
    # "imagenet_m3d",
    # "m3d",
    # # "r2plus",
    "r2plus1",
    # "nostride_r3d",
    "nostride_video_cc_small",
    "gru",
    "timesformer_facebook",
    "nostride_video_cc_small_of",
    "timesformer_facebook_in",
    "space_in_timesformer_facebook",
    # "performer"
]
# Model groups
conv_list = [
    "imagenet_m3d",
    "m3d",
    "r3d",
    "imagenet_r3d",
    "nostride_video_cc_small",
    "imagenet_r2plus1",
    "r2plus1",
]
of_list = ["nostride_video_cc_small_of"]
trans_list = ["timesformer_facebook", "timesformer_facebook_in", "space_in_timesformer_facebook"]

figsize = (10, 2)
# figsize = (10, 6)

experiments = {}
experiments_long = {}
for k in keep_cols:
    experiments[k] = {}
    experiments_long[k] = {}
# Store max performance per experiment for each model
for d in glob_dirs:
    exp_files = [os.path.join(d, ex) for ex in exp_list]
    for e in exp_files:
        exp = e.split(os.path.sep)[-1]
        if exp[0] == "_":
            # Optical flow experiment. Rename
            tag = "of_"
        else:
            tag = ""
        model_files = glob(os.path.join(e, "*"))
        model_files = [m for m in model_files if os.path.isdir(m)]
        for m in model_files:
            test_check = glob(os.path.join(m, "test_perf*.npz"))
            if len(test_check):
                model_name = m.split(os.path.sep)[-1]
                re_check = re.search("\de", model_name)
                if re_check is not None:
                    model_name = model_name.split("_{}".format(re_check[0]))[0]
                if model_name in rename_model:
                    model_name = rename_model[model_name]

                if len(tag):
                    model_name = "{}_of".format(model_name)
                    print(model_name)

                for t in test_check:
                    dataset = [int(x) for x in re.findall("\d+", t.split(os.path.sep)[-1])]
                    this_exp = "_".join([str(x) for x in dataset[::-1]])  # Reorder this...

                    # this_exp = "{}{}".format(tag, this_exp)

                    if this_exp not in keep_cols:
                        continue
                    try:
                        d = np.load(t)
                        if "acc" in d:
                            perf = np.load(t)["acc"]  # noqa
                            scores = np.load(t)["scores"]
                        else:
                            perf = np.load(t)["arr_0"]  # noqa
                            scores = []
                    except:  # noqa
                        print("Failed to load {}".format(t))
                        perf = None
                    if "imagenet_r2plus1" in model_name:
                        print(perf, this_exp)
                    if model_name in experiments[this_exp] and perf is not None and not np.isnan(perf):
                        # if perf > model_dict[model_name]:
                        #     model_dict[model_name] = perf  # dataset + [perf]
                        # experiments[this_exp][model_name] = max(perf, experiments[this_exp][model_name])
                        if perf > experiments[this_exp][model_name]:
                            experiments[this_exp][model_name] = perf
                            experiments_long[this_exp][model_name] = scores
                    elif perf is not None and not np.isnan(perf):
                        # model_dict[this_exp][model_name] = perf  # dataset + [perf]
                        experiments[this_exp][model_name] = perf  # dataset + [perf]
                        experiments_long[this_exp][model_name] = scores  # dataset + [perf]
                    else:
                        print(perf, model_name)

# Merge the OF experiments
exp_names = experiments.keys()
of_exps = [x for x in exp_names if x[0] == "_"]
for exp in of_exps:
    exp_data = experiments[exp]
    exp_key = exp[1:]
    experiments[exp_key].update(exp_data)
    experiments.pop(exp, None)

# Create a dataframe of everything
df = pd.DataFrame.from_dict(experiments)
df = df.drop(columns=set(df.columns).difference(keep_cols))
df = df.drop(set(df.index).difference(keep_models))

# Create a df of scores
sc_experiments, sc_models, sc_scores = [], [], []
for k, v in experiments_long.items():
    for l, q in v.items():
        sc_models.append([l] * len(q))
        sc_scores.append(q)
        sc_experiments.append([k] * len(q))
sc_models = np.concatenate(sc_models, 0)
sc_scores = np.concatenate(sc_scores, 0)
sc_experiments = np.concatenate(sc_experiments, 0)
df_scores = pd.DataFrame(np.stack((sc_models, sc_scores, sc_experiments), 1), columns=["models", "scores", "experiments"])
df_scores = df_scores[np.in1d(df_scores.models, keep_models)]
# df_scores = df_scores.drop(columns=set(df_scores.columns).difference(keep_cols))
# df_scores = df_scores.drop(set(df_scores.index).difference(keep_models))

# Means
long_df = pd.melt(df.reset_index(), id_vars='index', value_vars=[x for x in keep_cols if x in df.columns])
long_df["performance"] = long_df["value"]
exps = long_df["variable"]
of_or_not = [1 if "of" in exp else 0 for exp in exps]
trimmed_exps = [exp.replace("of_", "") for exp in exps]
split_exps = [np.asarray(exp.split("_")).astype(int) for exp in trimmed_exps]
split_exps = np.stack(split_exps)
long_df["length"] = split_exps[:, 0]
long_df["speed"] = split_exps[:, 1]
long_df["distractors"] = split_exps[:, 2]
long_df["optic_flow"] = of_or_not
long_df["model"] = long_df["index"]
long_df["experiments"] = trimmed_exps
long_df = long_df.drop(columns=["variable", "value", "index"])

# Save the long_df
long_df.to_pickle("model_generalization.pkl")

# Record missing data
print(pd.isnull(df))
print(df)

# CHECK = [
#     ["timesformer_facebook", "64_14_1"],
#     ["r3d_nostride", "64_14_1"],
#     ["slowfast", "32_14_1"]
# ]

# FIX_EVAL = [
#     ["performer", "64_14_1"],
#     ["slowfast", "64_14_1"]
# ]

# # Create plots
alpha = .01
ci = 99
tick_rot = 45
lw = 8
s = 100
outdir = "perf_plots"

# long_df['performance'] = pd.to_numeric(long_df.performance, errors="coerce")
long_df['performance'] = pd.to_numeric(np.stack(long_df.performance.to_numpy())) * 100
df_scores['scores'] = pd.to_numeric(df_scores["scores"]) * 100

# First do 32-length
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'
sns.set_style("ticks")
sns.set_context("paper")
f, ax = plt.subplots(1, 3, figsize=(figsize))
# f, ax = plt.subplots(1, 3, figsize=(10, 6.))
sns.despine()

output = "32_1_14.pdf"
output = os.path.join(outdir, output)
exp_key = np.in1d(long_df.experiments, ["32_1_0", "32_1_14", "32_1_25"])
df_32 = long_df[exp_key]
exp_key = np.in1d(df_scores.experiments, ["32_1_0", "32_1_14", "32_1_25"])
df_scores_32 = df_scores[exp_key]
model_names = df_32.model.to_numpy()

# Make a colormap
def_cmap = ["#FF5FE9"] * len(model_names)  # sns.color_palette(n_colors=len(model_names), palette="Reds")
cmap = {k: v for k, v in zip(model_names, def_cmap)}
conv_cmap = ["#9a38eb"] * len(conv_list)  # sns.color_palette("Blues", n_colors=len(conv_list))
of_cmap = ["#f54f25"] * len(of_list)  # sns.color_palette("Purples", n_colors=len(conv_list))
trans_cmap = ["#348ceb"] * len(trans_list)  # sns.color_palette("Greens", n_colors=len(conv_list))

conv_count = 0
trans_count = 0
of_count = 0
for k, v in cmap.items():
    if k in conv_list:
        cmap[k] = conv_cmap[conv_count]
        conv_count += 1
    elif k in trans_list:
        cmap[k] = trans_cmap[trans_count]
        trans_count += 1
    elif k in of_list:
        cmap[k] = of_cmap[of_count]
        of_count += 1

# Get order
order_df = df_32[np.in1d(df_32.experiments, ["32_1_14"])]
order_idx = np.argsort(np.stack(order_df.performance.to_numpy()))[::-1]
order_key = order_df.model.iloc[order_idx].to_numpy()

# Get human data
human = np.load("mturk_responses/human_challenge.npz")
ci = 99

# 32_14_0
it_df = df_32[np.in1d(df_32.experiments, ["32_1_14"])]
it_df_scores = df_scores_32[np.in1d(df_scores_32.experiments, ["32_1_14"])]
# g = sns.pointplot(x="model", y="performance",
#                       data=it_df, join=False, palette="dark", n_boot=1,
#                       markers="o", s=s, plot_kws=dict(alpha=0.8), ci=None, order=order_key, ax=ax[0])  # noqa
ax[1].set_xticks(np.arange(len(order_key)))
ax[1].set_xticklabels(order_key, rotation=tick_rot, ha="right")
# ax[0].vlines(x=np.arange(len(order_key)) - 0.0, ymin=0, ymax=it_df.performance.to_numpy()[order_idx], color='black', alpha=0.1, linewidth=lw)
# ax[0].scatter(np.arange(len(order_key)), it_df.performance.to_numpy()[order_idx], s=s, cmap="tab20", c=np.arange(len(order_key)), zorder=10)  # noqa
# g = sns.barplot(x="model", y="performance", ax=ax[0], order=order_key, cmap=cmap, df=it_df)
g = sns.barplot(hue=it_df.model, x=it_df.model, y=it_df.performance, order=order_key, palette=cmap, dodge=False, saturation=2, ax=ax[0])
g.legend_.remove()

ax[1].axhline(50, color='black', linestyle='--')
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())
g.set(ylabel=None)
# ax[0].set_ylabel("Accuracy")
ax[1].set_ylim([40, 100])
ax[1].xaxis.label.set_visible(False)
ax[1].axhline(np.mean(human["challenge_train_32_14_test_32_14.npy"]), color='black', linestyle='--', alpha=0.5)  # noqa
lb = np.percentile(human["challenge_train_32_14_test_32_14.npy"], (100 - ci) / 2)
ub = np.percentile(human["challenge_train_32_14_test_32_14.npy"], (ci + (100 - ci) / 2))
ax[1].add_patch(
    patches.Rectangle(
        (-1, lb),
        len(order_key) + 1,
        ub - lb,
        alpha=0.4,
        zorder=-1,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

# 32_1_0
it_df = df_32[np.in1d(df_32.experiments, ["32_1_0"])]
it_df_scores = df_scores_32[np.in1d(df_scores_32.experiments, ["32_1_0"])]
# g = sns.pointplot(x="model", y="performance",
#                       data=it_df, join=False, palette="dark", n_boot=1,
#                       markers="o", s=s, plot_kws=dict(alpha=0.8), ci=None, order=order_key, ax=ax[1])  # noqa
ax[0].set_xticks(np.arange(len(order_key)))
ax[0].set_xticklabels(order_key, rotation=tick_rot, ha="right")
# ax[1].vlines(x=np.arange(len(order_key)) - 0.0, ymin=0, ymax=it_df.performance.to_numpy()[order_idx], color='black', alpha=0.1, linewidth=lw)
# ax[1].scatter(np.arange(len(order_key)), it_df.performance.to_numpy()[order_idx], s=s, cmap="tab20", c=np.arange(len(order_key)), alpha=1., zorder=10)  # noqa
# g = sns.barplot(x="model", y="performance", ax=ax[1], order=order_key, cmap=cmap, df=it_df)
g = sns.barplot(hue=it_df.model, x=it_df.model, y=it_df.performance, order=order_key, palette=cmap, dodge=False, saturation=2, ax=ax[1])
g.legend_.remove()
ax[0].axhline(50, color='black', linestyle='--')
ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
g.set(ylabel=None)
# ax[1].set_ylabel("Accuracy")
ax[0].set_ylim([40, 100])
ax[0].xaxis.label.set_visible(False)
ax[0].axhline(np.mean(human["challenge_train_32_14_test_32_1.npy"]), color='black', linestyle='--', alpha=0.5)  # noqa
lb = np.percentile(human["challenge_train_32_14_test_32_1.npy"], (100 - ci) / 2)
ub = np.percentile(human["challenge_train_32_14_test_32_1.npy"], (ci + (100 - ci) / 2))
ax[0].add_patch(
    patches.Rectangle(
        (-1, lb),
        len(order_key) + 1,
        ub - lb,
        alpha=0.4,
        zorder=-1,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

# 32_1_25
it_df = df_32[np.in1d(df_32.experiments, ["32_1_25"])]
it_df_scores = df_scores_32[np.in1d(df_scores_32.experiments, ["32_1_25"])]
# g = sns.pointplot(x="model", y="performance",
#                       data=it_df, join=False, palette="dark", n_boot=1,
#                       plot_kws=dict(alpha=0.8), ci=None, order=order_key, ax=ax[2])  # noqa

ax[2].set_xticks(np.arange(len(order_key)))
ax[2].set_xticklabels(order_key, rotation=tick_rot, ha="right")
# ax[2].vlines(x=np.arange(len(order_key)) - 0.0, ymin=0, ymax=it_df.performance.to_numpy()[order_idx], color='black', alpha=0.1, linewidth=lw)
# ax[2].scatter(np.arange(len(order_key)), it_df.performance.to_numpy()[order_idx], s=s, cmap="tab20", c=np.arange(len(order_key)), alpha=1., zorder=10)  # noqa
# g = sns.barplot(x="model", y="performance", ax=ax[2], order=order_key, cmap=cmap, df=it_df)
g = sns.barplot(hue=it_df.model, x=it_df.model, y=it_df.performance, order=order_key, palette=cmap, dodge=False, saturation=2, ax=ax[2])
g.legend_.remove()
ax[2].axhline(50, color='black', linestyle='--')
ax[2].yaxis.set_major_formatter(mtick.PercentFormatter())
g.set(ylabel=None)
# ax[2].set_ylabel("Accuracy")
ax[2].set_ylim([40, 100])
ax[2].xaxis.label.set_visible(False)
ax[2].axhline(np.mean(human["challenge_train_32_14_test_32_25.npy"]), color='black', linestyle='--', alpha=0.5)  # noqa
lb = np.percentile(human["challenge_train_32_14_test_32_25.npy"], (100 - ci) / 2)
ub = np.percentile(human["challenge_train_32_14_test_32_25.npy"], (ci + (100 - ci) / 2))
ax[2].add_patch(
    patches.Rectangle(
        (-1, lb),
        len(order_key) + 1,
        ub - lb,
        alpha=0.4,
        zorder=-1,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

sns.despine()
# plt.tight_layout()2plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig(output)
plt.show()
plt.close(f)

# Now do 64-length
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'
sns.set_style("ticks")
sns.set_context("paper")
f, ax = plt.subplots(1, 3, figsize=(figsize))
# f, ax = plt.subplots(1, 3, figsize=(10, 6.))
sns.despine()

output = "64_1_14.pdf"
output = os.path.join(outdir, output)
exp_key = np.in1d(long_df.experiments, ["64_1_0", "64_1_14", "64_1_25"])
df_32 = long_df[exp_key]
exp_key = np.in1d(df_scores.experiments, ["64_1_0", "64_1_14", "64_1_25"])
df_scores_32 = df_scores[exp_key]
model_names = df_32.model.to_numpy()

# # Get order
# order_df = df_32[np.in1d(df_32.experiments, ["64_1_14"])]
# order_idx = np.argsort(np.stack(order_df.performance.to_numpy()))[::-1]
# order_key = order_df.model.iloc[order_idx].to_numpy()

# 64_14_0
it_df = df_32[np.in1d(df_32.experiments, ["64_1_14"])]
it_df_scores = df_scores_32[np.in1d(df_scores_32.experiments, ["64_1_14"])]
# g = sns.pointplot(x="model", y="performance",
#                       data=it_df, join=False, palette="dark", n_boot=1,
#                       markers="o", s=s, plot_kws=dict(alpha=0.8), ci=None, order=order_key, ax=ax[0])  # noqa
ax[1].set_xticks(np.arange(len(order_key)))
ax[1].set_xticklabels(order_key, rotation=tick_rot, ha="right")
# g = sns.barplot(x="model", y="performance", ax=ax[0], order=order_key, cmap=cmap, df=it_df)
g = sns.barplot(hue=it_df.model, x=it_df.model, y=it_df.performance, order=order_key, palette=cmap, dodge=False, saturation=2, ax=ax[0])
g.legend_.remove()
# ax[0].vlines(x=np.arange(len(order_key)) - 0.0, ymin=0, ymax=it_df.performance.to_numpy()[order_idx], color='black', alpha=0.1, linewidth=lw)
# ax[0].scatter(np.arange(len(order_key)), it_df.performance.to_numpy()[order_idx], s=s, cmap="tab20", c=np.arange(len(order_key)), zorder=10)  # noqa
ax[1].axhline(50, color='black', linestyle='--')
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())
g.set(ylabel=None)
# ax[0].set_ylabel("Accuracy")
ax[1].set_ylim([40, 100])
ax[1].xaxis.label.set_visible(False)
ax[1].axhline(np.mean(human["challenge_train_64_14_test_64_14.npy"]), color='black', linestyle='--', alpha=0.5)  # noqa
lb = np.percentile(human["challenge_train_64_14_test_64_14.npy"], (100 - ci) / 2)
ub = np.percentile(human["challenge_train_64_14_test_64_14.npy"], (ci + (100 - ci) / 2))
ax[1].add_patch(
    patches.Rectangle(
        (-1, lb),
        len(order_key) + 1,
        ub - lb,
        alpha=0.4,
        zorder=-1,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

# 64_1_0
it_df = df_32[np.in1d(df_32.experiments, ["64_1_0"])]
it_df_scores = df_scores_32[np.in1d(df_scores_32.experiments, ["64_1_0"])]
# g = sns.pointplot(x="model", y="performance",
#                       data=it_df, join=False, palette="dark", n_boot=1,
#                       markers="o", s=s, plot_kws=dict(alpha=0.8), ci=None, order=order_key, ax=ax[1])  # noqa
ax[0].set_xticks(np.arange(len(order_key)))
ax[0].set_xticklabels(order_key, rotation=tick_rot, ha="right")
# ax[1].vlines(x=np.arange(len(order_key)) - 0.0, ymin=0, ymax=it_df.performance.to_numpy()[order_idx], color='black', alpha=0.1, linewidth=lw)
# ax[1].scatter(np.arange(len(order_key)), it_df.performance.to_numpy()[order_idx], s=s, cmap="tab20", c=np.arange(len(order_key)), alpha=1., zorder=10)  # noqa
# g = sns.barplot(x="model", y="performance", ax=ax[1], order=order_key, cmap=cmap, df=it_df)
g = sns.barplot(hue=it_df.model, x=it_df.model, y=it_df.performance, order=order_key, palette=cmap, dodge=False, saturation=2, ax=ax[1])
g.legend_.remove()
ax[0].axhline(50, color='black', linestyle='--')
ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
g.set(ylabel=None)
# ax[1].set_ylabel("Accuracy")
ax[0].set_ylim([40, 100])
ax[0].xaxis.label.set_visible(False)
ax[0].axhline(np.mean(human["challenge_train_64_14_test_64_1.npy"]), color='black', linestyle='--', alpha=0.5)  # noqa
lb = np.percentile(human["challenge_train_64_14_test_64_1.npy"], (100 - ci) / 2)
ub = np.percentile(human["challenge_train_64_14_test_64_1.npy"], (ci + (100 - ci) / 2))
ax[0].add_patch(
    patches.Rectangle(
        (-1, lb),
        len(order_key) + 1,
        ub - lb,
        alpha=0.4,
        zorder=-1,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

# 32_1_25
it_df = df_32[np.in1d(df_32.experiments, ["64_1_25"])]
it_df_scores = df_scores_32[np.in1d(df_scores_32.experiments, ["64_1_25"])]
# g = sns.pointplot(x="model", y="performance",
#                       data=it_df, join=False, palette="dark", n_boot=1,
#                       plot_kws=dict(alpha=0.8), ci=None, order=order_key, ax=ax[2])  # noqa

ax[2].set_xticks(np.arange(len(order_key)))
ax[2].set_xticklabels(order_key, rotation=tick_rot, ha="right")
# ax[2].vlines(x=np.arange(len(order_key)) - 0.0, ymin=0, ymax=it_df.performance.to_numpy()[order_idx], color='black', alpha=0.1, linewidth=lw)
# ax[2].scatter(np.arange(len(order_key)), it_df.performance.to_numpy()[order_idx], s=s, cmap="tab20", c=np.arange(len(order_key)), alpha=1., zorder=10)  # noqa
# g = sns.barplot(x="model", y="performance", ax=ax[2], order=order_key, cmap=cmap, df=it_df)
g = sns.barplot(hue=it_df.model, x=it_df.model, y=it_df.performance, order=order_key, palette=cmap, dodge=False, saturation=2, ax=ax[2])
g.legend_.remove()
ax[2].axhline(50, color='black', linestyle='--')
ax[2].yaxis.set_major_formatter(mtick.PercentFormatter())
g.set(ylabel=None)
# ax[2].set_ylabel("Accuracy")
ax[2].set_ylim([40, 100])
ax[2].xaxis.label.set_visible(False)
ax[2].axhline(np.mean(human["challenge_train_64_14_test_64_25.npy"]), color='black', linestyle='--', alpha=0.5)  # noqa
lb = np.percentile(human["challenge_train_64_14_test_64_25.npy"], (100 - ci) / 2)
ub = np.percentile(human["challenge_train_64_14_test_64_25.npy"], (ci + (100 - ci) / 2))
ax[2].add_patch(
    patches.Rectangle(
        (-1, lb),
        len(order_key) + 1,
        ub - lb,
        alpha=0.4,
        zorder=-1,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

sns.despine()
# plt.tight_layout()
plt.savefig(output)
plt.show()
plt.close(f)

