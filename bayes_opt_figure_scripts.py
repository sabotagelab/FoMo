import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_experiment(file_name):
    with open(file_name, 'rb') as f:
        ex = pickle.load(f)
    return ex


def print_quartiles(num_exp, data):
    # get quartile percentages per experiment
    quart_df_dict = {
        "Evaluator": list(range(1, num_exp + 1)),
        "Q1": [],
        "Q2": [],
        "Q3": [],
        "Q4": []
    }
    q_inds = list(range(0, 20, 5))
    q_names = ["Q1", "Q2", "Q3", "Q4"]
    for n in range(num_exp):
        query = 'Experiment == ' + str(n + 1)
        exn = data.query(query)
        # sort by eval score descending
        exn = exn.sort_values(by=['Raw Evaluator Score'])
        for q, name in zip(q_inds, q_names):
            rands = exn['Random'][q:q + 5].sum()
            pct = (5 - rands) / 5
            quart_df_dict[name].append(pct)
    dfq = pd.DataFrame(quart_df_dict)
    print(dfq)
    print("Mean:        {0}  {1}  {2}  {3}".format(dfq['Q1'].mean(), dfq['Q2'].mean(),
                                                   dfq['Q3'].mean(), dfq['Q4'].mean()))


def plot_hist(data):
    gen_scores = data.query('Random == False')['Raw Evaluator Score']
    rand_scores = data.query('Random == True')['Raw Evaluator Score']
    gen_scores = (1 - gen_scores) * 101
    rand_scores = (1 - rand_scores) * 101
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots()
        ax.hist(gen_scores, bins=10, alpha=0.5, label="Optimized Formulas")
        ax.hist(rand_scores, bins=10, alpha=0.5, label="Random Formulas")
        plt.xlabel("Evaluator Score")
        plt.ylabel("Count")
        plt.title("Evaluator Score Distributions")
        plt.legend(loc='upper right')
        plt.savefig('score_histogram.pdf')


def print_rmse_and_r(data, print_out=True):
    all_eval_scores = data['Raw Evaluator Score']
    gen_eval_scores = data.query('Random == False')['Raw Evaluator Score']
    rand_eval_scores = data.query('Random == True')['Raw Evaluator Score']
    all_eval_scores = (1 - all_eval_scores) * 101
    gen_eval_scores = (1 - gen_eval_scores) * 101
    rand_eval_scores = (1 - rand_eval_scores) * 101
    all_model_scores = data['Raw Model Score']
    gen_model_scores = data.query('Random == False')['Raw Model Score']
    rand_model_scores = data.query('Random == True')['Raw Model Score']
    all_model_scores = (1 - all_model_scores) * 101
    gen_model_scores = (1 - gen_model_scores) * 101
    rand_model_scores = (1 - rand_model_scores) * 101

    gen_rmse = np.sqrt(np.mean((gen_eval_scores - gen_model_scores) ** 2))
    rand_rmse = np.sqrt(np.mean((rand_eval_scores - rand_model_scores) ** 2))
    all_rmse = np.sqrt(np.mean((all_eval_scores - all_model_scores) ** 2))

    gen_r = gen_eval_scores.corr(gen_model_scores)
    rand_r = rand_eval_scores.corr(rand_model_scores)
    all_r = all_eval_scores.corr(all_model_scores)
    if print_out:
        print("RMSE - Optimized: {0}, Random: {1}, All: {2}".format(gen_rmse, rand_rmse, all_rmse))
        print("PCC - Optimized: {0}, Random: {1}, All: {2}".format(gen_r, rand_r, all_r))
    return gen_rmse, rand_rmse, all_rmse, gen_r, rand_r, all_r


def print_evaluator_stats(num_exp, data):
    stat_df_dict = {
        "Evaluator": list(range(1, num_exp + 1)),
        "RMSE for Optimized Formulas": [],
        "RMSE for Random Formulas": [],
        "RMSE for All Formulas": [],
        "PCC for Optimized Formulas": [],
        "PCC for Random Formulas": [],
        "PCC for All Formulas": []
    }
    for n in range(num_exp):
        query = 'Experiment == ' + str(n + 1)
        exn = data.query(query)
        stats = print_rmse_and_r(exn, print_out=False)
        stat_df_dict["RMSE for Optimized Formulas"].append(stats[0])
        stat_df_dict["RMSE for Random Formulas"].append(stats[1])
        stat_df_dict["RMSE for All Formulas"].append(stats[2])
        stat_df_dict["PCC for Optimized Formulas"].append(stats[3])
        stat_df_dict["PCC for Random Formulas"].append(stats[4])
        stat_df_dict["PCC for All Formulas"].append(stats[5])
    dfs = pd.DataFrame(stat_df_dict)
    pd.set_option('display.max_columns', None)
    print(dfs)


def plot_all_scatter(data):
    gen_eval_scores = data.query('Random == False')['Raw Evaluator Score']
    rand_eval_scores = data.query('Random == True')['Raw Evaluator Score']
    gen_eval_scores = (1 - gen_eval_scores) * 101
    rand_eval_scores = (1 - rand_eval_scores) * 101
    gen_model_scores = data.query('Random == False')['Raw Model Score']
    rand_model_scores = data.query('Random == True')['Raw Model Score']
    gen_model_scores = (1 - gen_model_scores) * 101
    rand_model_scores = (1 - rand_model_scores) * 101
    with plt.style.context(['science', 'scatter', 'ieee']):
        fig, ax = plt.subplots()
        ax.scatter(gen_eval_scores, gen_model_scores, alpha=0.8, label="Optimized Formulas")
        ax.scatter(rand_eval_scores, rand_model_scores, alpha=0.8, label="Random Formulas", marker="s")
        plt.xlabel("Evaluator Score")
        plt.ylabel("Model Score")
        plt.title("Evaluator Score vs. Model Score")
        plt.legend(loc='upper left')
        plt.savefig('score_scatter.pdf')


def plot_evaluator_scatter(num_exp, data):
    lw = int(np.ceil(np.sqrt(num_exp)))
    with plt.style.context(['science', 'scatter', 'ieee']):
        fig, axs = plt.subplots(lw, lw, sharex=True, sharey=True, constrained_layout=True)
        for n in range(num_exp):
            query = 'Experiment == ' + str(n + 1)
            exn = data.query(query)
            gen_eval_scores = exn.query('Random == False')['Raw Evaluator Score']
            rand_eval_scores = exn.query('Random == True')['Raw Evaluator Score']
            gen_eval_scores = (1 - gen_eval_scores) * 101
            rand_eval_scores = (1 - rand_eval_scores) * 101
            gen_model_scores = exn.query('Random == False')['Raw Model Score']
            rand_model_scores = exn.query('Random == True')['Raw Model Score']
            gen_model_scores = (1 - gen_model_scores) * 101
            rand_model_scores = (1 - rand_model_scores) * 101
            x = (n // lw) % lw
            y = n % lw
            axs[x, y].scatter(gen_eval_scores, gen_model_scores, alpha=0.8, label="Optimized $\phi$")
            axs[x, y].scatter(rand_eval_scores, rand_model_scores, alpha=0.8, label="Random $\phi$", marker="s")
            # axs[x, y].axis('equal')
            title = "Evaluator " + str(n + 1)
            axs[x, y].set_title(title)
        fig.supxlabel("Evaluator Score")
        fig.supylabel("Model Score")
        fig.suptitle("Evaluator Score vs. Model Score")
        plt.legend(loc="upper right")
        plt.savefig('per_eval_score_scatter.pdf')


if __name__ == '__main__':
    # load the numpy data [xs, ys, zs]:[formulas, human score, model score]
    nex = 4
    ex1 = get_experiment('valid_experiment_cliffworld_1.pkl')
    ex2 = get_experiment('valid_experiment_cliffworld_4.pkl')
    ex3 = get_experiment('valid_experiment_cliffworld_5.pkl')
    ex4 = get_experiment('valid_experiment_cliffworld_6.pkl')

    # create column dictionary
    exp_id = [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20
    treat = [1] * 20 + [2] * 20 + [1] * 40
    random = ([False] * 10 + [True] * 10) * 4
    forms = np.concatenate((ex1[0][:, 0], ex2[0][:, 0], ex3[0][:, 0], ex4[0][:, 0]))
    sub_scr = np.concatenate((ex1[1][:, 0], ex2[1][:, 0], ex3[1][:, 0], ex4[1][:, 0]))
    mdl_scr = np.concatenate((ex1[2][:, 0], ex2[2][:, 0], ex3[2][:, 0], ex4[2][:, 0]))
    df = pd.DataFrame(
        {
            "Experiment": exp_id,
            "Treatment": treat,
            "Random": random,
            "Formula": forms,
            "Raw Evaluator Score": sub_scr,
            "Raw Model Score": mdl_scr
        }
    )

    print_quartiles(nex, df)
    plot_hist(df)
    print_rmse_and_r(df)
    plot_all_scatter(df)
    plot_evaluator_scatter(nex, df)
    print_evaluator_stats(nex, df)
