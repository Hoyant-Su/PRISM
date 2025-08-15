import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import pandas as pd

from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams, font_manager as fm

font_path = "../Tools/Fonts/ARIAL.TTF"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

rcParams['font.family'] = font_name

def plot_kaplan_meier(model, risk, time, indicator, log_dir, test_c_index, threshold, penalizer):
    print("[!] plot kaplan meier curve...")
    indicator = indicator.astype(np.int64)

    import pandas as pd
    df = pd.DataFrame({
    'Survival_Time': time,
    'Event': indicator,
    'Risk_Score': risk
    })

    df['Survival_Time'] = df['Survival_Time'] / 30

    median_risk = np.median(df['Risk_Score'])
    print("median of risks: ", median_risk)
    df['Risk_Group'] = ['Low' if score < median_risk else 'High' for score in df['Risk_Score']]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6, 6))

    censor_style = {
    'marker': '|',
    'ms': 6,
    'mew': 1
    }

    high_risk = df[df['Risk_Group'] == 'High']
    kmf.fit(high_risk['Survival_Time'], event_observed=high_risk['Event'])
    kmf.plot(label='High Risk', ci_show=False, color='purple')

    low_risk = df[df['Risk_Group'] == 'Low']
    try:
        kmf.fit(low_risk['Survival_Time'], event_observed=low_risk['Event'])
    except:
        print(df['Risk_Score'])
        print(low_risk)
        print("error")
        assert False

    kmf.plot(label='Low Risk', ci_show=False, color="orange")

    results = logrank_test(high_risk['Survival_Time'], low_risk['Survival_Time'], event_observed_A=high_risk['Event'], event_observed_B=low_risk['Event'])
    p_value = results.p_value

    plt.xlabel('Time (months)', fontsize=20)
    plt.ylabel('Survival Probability', fontsize=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(loc='upper right', fontsize=16)

    plt.text(x=0.05, y=0.04, s=f'P-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=20, ha='left', va='bottom')
    plt.text(x=0.05, y=0.14, s=f'C-index: {test_c_index[0]:.3f}', transform=plt.gca().transAxes, fontsize=20, ha='left', va='bottom')

    plt.ylim([0.48, 1.02])

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title(f'Kaplan-Meier Survival Curve ({model})', fontsize=20)

    plt.savefig(f"{log_dir}/KM_{model}_merged_threshold_{threshold}_penalizer_{penalizer}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()