from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
from tbparse import SummaryReader
import numpy as np

plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.grid'] = True

if __name__ == "__main__":
  bw_key = 'charts/mean_bandwidth'
  col_key = 'charts/mean_collision_bw'
  widest_key = 'charts/mean_widest_bw'
  missed_key = 'charts/mean_missed_bw'
  bw_diff_key = 'charts/mean_bw_diff'
  fc_diff_key = 'charts/mean_fc_diff'
  reward_key = 'charts/episodic_return'

  log_dirs = ["/home/shane/src/mpar-sim/trs/logs/ppo_continuous/2_64_experiment1",
              "/home/shane/src/mpar-sim/trs/logs/ddqn/2_64_experiment1/mpar_sim"]
  labels = ["PPO", "DDQN"]

  # 2.4 GHz
  # saa_metrics = {
  #     bw_key: 0.4703232515099628*100e6,
  #     col_key: 0.005276336570344859*100e6,
  #     widest_key: 0.4707711982752102*100e6,
  #     missed_key: 0.0004479467652508627*100e6,
  #     bw_diff_key: 0.12139015498523904*100e6,
  #     fc_diff_key: 0.15262375715231874*100e6,
  # }
  # 2.64 GHz
  saa_metrics = {
      bw_key: 0.7975037796151643*100e6,
      col_key: 0.001532964793895967*100e6,
      widest_key: 0.7973797337374875*100e6,
      missed_key: -0.00012404587768131316*100e6,
      bw_diff_key: 0.20490902698742156*100e6,
      fc_diff_key: 0.07528457495416599*100e6,
  }
  
  color_cycle = ['#1f77b4', '#d62728']

  fig, axes = plt.subplots(2, 3)
  # fig.tight_layout()
  plt.setp(axes, xlim=(0, 2))
  plt.margins(0.0)
  idir = 0
  for dir in log_dirs:

    # Record metrics
    bws = []
    cols = []
    widests = []
    misses = []
    bw_diffs = []
    fc_diffs = []
    rewards = []

    folder_reader = SummaryReader(dir, pivot=True)
    step_axis = folder_reader.scalars.step
    for exp_name, reader in folder_reader.children.items():
      bws.append(reader.scalars[bw_key])
      cols.append(reader.scalars[col_key])
      widests.append(reader.scalars[widest_key])
      misses.append(reader.scalars[missed_key])
      bw_diffs.append(reader.scalars[bw_diff_key])
      fc_diffs.append(reader.scalars[fc_diff_key])
      rewards.append(reader.scalars[reward_key])

    bws = pd.concat(bws, axis=1)*100e6
    cols = pd.concat(cols, axis=1)*100e6
    widests = pd.concat(widests, axis=1)*100e6
    misses = pd.concat(misses, axis=1)*100e6
    bw_diffs = pd.concat(bw_diffs, axis=1)*100e6
    fc_diffs = pd.concat(fc_diffs, axis=1)*100e6
    rewards = pd.concat(rewards, axis=1)
    

    ax = axes[0, 0]
    ax.plot(step_axis/1e6, rewards.mean(axis=1), color=color_cycle[idir], label=labels[idir])
    ax.fill_between(step_axis/1e6, rewards.min(
        axis=1), rewards.max(axis=1), alpha=0.2, color=color_cycle[idir])
    ax.set_xlabel('Training Steps (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episodic Reward', fontsize=14, fontweight='bold')
    ax.set_title('(a) Mean episode reward')
    ax.legend()
    ax.set_ylim(-300, 0)

    ax = axes[0, 1]
    ax.plot(step_axis/1e6, bws.mean(axis=1)/1e6, color=color_cycle[idir], label=labels[idir])
    ax.fill_between(step_axis/1e6, bws.min(axis=1)/1e6,
                    bws.max(axis=1)/1e6, alpha=0.2, color=color_cycle[idir])
    
    ax.set_xlabel('Training Steps (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Radar Bandwidth (MHz)', fontsize=14, fontweight='bold')
    ax.set_title('(b) Mean radar bandwidth utilization')
    if idir == 0:
      ax.axhline(y=saa_metrics[bw_key]/1e6, linestyle='--', color='k', label='SAA')
    ax.legend()

    ax = axes[0, 2]
    ax.plot(step_axis/1e6, misses.mean(axis=1)/1e6, color=color_cycle[idir], label=labels[idir])
    ax.fill_between(step_axis/1e6, misses.min(axis=1)/1e6,
                    misses.max(axis=1)/1e6, alpha=0.2, color=color_cycle[idir])
    ax.set_xlabel('Training Steps (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Missed Bandwidth (MHz)', fontsize=14, fontweight='bold')
    ax.set_title('(c) Mean missed opportunity bandwidth')
    if idir == 0:
      ax.axhline(y=saa_metrics[missed_key]/1e6, linestyle='--', color='k', label='SAA')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(step_axis/1e6, cols.mean(axis=1)/1e6, color=color_cycle[idir], label=labels[idir])
    ax.fill_between(step_axis/1e6, cols.min(axis=1)/1e6,
                    cols.max(axis=1)/1e6, alpha=0.2, color=color_cycle[idir])
    ax.set_xlabel('Training Steps (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Collision Bandwidth (MHz)', fontsize=14, fontweight='bold')
    ax.set_title('(d) Mean collision bandwidth')
    if idir == 0:
      ax.axhline(y=saa_metrics[col_key]/1e6, linestyle='--', color='k', label='SAA')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(step_axis/1e6, bw_diffs.mean(axis=1)/1e6, color=color_cycle[idir], label=labels[idir])
    ax.fill_between(step_axis/1e6, bw_diffs.min(axis=1) /
                    1e6, bw_diffs.max(axis=1)/1e6, alpha=0.2, color=color_cycle[idir])
    ax.set_xlabel('Training Steps (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\Delta B$ (MHz/pulse)', fontsize=14, fontweight='bold')
    ax.set_title('(e) Mean change in bandwidth')
    if idir == 0:
      ax.axhline(y=saa_metrics[bw_diff_key]/1e6, linestyle='--', color='k', label='SAA')
    ax.legend()

    ax = axes[1, 2]
    ax.plot(step_axis/1e6, fc_diffs.mean(axis=1)/1e6, color=color_cycle[idir], label=labels[idir])
    ax.fill_between(step_axis/1e6, fc_diffs.min(axis=1) /
                    1e6, fc_diffs.max(axis=1)/1e6, alpha=0.2, color=color_cycle[idir])
    ax.set_xlabel('Training Steps (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\Delta f_c$ (MHz/pulse)', fontsize=14, fontweight='bold')
    ax.set_title('(f) Mean center frequency shift')
    if idir == 0:
      ax.axhline(y=saa_metrics[fc_diff_key]/1e6, linestyle='--', color='k', label='SAA')
    ax.legend()
    
    idir += 1
  fig.subplots_adjust(left=0.05, bottom=0.05, right=0.97, top=0.97)
  # plt.savefig('comparison.pdf', bbox_inches='tight')
  plt.show()

  # plt.savefig('./test.pdf')
  # print(reader.scalars)
