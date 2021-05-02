__author__ = """Peter Davidson FIA"""
__email__ = 'peter.davidson@wefox.com'
__version__ = '0.1'
__license__ = 'If you find this helpful, please say thank you'

import json
import pandas as pd, numpy as np
def shear_array(df):
# This function just shears an array
# E.g. x = np.arange(9).reshape(3,3)
#      shear_array(x)
    rows, cols = df.shape
    # Add padding on right of array to make space for shear
    temp = np.pad(df, ((0,0), (0,max(rows-cols,rows-1))), mode='constant')
    y = np.arange(rows)
    y[y < 0] += temp.shape[1]
    rx, colx = np.ogrid[:temp.shape[0], :temp.shape[1]]
    colx = colx - y[:, np.newaxis]
    return temp[rx, colx]

# Dictionary of the different models to illustrate:
models = {
    0: {'title': 'Basic cashflows, no reserves',
        'upr_fac': 0,
        'resv_fac': 0,
        'ibnr_fac': 0,
        'dac_fac': 0
    },
    1: {'title': 'UPR to spread premium earnings',
        'upr_fac': 1,
        'resv_fac': 0,
        'ibnr_fac': 0,
        'dac_fac': 0
        },
    2: {'title': 'Reserves recognise loss when reported',
        'upr_fac': 1,
        'resv_fac': 1,
        'ibnr_fac': 0,
        'dac_fac': 0
        },
    3: {'title': 'IBNR recognises loss when incurred',
        'upr_fac': 1,
        'resv_fac': 1,
        'ibnr_fac': 1,
        'dac_fac': 0
        },
    4: {'title': 'DAC to spread commission',
        'upr_fac': 1,
        'resv_fac': 1,
        'ibnr_fac': 1,
        'dac_fac': 1
        }
}
# Basic product information:
prod1 = {
    'premium': 100,
    'loss_ratio': 0.7,
    'term': 12,
    'pattern_payment': 18,
    'pattern_reporting': 6,
    'comm_rate': 0.10
}

# Cashflows for a model:
def model_cashflows(prod_data, upr_fac, resv_fac, ibnr_fac, dac_fac):
    proj_horiz = range(30)
    prod = pd.DataFrame(index=proj_horiz)
    prod.at[0, 'gwp_i'] = prod_data['premium']
    prod.gwp_i = prod.gwp_i.fillna(0)
    prod['gwp'] = prod.gwp_i.cumsum()

    # Earnings before factor:
    prod.at[0, 'upr'] = -prod_data['premium']
    prod.at[prod_data['term'], 'upr'] = 0
    prod.upr = prod.upr.interpolate()
    prod['d_upr'] = prod.upr.diff().fillna(prod.upr)
    prod['gep_i'] = prod[['gwp_i', 'd_upr']].sum(axis=1)
    prod['gep'] = prod.gep_i.cumsum()
    prod['ult_i'] = - prod.gep_i * prod_data['loss_ratio']

    # Earnings after factor:
    prod.upr = prod.upr * upr_fac
    prod['d_upr'] = prod.d_upr * upr_fac
    prod['gep_i'] = prod[['gwp_i', 'd_upr']].sum(axis=1)
    prod['gep'] = prod.gep_i.cumsum()

    prod['comm_i'] = - prod.gwp_i * prod_data['comm_rate']
    prod['commission'] = prod.comm_i.cumsum()
    prod.at[0, 'dac'] = prod_data['premium'] * prod_data['comm_rate'] * dac_fac
    prod.at[prod1['term'], 'dac'] = 0
    prod.dac = prod.dac.interpolate()
    prod['d_dac'] = prod.dac.diff().fillna(prod.dac)

    patt_claims = np.ones(prod_data['pattern_payment']) / prod_data['pattern_payment']
    claims = np.outer(prod.ult_i, patt_claims)
    claims = shear_array(claims)
    claims = pd.DataFrame(claims).melt()
    claims.columns = ['proj', 'claims_i']
    claims = claims.groupby(by='proj').sum().iloc[:len(proj_horiz)-1]

    patt_inc = np.ones(prod_data['pattern_reporting']) / prod_data['pattern_reporting']
    inc = np.outer(prod.ult_i, patt_inc)
    inc = shear_array(inc)
    inc = pd.DataFrame(inc).melt()
    inc.columns = ['proj', 'inc_i']
    inc = inc.groupby(by='proj').sum().iloc[:len(proj_horiz)-1]

    prod = pd.concat((prod, claims, inc), axis=1)
    prod = prod.fillna(0)
    prod['claims'] = prod.claims_i.cumsum()
    prod['inc'] = prod.inc_i.cumsum()
    prod['ult'] = prod.ult_i.cumsum()
    prod['reserves'] = (prod.inc - prod.claims) * resv_fac
    prod['d_reserves'] = prod.reserves.diff().fillna(prod.reserves)
    prod['ibnr'] = (prod.ult - prod.inc) * ibnr_fac
    prod['d_ibnr'] = prod.ibnr.diff().fillna(prod.ibnr)
    prod['profit'] = prod[['gwp', 'upr', 'commission', 'dac', 'claims', 'reserves', 'ibnr']].sum(axis=1)
    prod['cash'] = prod[['gwp', 'commission', 'claims']].sum(axis=1)

    profit = prod[['gwp', 'upr', 'gep', 'commission', 'dac', 'claims', 'reserves', 'ibnr', 'profit']].copy()
    asset_positions = ['cash', 'dac']
    liability_positions = ['upr', 'reserves', 'ibnr']
    bs = prod[asset_positions + liability_positions].copy()
    bs['equity'] = bs.sum(axis=1)
    bs[liability_positions] = -1 * bs[liability_positions]
    bs['liabilities'] = bs[liability_positions].sum(axis=1)
    bs['assets'] = bs[asset_positions].sum(axis=1)
    bs['dummy_zero'] = 0

    # cashflow = prod[['gwp_i', 'comm_i', 'claims_i']].sum(axis=1)
    cashflow = bs.cash.to_frame()
    # profit_m = prod[['gwp_i', 'd_upr', 'comm_i', 'd_dac', 'claims_i', 'd_reserves', 'd_ibnr']].sum(axis=1)
    profit_m = profit.profit.to_frame()

    return cashflow, profit_m, bs, profit

# Run each model and save results:
def generate_results(product, model):
    graph_data = model_cashflows(
        product,
        models[model['model_idx']]['upr_fac'],
        models[model['model_idx']]['resv_fac'],
        models[model['model_idx']]['ibnr_fac'],
        models[model['model_idx']]['dac_fac']
    )
    keys = ['cashflow_results', 'profit_m_results', 'bs_results', 'profit_results']

    cashflow_results = graph_data[0].to_dict(orient='records')
    profit_m_results = graph_data[1].to_dict(orient='records')

    bs_records = {}
    for key in ['cash', 'dac', 'equity', 'dummy_zero', 'ibnr', 'upr']:
        bs_records[key] = graph_data[2][key].to_frame().to_dict(orient='records')

    pl_records = {}
    for key in ['gwp', 'upr', 'gep', 'commission', 'dac', 'claims', 'reserves', 'ibnr', 'profit']:
        pl_records[key] = graph_data[3][key].to_frame().to_dict(orient='records')

    bs_cash = [{'x': 'assets', 'y': bs_records['cash'][i]['cash'], 'label': 'cash'} for i in range(30)]
    bs_dac = [{'x': 'assets', 'y': bs_records['dac'][i]['dac'], 'label': 'dac'} for i in range(30)]
    bs_equity = [{'x': 'equity', 'y': bs_records['equity'][i]['equity'], 'label': 'equity'} for i in range(30)]
    bs_dummy_zero = [{'x': 'equity', 'y': bs_records['dummy_zero'][i]['dummy_zero'], 'label': 'dummy_zero'} for i in range(30)]
    bs_ibnr = [{'x': 'liabilities', 'y': bs_records['ibnr'][i]['ibnr'], 'label': 'ibnr'} for i in range(30)]
    bs_upr = [{'x': 'liabilities', 'y': bs_records['upr'][i]['upr'], 'label': 'upr'} for i in range(30)]

    bs_layer_1 = list(zip(bs_cash, bs_equity, bs_ibnr))
    bs_layer_2 = list(zip(bs_dac, bs_dummy_zero, bs_upr))
    bs_results = [bs_layer_1, bs_layer_2]
    pl_results = [
        [{'x': key, 'y': pl_records[key][i][key]} for key in pl_records] for i in range(30)
    ]

    return {
        'cashflow_results': cashflow_results,
        'profit_m_results': profit_m_results,
        'bs_results': bs_results,
        'pl_results': pl_results,
        }




# import matplotlib.pyplot as plt
# import matplotlib._color_data as mcd
# from matplotlib.animation import FuncAnimation
# import matplotlib.animation as animation

# def bar_chart_data(df, stocks=None):
#     if stocks is None:
#         stocks = []
#     cols = len(df)
#     bottom = df.copy()
#     bottom.iloc[:] = 0
#     tops = df.copy()
#     tops.iloc[:] = 0

#     for c in range(cols):
#         if c == 0 or df.index[c] in stocks:
#             bottom.iloc[c] = 0
#             tops.iloc[c] = max(0, df.iloc[c])
#         else:
#             bottom.iloc[c] = bottom.iloc[c - 1] + df.iloc[c - 1]
#             tops.iloc[c] = tops.iloc[c - 1] + min(0, df.iloc[c - 1]) + max(0, df.iloc[c])
#     return bottom, tops

# def get_waterfall_colours(df, stocks=None):
#     """Returns color palette for drawing waterfall bars"""
#     default_colours = mcd.XKCD_COLORS
#     index_rgb = ['xkcd:orangered', 'xkcd:green','xkcd:blue', 'xkcd:brown'] # Red, green, blue, brown
#     palette_subset = [default_colours[i] for i in index_rgb]
#     colour_palette = []
#     for index, value in df.items():
#         if index in stocks:
#             if value >= 0:
#                 idx = 2  # Blue base
#             else:
#                 idx = 3
#         elif value >= 0:
#             idx = 1  # Green uptick
#         else:
#             idx = 0  # Red downtick
#         colour_palette.append(palette_subset[idx])
#     return colour_palette

# def colour_red_green(x):
#     return 'xkcd:green' if x >= 0 else 'xkcd:red'

# # plt.style.use('ggplot')
# plt.xkcd()

# # Set up a plot space:
# fig = plt.figure(figsize=(8, 8))

# # Monthly cashflows, balance sheet and cumulative p&l:
# ax_cash = plt.subplot(421)
# ax_cash.set(xlim=(0, 30), ylim=(0, 100))
# # ax_cash.set(xlim=(-10, 10))
# ax_cash.yaxis.set_visible(True)
# ax_pm = plt.subplot(423)
# ax_pm.set(xlim=(0, 30), ylim=(-20, 70))
# ax_pm.yaxis.set_visible(True)
# ax_bs = plt.subplot(222)
# ax_bs.set(ylim=(0, 120))
# ax_pl = plt.subplot(212)
# ax_pl.set( ylim=(-10, 100))

# # Set titles for each of the plots:
# ax_cash.set_title('cumulative cashflows')
# ax_pm.set_title('cumulative profit')
# ax_pl.set_title('profit and loss')
# ax_bs.set_title('balance sheet')

# # Initial commentary:
# model_key = 0
# fig.suptitle(models[model_key]['title'], fontsize=20)
# start_time = 0

# # Cashflow chart
# cash_line, = ax_cash.plot(0, 0)
# profit_line, = ax_pm.plot(0, 0)
# # cf_colors = [colour_red_green(i) for i in cashflow_results[model_key]]
# # rects_cf = ax_cash.barh(1, cashflow_results[model_key][start_time], color=cf_colors[start_time])
# # pm_colors = [colour_red_green(i) for i in profit_m_results[model_key]]
# # rects_pm = ax_pm.barh(1, profit_m_results[model_key][start_time], color=pm_colors[start_time])

# # P&L chart:
# labels = list(profit_results[model_key].columns)
# bar_bottoms = bar_chart_data(profit_results[model_key].iloc[start_time], stocks=['gep', 'profit'])[0]
# bar_colours = get_waterfall_colours(profit_results[4].iloc[1], stocks=['gep', 'profit'])
# rects_pl = ax_pl.bar(labels,
#                      height=[i for i in profit_results[model_key].iloc[start_time]],
#                      width=0.35,
#                      bottom=bar_bottoms,
#                      color=bar_colours)

# # Balance sheet chart:
# bs_1 = bs_results[model_key][['cash', 'liabilities', 'upr']]
# bs_2 = bs_results[model_key][['dac', 'equity', 'reserves']]
# bs_3 = bs_results[model_key][['dummy_zero', 'dummy_zero', 'ibnr']]

# bs1_start = [i for i in bs_1.iloc[start_time]]
# bs2_start = [i for i in bs_2.iloc[start_time]]
# bs3_start = [i for i in bs_3.iloc[start_time]]
# bs3_bottoms_start = [bs1_start[i] + bs2_start[i] for i in range(3)]
# x_labels = ['assets', 'equity', 'liabilities']
# rects_bs1 = ax_bs.bar(
#     x_labels,
#     height=bs1_start,
#     width=0.35,
#     color=['xkcd:blue', 'xkcd:white', 'xkcd:indigo'],
#     label='Women')
# rects_bs2 = ax_bs.bar(
#     x_labels,
#     height=bs2_start,
#     width=0.35,
#     color=['xkcd:violet', colour_red_green(bs2_start[1]), 'xkcd:magenta'],
#     bottom=bs1_start, label='Women')
# rects_bs3 = ax_bs.bar(
#     x_labels,
#     height=bs3_start,
#     width=0.35,
#     color=['xkcd:blue', 'xkcd:white',  'xkcd:lavender'],
#     bottom=bs3_bottoms_start, label='Women')

# # Initialize the annotations:
# asset_labels = ['cash', 'dac']
# liability_labels=['upr', 'reserves', 'ibnr']
# an1 = ax_bs.annotate(asset_labels[0],
#         (0, rects_bs1[0]._height/2),
#         textcoords="offset points",
#         xytext=(0.2, 0))
# an2 = ax_bs.annotate(asset_labels[1],
#                (0, rects_bs1[0]._height + rects_bs2[0]._height/ 2),
#                 textcoords="offset points",
#                xytext=(0.2, 0))
# an3 = ax_bs.annotate(liability_labels[0],
#                (2, rects_bs1[2]._height / 2),
#                textcoords="offset points",
#                xytext=(-0.2, 0))
# an4 = ax_bs.annotate(liability_labels[1],
#                (2, rects_bs1[2]._height + rects_bs2[2]._height/ 2),
#                 textcoords="offset points",
#                xytext=(-0.2, 0))
# an5 = ax_bs.annotate(liability_labels[2],
#                (2, rects_bs1[2]._height + rects_bs2[2]._height +rects_bs3[2]._height/ 2),
#                 textcoords="offset points",
#                xytext=(-0.2, 0))
# def update_bs_annotations():
#     an1.xy = (0, rects_bs1[0]._height/2)
#     an2.xy = (0, rects_bs1[0]._height + rects_bs2[0]._height / 2)
#     an3.xy = (2, rects_bs1[2]._height / 2)
#     an4.xy = (2, rects_bs1[2]._height + rects_bs2[2]._height / 2)
#     an5.xy = (2, rects_bs1[2]._height + rects_bs2[2]._height + rects_bs3[2]._height / 2)

# plt.tight_layout()
# # Text box to show time interval:
# time_text = ax_bs.text(0.3, -0.3, 'time=', transform=ax_bs.transAxes, size=16)
# time_text.set_text('time= ' + str(start_time))
# # Text on product info
# product_text = ax_bs.text(-0.2, -0.5, 'Premium: 100, loss ratio: 70%, commission: 10%', transform=ax_bs.transAxes, size=12)


# # ax.legend((eur_line, chf_line, gbp_line), ('eur', 'chf', 'gbp'), loc='lower right')
# time_key = start_time
# # Add pauses to start and end
# pad_start = 10
# pad_end = 10
# proj_periods = 30
# model_interval = pad_start + proj_periods + pad_end

# def animate(i):
#     global bs_1, bs_2, bs_3
#     global model_key, time_key
#     global pad_start, proj_periods, model_interval

#     time_key = i % model_interval
#     time_key = min(max(time_key - pad_start, 0), proj_periods-1)
#     time_text.set_text('time: ' + str(time_key))

#     if i % model_interval == 0:
#         model_key = min(int(i/model_interval), len(models.keys())-1)
#         fig.suptitle(models[model_key]['title'], fontsize=20)
#         bs_1 = bs_results[model_key][['cash', 'liabilities', 'upr']]
#         bs_2 = bs_results[model_key][['dac', 'equity', 'reserves']]
#         bs_3 = bs_results[model_key][['dummy_zero', 'dummy_zero', 'ibnr']]

#     # update cashflow:
#     cash_line.set_data(range(time_key), cashflow_results[model_key][:time_key])
#     # rects_cf[0].set_width(cashflow_results[model_key][time_key])
#     # rects_cf[0].set_color(cf_colors[key])
#     profit_line.set_data(range(time_key), profit_m_results[model_key][:time_key])
#     # rects_pm[0].set_width(profit_m_results[model_key][time_key])
#     # rects_pm[0].set_color(pm_colors[key])

#     # update bs:
#     new_bs1 = [j for j in bs_1.iloc[time_key]]
#     new_bs2 = [j for j in bs_2.iloc[time_key]]
#     new_bs3 = [j for j in bs_3.iloc[time_key]]
#     new_bs3_bottoms = [new_bs1[j] + new_bs2[j] for j in range(3)]
#     for k in range(3):
#         rects_bs1[k].set_height(new_bs1[k])
#         rects_bs2[k].set_height(new_bs2[k])
#         rects_bs2[k].set_y(new_bs1[k])
#         rects_bs2[1].set_color(colour_red_green(new_bs2[1])) # Change colour of equity bar
#         rects_bs3[k].set_height(new_bs3[k])
#         rects_bs3[k].set_y(new_bs3_bottoms[k])
#     update_bs_annotations()

#     # Update p&l waterfall:
#     new_heights = [j for j in profit_results[model_key].iloc[time_key]]
#     new_bottoms = bar_chart_data(profit_results[model_key].iloc[time_key], stocks=['gep', 'profit'])[0]
#     for k in range(len(rects_pl)):
#         rects_pl[k].set_height(new_heights[k])
#         rects_pl[k].set_y(new_bottoms[k])


# anim = FuncAnimation(
#     fig, animate, interval=100, frames=len(profit_results)*model_interval)

# plt.draw()
# plt.show()

# fldr =r'C:\Users\peter\Desktop\\'
# # anim.save(fldr + 'finance.gif', writer='imagemagick')
# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=8, metadata=dict(artist='Peter Davidson'), bitrate=1800)
# anim.save(fldr + 'finance.mp4', writer=writer)