# Copyright (c) 2022 Dai HBG


"""
该文件定义了所有Cython版本得默认算子字典
"""

default_operation_dic = {'1': ['csrank', 'zscore', 'neg_2d', 'neg_3d', 'csindneutral', 'csind', 'absv_2d', 'absv_3d',
                               'log_2d', 'log_3d', 'logv_2d', 'logv_3d'],
                         '1_num': ['wdirect', 'tsrank_2d', 'tsrank_3d', 'tskurtosis_2d', 'tskurtosis_3d',
                                   'tsskew_2d', 'tsskew_3d', 'tsmean_2d', 'tsmean_3d', 'tsstd_2d', 'tsstd_3d',
                                   'tsdelay_2d', 'tsdelay_3d', 'tsdelta_2d', 'tsdelta_3d t', 'tsmax_2d', 'tsmax_3d',
                                   'tsmin_2d', 'tsmin_3d', 'tsmaxpos_2d', 'tsmaxpos_3d', 'tsminpos_2d', 'tsminpos_3d',
                                   'powv_2d', 'powv_3d', 'tspct_2d', 'tspct_3d', 'discrete'],
                         '1_num_num': ['intratsmax_3d', 'intratsmaxpos_3d', 'intratsmin_3d',
                                       'intratsminpos_3d', 'intratsmean_3d', 'intratsstd_3d',
                                       'tsfftreal', 'tsfftimag', 'tshpf', 'tslpf',
                                       'tsquantile_2d', 'tsquantile_3d', 'tsquantileupmean_2d', 'tsquantileupmean_3d',
                                       'tsquantiledownmean_2d', 'tsquantiledownmean_3d'],
                         '1_num_num_num': ['intraquantile_3d', 'intraquantileupmean_3d',
                                           'intraquantiledownmean_3d'],
                         '2': ['add_2d', 'add_3d', 'add_num_2d', 'add_num_3d',
                               'minus_2d', 'minus_3d', 'minus_num_2d', 'minus_num_3d',
                               'prod_2d', 'prod_3d', 'prod_num_2d', 'prod_num_3d',
                               'div_2d', 'div_3d', 'div_num_2d', 'div_num_3d',
                               'lt_2d', 'lt_3d', 'le_2d', 'le_3d', 'gt_2d', 'gt_3d', 'ge_d', 'ge_3d',
                               'intratsregres_3d'],
                         '2_num': ['tscorr_2d', 'tscorr_3d', 'tsregres_2d', 'tsregres_3d'],
                         '2_num_num': ['intratscorr_3d', 'intratsregres_3d',
                                       'bitsquantile_2d', 'bitsquantile_3d', 'bitsquantileupmean_2d',
                                       'bitsquantileupmean_3d', 'bitsquantiledownmean_2d', 'bitsquantiledownmean_2d'
                                       ],
                         '2_num_num_num': ['tssubset', 'biintraquantile_3d', 'biintraquantileupmean_3d',
                                           'biintraquantiledownmean_3d'],
                         '3': ['condition', 'tsautocorr'],
                         'intra_data': ['intra_open', 'intra_high', 'intra_low',
                                        'intra_close', 'intra_avg', 'intra_volume',
                                        'intra_money']
                         }

default_dim_operation_dic = {'2_2': ['csrank', 'zscore', 'neg_2d', 'csindneutral', 'csind', 'absv_2d',
                                     'wdirect', 'tsrank_2d', 'tskurtosis_2d', 'tsskew_2d',
                                     'tsmean_2d', 'tsstd_2d', 'tsdelay_2d', 'tsdelta_2d', 'tsmax_2d',
                                     'tsmin_2d', 'tsmaxpos_2d', 'tsminpos_2d', 'powv_2d', 'tspct_2d',
                                     'add_2d', 'prod_2d', 'minus_2d', 'div_2d', 'lt_2d', 'le_2d', 'gt_2d', 'ge_2d',
                                     'add_num_2d', 'prod_num_2d', 'minus_num_2d', 'div_num_2d',
                                     'condition', 'tsautocorr', 'tssubset',
                                     'tsfftreal', 'tsfftimag', 'tshpf', 'tslpf',
                                     'tsquantile_2d', 'tsquantileupmean_2d', 'tsquantiledownmean_2d',
                                     'bitsquantile_2d', 'bitsquantileupmean_2d', 'bitsquantiledownmean_2d',
                                     'discrete', 'log_2d', 'logv_2d'
                                     ],
                             '3_3': ['neg_3d', 'absv_3d', 'add_3d', 'prod_3d', 'minus_3d', 'div_3d',
                                     'add_num_3d', 'prod_num_3d', 'minus_num_3d', 'div_num_3d',
                                     'lt_3d', 'le_3d', 'gt_3d', 'ge_3d', 'log_3d', 'logv_3d',
                                     'intratsregres_3d',
                                     'intratsfftreal', 'intratsfftimag', 'intratshpf', 'intratslpf',
                                     'tsquantile_3d', 'tsquantileupmean_3d', 'tsquantiledownmean_3d',
                                     'biintraquantile_3d', 'biintraquantileupmean_3d',
                                     'biintraquantiledownmean_3d'],
                             '3_2': ['intratsmax', 'intratsmaxpos', 'intratsmin',
                                     'intratsminpos', 'intratsmean', 'intratsstd', 'intratscorr',
                                     'intraquantile', 'intraquantileupmean', 'intraquantiledownmean',
                                     'biintraquantile', 'biintraquantileupmean', 'biintraquantiledownmean']
                             }
