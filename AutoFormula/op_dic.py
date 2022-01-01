# Copyright (c) 2021 Dai HBG


"""
该文件定义了所有默认算子字典
"""

default_operation_dic = {'1': ['csrank', 'zscore', 'neg', 'csindneutral', 'csind', 'absv'],
                         '1_num': ['wdirect', 'tsrank', 'tskurtosis', 'tsskew',
                                   'tsmean', 'tsstd', 'tspct', 'tsdelay', 'tsdelta', 'tsmax',
                                   'tsmin', 'tsmaxpos', 'tsminpos', 'powv',
                                   'discrete'],
                         '1_num_num': ['intratsmax', 'intratsmaxpos', 'intratsmin',
                                       'intratsminpos', 'intratsmean', 'intratsstd',
                                       'tsfftreal', 'tsfftimag', 'tshpf', 'tslpf',
                                       'tsquantile', 'intratsquantile', 'tsquantileupmean',
                                       'tsquantiledownmean', 'intratsquantileupmean',
                                       'intratsquantiledownmean'],
                         '1_num_num_num': ['intraquantile', 'intraquantileupmean',
                                           'intraquantiledownmean'],
                         '2': ['add', 'prod', 'minus', 'div', 'lt', 'le', 'gt', 'ge'],
                         '2_num': ['tscorr', 'tscov', 'tsregres'],
                         '2_num_num': ['intratscorr', 'intratsregres',
                                       'bitsquantile', 'biintratsquantile', 'bitsquantileupmean',
                                       'bitsquantiledownmean', 'biintratsquantileupmean',
                                       'biintratsquantiledownmean'
                                       ],
                         '2_num_num_num': ['tssubset', 'biintraquantile', 'biintraquantileupmean',
                                           'biintraquantiledownmean'],
                         '3': ['condition', 'tsautocorr'],
                         'intra_data': ['intra_open', 'intra_high', 'intra_low',
                                        'intra_close', 'intra_avg', 'intra_volume',
                                        'intra_money']
                         }

default_dim_operation_dic = {'2_2': ['csrank', 'zscore', 'neg', 'csindneutral', 'csind', 'absv',
                                     'wdirect', 'tsrank', 'tskurtosis', 'tsskew',
                                     'tsmean', 'tsstd', 'tsdelay', 'tsdelta', 'tsmax',
                                     'tsmin', 'tsmaxpos', 'tsminpos', 'powv', 'tspct',
                                     'add', 'prod', 'minus', 'div', 'lt', 'le', 'gt', 'ge',
                                     'condition', 'tsautocorr', 'tssubset',
                                     'tsfftreal', 'tsfftimag', 'tshpf', 'tslpf',
                                     'tsquantile', 'tsquantileupmean', 'tsquantiledownmean',
                                     'bitsquantile', 'bitsquantileupmean', 'bitsquantiledownmean',
                                     'discrete'
                                     ],
                             '3_3': ['neg', 'absv', 'add', 'prod', 'minus', 'div', 'lt', 'le', 'gt', 'ge',
                                     'intratsregres',
                                     'intratsfftreal', 'intratsfftimag', 'intratshpf', 'intratslpf',
                                     'intratsquantile', 'intratsquantileupmean', 'intratsquantiledownmean',
                                     'biintratsquantile', 'biintratsquantileupmean',
                                     'biintratsquantiledownmean'],
                             '3_2': ['intratsmax', 'intratsmaxpos', 'intratsmin',
                                     'intratsminpos', 'intratsmean', 'intratsstd', 'intratscorr',
                                     'intraquantile', 'intraquantileupmean', 'intraquantiledownmean',
                                     'biintraquantile', 'biintraquantileupmean', 'biintraquantiledownmean']
                             }
