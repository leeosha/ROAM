#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys

sys.path.append('/home/admin')

import aida
from aida.data_loader import read_odps_by_shell
from odps import ODPS
import time
import xdl
import tensorflow as tf
import numpy as np
# numba库加速
# import numba
# from numba import jit,int32,float64,void,int64
import multiprocessing as mp
import queue
import ctypes
import math

# 加载计算函数
from cal_op import CalculateOpt, CalculateBeta, CalculateSigma, CalculateL2NormLoss, update_grad

aida.reset_log_config(level='INFO')  # 更改日志级别DEBUG < INFO < WARNING < ERROR < CRITICAL
logger = aida.get_logger()

logger.info('start, cpu count: %d', mp.cpu_count())

runtime = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# 默认加上当前时间为job后缀
ctx = xdl.simple_context()
# UPDATE : 加载config文件中的job_config配置，原来的ctx._config方法有点危险，后面改为ctx.get()
# job_config=ctx._config['job_config']
job_config = ctx.get()['job_config']
initFlag = True
stage_flag = 1
current_batch = -1
with ctx.scope():
    logger.info('start, tf version: %s', str(tf.__version__))

    total_loss_tensor = tf.placeholder(tf.float64, name='total_loss_tensor')
    total_loss_mean = tf.reduce_mean(total_loss_tensor)
    total_loss_summary = tf.summary.scalar("total_loss_summary", total_loss_mean)


    # ********************************************************************#
    ##single process，即每次multiprocess的单个小规模计算逻辑，每轮输入supply会自动分配，不需要自己分配
    ##流程为从input_queue取supply进行计算，最后结果构造成跟dc一致的param_deltas格式放入output_queue，并行程序会自动merge结果
    # ********************************************************************#
    # @jit(parallel=True)
    def single_process(input_queue, output_queue):
        # logger.info("single process start")

        # 初始化梯度缓存
        cost_list = np.zeros(model_count, dtype=np.float64)
        gmv_list = np.zeros(model_count, dtype=np.float64)
        grad_alpha_list = np.zeros(model_count, dtype=np.float64)
        grad_sigma_list = np.zeros(model_count, dtype=np.float64)
        grad_eta_list = np.zeros(model_count, dtype=np.float64)
        grad_zeta_list = np.zeros(model_count, dtype=np.float64)

        # 初始化监控参数，存储格式为[sum_si,lambda_*sum_clk]
        monitor_param = np.zeros(3, dtype=np.float64)
        # 训练内计算部分耗时统计，秒计算
        # 针对不同的线程，取对应supply的结果用于分配
        num = 0
        global initFlag
        global stage_flag
        beta_zero_cnt = 0
        for supplys in input_queue:
            for ind,supply in supplys.enumerate():
                # 查看bernoulli格式传入后的样子
                # for each supply，计算一遍分配逻辑，并累计loss
                # 1.1 构造本次supply的roam_param和数据
                si = supply[0]
                # UPDATE: 关键，用keys或indexes取数据，indexes性能比较好，keys每次都要重新转indexes
                indexes = supply[1]
                scores = supply[2]
                ctrs = supply[3]
                cvrs = supply[4]
                cpcs = supply[5]
                prices = supply[6]
                roam_param_array = model_info.get_values(indexes=indexes)
                if stage_flag == 1:
                    # 2.1 根据roam_param计算beta
                    beta = CalculateBeta(roam_param_array, scores, ctrs, cvrs, cpcs, prices)
                    if beta==0:
                        beta_zero_cnt += 1
                    # 2.2 根据roam_param和beta计算xij
                    xij_array = CalculateOpt(roam_param_array, scores, ctrs, cvrs, cpcs, prices, beta)
                    # print("x_ij:")
                    # print(xij_array)
                if stage_flag == 2:
                    if initFlag:
                        beta = CalculateBeta(roam_param_array, scores, ctrs, cvrs, cpcs, prices)
                        # cache beta
                        cache_beta_values[ind] = beta
                    else:
                        # get beta from cache
                        beta = cache_beta_values[ind]
                    xij_array = CalculateSigma(roam_param_array, beta, current_batch, scores, ctrs, cvrs, cpcs, prices)
                # 计算l2NormLoss
                l2normloss = CalculateL2NormLoss(roam_param_array, xij_array, si)
                # grad_alpha_list和grad_sigma_list完全相同，去掉一次
                update_grad(roam_param_array, cost_list, gmv_list,grad_alpha_list, grad_eta_list, grad_zeta_list,  monitor_param, indexes, xij_array, si, scores, ctrs, cvrs, cpcs, prices, l2normloss)
                # 轮数增加
                num += 1
        # print("beta_zero_cnt:",beta_zero_cnt)
        # 训练结果放到param_deltas中
        # grad_alpha_list和grad_sigma_list完全相同
        param_deltas = {'cost': cost_list,
                        'gmv':gmv_list,
                        'grad_alpha': grad_alpha_list,
                        'grad_sigma': np.copy(grad_alpha_list),
                        'grad_eta': np.copy(grad_eta_list),
                        'grad_zeta': np.copy(grad_zeta_list),
                        'monitor_param': monitor_param}

        # logger.info("supply_cnt:" + str(num))
        # 放入梯度累计队列
        output_queue.put(param_deltas)
        # logger.info("queue put done")


    # ********************************************************************#
    ##主程序pipeline代码，supply跟model的定义都是全局的不用重新输入，里面为每轮的计算和merge、停止条件、以及output部分
    # ********************************************************************#
    def loop():
        logger.info('start loop')
        # 初始化summary写入器
        summary_writer = tf.summary.FileWriter(ctx.get()['summary']['output_dir'], graph=ctx._sess._sess.graph)
        # ********************************************************************#
        ##统计supply和model的全局信息
        # ********************************************************************#
        # supply信息，因为supply载入到worker是分片的，因此需要merge一下
        worker_supply_cnt = len(all_supplys)
        logger.info('context start, worker supply count: %d', len(all_supplys))
        worker_sum_si = sum(item[0] for item in all_supplys)
        logger.info('context start, worker supply sum_si: %d', worker_sum_si)
        all_supply_param_dc = aida.DataContainer()
        all_supply_param_dc.add_param(name='all_supply_param', initial_value=[0.0] * 2, value_type=tf.float64)
        all_supply_param_dc.stage_update(ctx, {'all_supply_param': np.asarray([worker_supply_cnt, worker_sum_si])})
        all_supply_param = all_supply_param_dc.get_param('all_supply_param')
        all_supply_count = all_supply_param[0]
        all_supply_sum_si = all_supply_param[1]
        del all_supply_param_dc
        # logger.info('context start, all supply count: %d', all_supply_count)
        # logger.info('context start, all supply sum_si: %d', all_supply_sum_si)
        # 全局的demand信息
        start_model = model_info.output()
        all_model_demand_pv = 0.0
        all_model_count = 0
        batch_max = 0
        for demand_hash_id in start_model:
            all_model_demand_pv += start_model[demand_hash_id][5]
            all_model_count += 1
            if start_model[demand_hash_id][8] > batch_max:
                batch_max = start_model[demand_hash_id][8]
        # logger.info('context start, all model demand pv: %d', all_model_demand_pv)
        # logger.info('context start, all model count: %d', all_model_count)
        # ********************************************************************#
        ##开始迭代训练
        # ********************************************************************#
        stop = False
        iter_param = 0
        stage2_iter_param = 0
        global stage_flag
        global initFlag
        global current_batch
        supply_num = int(all_supply_count.astype(int))
        global cache_beta_values
        cache_beta_values = mp.Array(ctypes.c_double, supply_num, lock=False)
        prev_totalLoss = 9999999  # 初始化prev_totalLoss
        min_total_loss = 9999999
        while not stop:
            # 业务的代码，最后生成一个param_deltas, {param_name_to_update: update_delta}，跟注册时一致
            begin = time.time()
            result = aida.parallel_runner.multi_process_with_inqueue(job_config['thread_num'], single_process,
                                                                     all_supplys, slice_len=2000)
            # result = aida.multi_process(job_config['thread_num'], single_process, supply)
            # logger.info('done multi process, use time: %.2f', time.time() - begin)
            # 是将结果累加上
            begin = time.time()
            dc.stage_update(ctx, result)
            # logger.info('done stage_update process, use time: %.2f', time.time() - begin)

            # 对每个订单，根据dc里面的累计梯度更新参数
            # UPDATE: 关键，更新CondensedModel必须使用类方法
            def update_func_generator(ind):
                def update_func(model, grad_norm, cost):
                    # alpha, sigma, theta, priority, demand_pv, uvsupply, frequency, batch_id, price, penality, eta, zeta
                    priority = model[4]
                    demand_pv = model[5]
                    batch_id = model[8]
                    learning_rate = job_config['alpha_learning_rate_']
                    if stage_flag==1:
                        learning_rate = learning_rate * pow(0.9,iter_param/500)
                    grad = (demand_pv - cost)/ grad_norm
                    # print("alpha_grad:",grad,",demand_pv:",demand_pv,",alpha:",model[ind])
                    if (cost >= 0 or iter_param > 0) and (current_batch <= 0 or batch_id <= current_batch):
                        model[ind] = max(0.0, model[ind] -  learning_rate  * grad)

                return update_func

            def update_func_generator_eta(ind):
                def update_func(model, grad_norm, cost):
                    # alpha, sigma, theta, priority, demand_pv, uvsupply, frequency, batch_id, price, penality, eta, zeta ,target_roi_lower_bound,roi_coeff
                    demand_pv = model[5]
                    learning_rate = job_config['eta_learning_rate_']
                    if stage_flag==1:
                        learning_rate = learning_rate * pow(0.9,iter_param/500)
                    grad = (demand_pv - cost)/ grad_norm
                    # print("eta_grad:", grad, ",eta:", model[ind])
                    if iter_param > 0 and cost>demand_pv:
                        model[ind] = max(0.0, model[ind] -  learning_rate  * grad)

                return update_func

            def update_func_generator_zeta(ind):
                def update_func(model, grad_norm, cost):
                    # alpha, sigma, theta, priority, demand_pv, uvsupply, frequency, batch_id, price, penality, eta, zeta ,target_roi_lower_bound,roi_coeff
                    demand_pv = model[5]
                    learning_rate = job_config['zeta_learning_rate_']
                    if stage_flag==1:
                        learning_rate = learning_rate * pow(0.9,iter_param/500)
                    grad = (demand_pv - cost)/ grad_norm
                    # print("zeta_grad:", grad, ",zeta:", model[ind])
                    if iter_param > 0 and cost>demand_pv:
                        model[ind] = max(0.0, model[ind] -  learning_rate  * grad)

                return update_func

            if stage_flag == 1:
                # logger.info('update_model....')
                # logger.info(dc.get_param('grad_alpha'))
                model_info.update_model(deltas=(dc.get_param('grad_alpha'),dc.get_param('cost')), update_func=update_func_generator(1))
                model_info.update_model(deltas=(dc.get_param('grad_eta'),dc.get_param('cost')), update_func=update_func_generator_eta(11))
                model_info.update_model(deltas=(dc.get_param('grad_zeta'),dc.get_param('cost')), update_func=update_func_generator_zeta(12))
            model_info.update_model(deltas=(dc.get_param('grad_sigma'),dc.get_param('cost')), update_func=update_func_generator(2))

            # 新版本不需要stage_clear了，每轮聚合后会自动清除梯度
            # dc.stage_clear(ctx)

            # logger.info('done iter:' + str(iter_param))
            # logger.info('stage: ' + str(stage_flag) + ', current batch:' + str(current_batch))
            # ********************************************************************#
            ##监控monitor
            # ********************************************************************#
            monitor_param = dc.get_param('monitor_param')
            sumAllocPV = monitor_param[0] # cost
            sumAllocClk = monitor_param[1] # lambda * cost
            l2normloss = monitor_param[2]

            sumSuccessClk = 0.0 # predict_cost
            sumAllocGmv = 0.0 # predict_gmv
            finishTaskCnt = 0
            gapLoss = 0.0
            model_alloc_pv_dict = dc.get_param('cost')
            model_alloc_gmv_dict = dc.get_param('gmv')
            for demand_hash_id in start_model:
                cache_demand_idx = model_info.get_indexes([demand_hash_id])[0]
                cache_demand_pv = float(start_model[demand_hash_id][5])
                # gap = sum alpha_j * (dj - allocated_j)
                alpha = float(start_model[demand_hash_id][1])

                gapLoss += float(start_model[demand_hash_id][1]) * (
                            cache_demand_pv - model_alloc_pv_dict[cache_demand_idx])
                if model_alloc_pv_dict[cache_demand_idx] >= cache_demand_pv:
                    finishTaskCnt += 1
                    sumSuccessClk += cache_demand_pv
                    # if model_alloc_pv_dict[cache_demand_idx] >= cache_demand_pv + 1:
                    #     logger.info('over_alloc_demand===demand_hash_id:' + str(demand_hash_id) + ';alpha:' + str(
                    #         start_model[demand_hash_id][1]) + ';theta:' + str(
                    #         start_model[demand_hash_id][3]) + ';demand_pv:' + str(cache_demand_pv) + ';alloc_pv:' + str(
                    #         model_alloc_pv_dict[cache_demand_idx]))
                else:
                    sumSuccessClk += model_alloc_pv_dict[cache_demand_idx]
                sumAllocGmv += model_alloc_gmv_dict[cache_demand_idx]

            sumSuccessGmv = sumAllocGmv * (job_config['lambda_']*sumSuccessClk/sumAllocClk)
            roi = sumSuccessGmv/sumSuccessClk
            # totalLoss = l2normloss - sumSuccessClk
            totalLoss = l2normloss - job_config['lambda_']*sumSuccessClk
            if iter_param % 10 ==0 :
                logger.info('iter:'+str(iter_param)
                        +",all_supply_sum_si:"+str(all_supply_sum_si)
                        +',all_model_count:' + str(all_model_count)
                        +',all_model_demand_pv:' + str(all_model_demand_pv)
                        +',sumSuccessClk:' + str(sumSuccessClk)
                        +',cost_rate:'+str(sumSuccessClk/all_model_demand_pv)
                        +',roi:' + str(roi)
                        +',finishTaskCnt:' + str(finishTaskCnt)
                        +',l2normloss:' + str(l2normloss)
                        +',totalLoss:' + str(totalLoss)
                    )
                # logger.info(
                #     'global_ext_param===' + 'all_supply_count:' + str(all_supply_count) + ';all_supply_sum_si:' + str(
                #         all_supply_sum_si) + ';all_model_demand_pv:' + str(all_model_demand_pv) + ';all_model_count:' + str(
                #         all_model_count))
                # logger.info('iter_ext_param===' + 'sumAllocPV:' + str(sumAllocPV) + ';sumAllocClk:' + str(sumAllocClk)
                #     + ';sumSuccessClk:' + str(sumSuccessClk)
                #     + ';roi:' + str(roi)
                #     + ';finishTaskCnt:' + str(finishTaskCnt))
                # logger.info('iter_ext_param===' + 'l2normloss:' + str(l2normloss) + ';gapLoss:' + str(
                #     gapLoss) + ';totalLoss:' + str(totalLoss))
            # 在循环中，计算loss后，输入到sess图中run一下把float格式转为summary格式，并通过summary_writer写入summary文件
            iter_total_loss_summary = ctx._sess._sess.run(total_loss_summary,
                                                          feed_dict={total_loss_tensor: float(totalLoss)})
            summary_writer.add_summary(iter_total_loss_summary, iter_param)

            # 停止条件
            if initFlag:
                logger.info('change initFlag to false')
            initFlag = False
            # if stage_flag == 2:
            #     stage2_iter_param += 1
            # if stage_flag == 2 and stage2_iter_param % job_config['batch_iter_num_'] == 0:
            #     current_batch += 1
            #     logger.info('stage2_iter = ' + str(stage2_iter_param) + ', next_batch_id = ' + str(current_batch))

            if stage_flag == 1 and (iter_param >= job_config['stage1_iter_max_'] or abs(gapLoss) < job_config[
                'gapLoss_stop_threshold_']
                # or abs(prev_totalLoss-totalLoss) < job_config['stage1_loss_diff_stop_threshold_']
                # loss变大1%，停止
                or (totalLoss<0 and totalLoss > min_total_loss and (totalLoss-min_total_loss)/abs(min_total_loss)> 0.02 )
                # loss减小幅度小于0.01%，停止
                # or (totalLoss<0 and totalLoss<min_total_loss and (min_total_loss-totalLoss)/abs(min_total_loss)< 0.0001)
                ):
                logger.info(
                    'stop stage1: iter_param=' + str(iter_param) + ', gapLoss=' + str(gapLoss) + ', totalLoss=' + str(
                        totalLoss) + ', pre_totalLoss=' + str(prev_totalLoss) + ', min_total_loss=' + str(min_total_loss))
                # stage_flag = 2
                # current_batch = 1
                # initFlag = True
                # stage2_iter_param = 0
                stop = True
            # else:
            #     if stage_flag == 2 and (
            #             current_batch > batch_max or stage2_iter_param > job_config['stage2_iter_max_']
            #             # or abs(prev_totalLoss - totalLoss)  < job_config['stage2_loss_diff_stop_threshold_']
            #             or (totalLoss<0 and totalLoss > min_total_loss and (totalLoss-min_total_loss)/abs(min_total_loss)> 0.01 )
            #             # or (totalLoss<0 and totalLoss<min_total_loss and (min_total_loss-totalLoss)/abs(min_total_loss)< 0.0001)
            #             ):
            #         stop = True
            #         logger.info('stop stage2: current_batch=' + str(current_batch) + ', stage2_iter_param=' + str(
            #             stage2_iter_param) + ', totalLoss=' + str(totalLoss) + ', pre_totalLoss=' + str(prev_totalLoss)
            #             + ', min_total_loss=' + str(min_total_loss))

            iter_param += 1
            if min_total_loss>totalLoss:
                min_total_loss = totalLoss
            prev_totalLoss = totalLoss

        # 关闭summary_writer
        summary_writer.close()
        logger.info('done update')
        # 由chief worker输出
        if 0 == ctx.worker_id:
            output()
            logger.info('output done, job finish')


    # ********************************************************************#
    ##输出结果的函数
    # ********************************************************************#
    def output():
        global runtime
        records = []
        # UPDATE: 调用output按原格式返回
        final_model = model_info.output()
        for demand_hash_id in final_model:
            records.append([demand_hash_id, final_model[demand_hash_id][1], final_model[demand_hash_id][2], final_model[demand_hash_id][11], final_model[demand_hash_id][12]])
        ds  = job_config['odps_io_config']['supply_table_partition'].split(',')[0].split('=')[1]
        pt = ds
        if len(job_config['odps_io_config']['supply_table_partition'].split(','))>1:
            hh  = job_config['odps_io_config']['supply_table_partition'].split(',')[1].split('=')[1]
            pt = ds + hh
            logger.info('output to %s/pt=%s', job_config['odps_io_config']['output_odps_table_name'], pt)
        aida.write_odps(odps_api, job_config['odps_io_config']['output_odps_table_name'], 'pt=' + pt, records)


    # ********************************************************************#
    ##解析model的函数
    # ********************************************************************#
    def convert_model(model_records):
        res = {}
        idx = 0
        valid_demand_ids = set()
        for record in model_records:
            # logger.info('read record:')
            # logger.info(record)
            # 过滤theta<=0的脏数据
            if record[5] <= 0:
                continue
            # idx放在了0位置，之后跟odps表中的列顺序一致，分别为alpha,sigma,theta,priority,demand_pv,uvsupply,frequency,batch_id,price,penality,eta,zeta,target_roi_lower_bound,roi_coeff
            target_roi_uppper_bound = 20
            res[record[0]] = np.asarray([idx] + record[3:17] + [target_roi_uppper_bound])
            # 数据修正，如果priority<0则定义priority=1
            res[record[0]][6] = 1 if res[record[0]][6] < 0 else res[record[0]][6]
            # 将idx放入demand_set
            valid_demand_ids.add(record[0])
            idx += 1
        # UPDATE: 用CondensedModel包装
        return aida.CondensedModel(res), idx, valid_demand_ids


    # ********************************************************************#
    ##初始化odps api
    # ********************************************************************#
    odps_api = ODPS(job_config['access_id'], job_config['access_key'], project=job_config['odps_io_config']['project'])
    # ********************************************************************#
    ##读取model
    # ********************************************************************#
    logger.info('read model')
    # 读取model # 时间分区后续需要优化为动态分区
    logger.info('demand_table_partition:' + job_config['odps_io_config']['demand_table_partition'])
    model_records = aida.read_odps_all(odps_api, job_config['odps_io_config']['demand_table'],
                                       job_config['odps_io_config']['demand_table_partition'])  # TODO: maxpt
    # 这里的demands为后面训练要使用的订单信息，存放在内存当中，业务方自行定义以什么形式表示及使用
    # 下一版会提供size等函数
    model_info, model_count, valid_demand_ids = convert_model(model_records)
    # 显示model大小
    logger.info("model_info load done , record number: %d" % model_count)
    # ********************************************************************#
    ##定义一个全局参数存储器，用于训练过程中的参数交互
    # ********************************************************************#
    dc = aida.DataContainer()
    grad_slice_num = job_config['grad_slice_num']
    dc.add_param(name='cost', initial_value=[0.0] * model_count, value_type=tf.float64)
    dc.add_param(name='gmv', initial_value=[0.0] * model_count, value_type=tf.float64)
    dc.add_param(name='grad_alpha', initial_value=[0.0] * model_count, value_type=tf.float64)
    dc.add_param(name='grad_sigma', initial_value=[0.0] * model_count, value_type=tf.float64)
    dc.add_param(name='grad_eta', initial_value=[0.0] * model_count, value_type=tf.float64)
    dc.add_param(name='grad_zeta', initial_value=[0.0] * model_count, value_type=tf.float64)
    dc.add_param(name='monitor_param', initial_value=[0.0] * 3, value_type=tf.float64)
    # ********************************************************************#
    ##读取supply数据，其中record_parser为record的预处理格式转换器，将比较复杂的预处理放在这里可以减少训练过程中的重复处理耗时
    # ********************************************************************#
    logger.info('read supply')
    model_info_dict = model_info.output()


    def record_parser(cols, worker_id, worker_num):
        global model_info_dict
        if hash(cols[0]) % worker_num != worker_id:
            return None
        # 将supply转换为numpy格式
        si = float(cols[2].split(',')[0])
        ad_demand_ids = cols[3].split(',')
        ad_demand_score1 = cols[5].split(',')
        ad_demand_ctr = cols[6].split(',')
        ad_demand_cvr = cols[7].split(',')
        ad_demand_cpc = cols[8].split(',')
        ad_demand_price = cols[9].split(',')
        if len(ad_demand_ids) != len(ad_demand_score1):
            logger.warning('len(ad_demand_ids)!=len(ad_demand_score1) in supply_id :' + cols[0])
            return None
        # 删除所有不在索引里的demand
        i = 0
        while i < len(ad_demand_ids):
            if int(ad_demand_ids[i]) in model_info_dict:
                i += 1
                pass
            else:
                del ad_demand_ids[i]
                del ad_demand_score1[i]
                del ad_demand_ctr[i]
                del ad_demand_cvr[i]
                del ad_demand_cpc[i]
                del ad_demand_price[i]
        if len(ad_demand_ids) == 0:
            return None
        # UPDATE: 用类方法转indexes
        ad_demand_idxs = model_info.get_indexes([int(demand_id) for demand_id in ad_demand_ids])
        # UPDATE: 把index排序一下
        sort_array = sorted(zip(ad_demand_idxs, np.asarray(ad_demand_score1, dtype=np.float64),
            np.asarray(ad_demand_ctr, dtype=np.float64),
            np.asarray(ad_demand_cvr, dtype=np.float64),
            np.asarray(ad_demand_cpc, dtype=np.float64),
            np.asarray(ad_demand_price, dtype=np.float64)))
        ad_demand_idxs, ad_demand_score1_array,ad_demand_ctr_array,ad_demand_cvr_array,ad_demand_cpc_array,ad_demand_price_array = zip(*sort_array)
        # UPDATE: 后续用到score的地方都会乘lambda，所以提前做好
        ad_demand_score1_array = np.asarray(ad_demand_score1_array) * job_config['lambda_']
        return (si, np.asarray(ad_demand_idxs), ad_demand_score1_array,np.asarray(ad_demand_ctr_array),np.asarray(ad_demand_cvr_array),np.asarray(ad_demand_cpc_array),np.asarray(ad_demand_price_array))


    all_supplys = read_odps_by_shell(access_id=job_config['access_id'], access_key=job_config['access_key'],
                                     project=job_config['odps_io_config']['supply_table_project'],
                                     table_name=job_config['odps_io_config']['supply_table'],
                                     partition=job_config['odps_io_config']['supply_table_partition'],
                                     worker_id=ctx.worker_id, worker_num=ctx.worker_num, shard_col=0,
                                     record_parser=record_parser, io_thread=4)

    # ********************************************************************#
    ##主程序运行，开始训练
    # ********************************************************************#
    ctx.start(loop)