import numpy as np
import pandas as pd
import torch
import pickle
from simulator.envs import *
from route_planning.route import *
from algorithm.AC import *
from simulator.dispatch import *
from simulator.utility import cal_distance, cal_best_route, cal_route_dir, process_memory
import json
import random
import os
import importlib
import config

# 固定随机种子，保证可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.SEED)

# 基础训练参数
maxDay = config.MAX_DAY
maxTime = config.MAX_TIME
matrixX = config.MATRIX_X
matrixY = config.MATRIX_Y
memorySize = config.MEMORY_SIZE
batchSize = config.BATCH_SIZE

# 数据读取
realOrder = pd.read_pickle('data/OrderList.pickle')      # 30天订单信息
courierInit = pd.read_pickle('data/courierInit.pickle')    # 骑手初始化信息

# 保证日志和模型保存目录存在
os.makedirs("info", exist_ok=True)
os.makedirs(os.path.join("info", "step_info"), exist_ok=True)
os.makedirs("model", exist_ok=True)

# 定义训练过程函数，参数 noise_epsilon 用于更新噪声参数
def train_model(fee_noise_epsilon=None, time_noise_epsilon=None, privacy_mode=config.PRIVACY_MODE_FEE, privacy_mode_time=config.PRIVACY_MODE_TIME):
    # 如果启用了对应的隐私模式，则使用传入的噪声参数，否则设为 None
    current_fee_noise_epsilon = fee_noise_epsilon if privacy_mode and fee_noise_epsilon is not None else None
    current_time_noise_epsilon = time_noise_epsilon if privacy_mode_time and time_noise_epsilon is not None else None

    # 初始化环境时传入两个噪声参数
    env = Region(courierInit, realOrder, matrixX, matrixY, maxDay, maxTime + 1, current_fee_noise_epsilon, current_time_noise_epsilon)
    env.set_node_info()
    env.set_courier_info()

    # 状态与动作维度
    actionDim = 7
    stateDim = 223

    # 学习率和其他训练参数
    actorLr = 0.001
    criticLr = 0.0001
    gamma = 0.9
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = ActorCritic(stateDim, actionDim, actorLr, criticLr, gamma, batchSize, device)
    replayBuffer = ReplayBuffer(memorySize, batchSize)

    # 用于记录每天训练指标的字典
    results = {}
    dayRecorder = open('info/day_reward.txt', 'a+')
    dayRecorder.truncate(0)

    dayIndex = 0
    while dayIndex < maxDay:
        originalOrderTime = 0
        realOrderTime = 0

        env.get_day_info(dayIndex)
        env.reset_clean()
        dayReward = []
        stepRecorder = open(f'info/step_info/day{dayIndex}.txt', 'a+')
        stepRecorder.truncate(0)
        print(f'Day {dayIndex}:')

        T = 0
        while T < maxTime:
            dDict = {}
            stepRecorder.write("step_info" + str(T) + ':' + ' ')
            for order in env.dayOrder[env.cityTime]:
                originalOrderTime += order.orderDistance * 4 + 3
                courierList = env.action_collect(order)
                # 计算骑手状态
                courierStateArray = env.courier_state_compute(courierList)
                # 计算订单状态
                orderStateArray = env.order_state_compute(courierList, order)
                # 计算系统整体供需特征
                supplydemandStateArray = env.sd_state_compute()
                stateArray = np.hstack((courierStateArray,
                                        supplydemandStateArray.reshape(1, 200).repeat(courierStateArray.shape[0], axis=0)))
                stateMatchArray = np.hstack((stateArray, orderStateArray))
                # 选择动作
                action = agent.take_action(stateMatchArray)
                courierSolution = courierList[action]
                addMoney = orderStateArray[action][2]
                addTime = orderStateArray[action][3]
                reward = env.cal_reward(courierSolution, addMoney, addTime)
                d = DispatchSolution()
                d.add_state(stateMatchArray)
                d.add_action(action)
                d.add_reward(reward)
                dDict[courierSolution] = d
                courierSolution.add_new_order(order)

            dispatchDict = env.step(dDict)
            T += 1

            stepReward = []
            for _, dispatch in dispatchDict.items():
                state, action, reward, nextState = process_memory(dispatch)
                replayBuffer.add(state, action, reward, nextState)
                stepReward.append(reward)
                dayReward.append(reward)
            meanStepReward = round(float(np.mean(np.array(stepReward))), 4)
            stepRecorder.write(str(meanStepReward) + "\n")

            if T == maxTime:
                for _ in range(20):
                    batchState, batchAction, batchReward, batchNextState = replayBuffer.sample()
                    agent.update(batchState, batchAction, batchReward, batchNextState, dayIndex+1)

        courierAccEfficiency = []
        for courier in env.courierList:
            courierAccEfficiency.append(courier.accEfficiency)
            env.overdueOrder += courier.route.overdueOrder
            realOrderTime += courier.accOrderTime
        fullOrder = sum(len(slotOrder) for slotOrder in env.dayOrder)
        stepRecorder.close()

        meanEff = np.mean(np.array(courierAccEfficiency))
        varEff = np.std(np.array(courierAccEfficiency))
        overdueRate = round(env.overdueOrder / fullOrder, 4)
        meanDayReward = round(float(np.mean(np.array(dayReward))), 4)

        daily_metrics = {
            "meanReward": meanDayReward,
            "meanEff": meanEff,
            "varEff": varEff,
            "overdueRate": overdueRate
        }
        results[f"day{dayIndex}"] = daily_metrics

        dayRecorder.write(f'day{dayIndex} meanReward: {meanDayReward}\n')
        dayRecorder.write(f'day{dayIndex} meanEff: {meanEff}\n')
        dayRecorder.write(f'day{dayIndex} varEff: {varEff}\n')
        dayRecorder.write(f'day{dayIndex} overdueRate: {overdueRate}\n')
        print(f'Day {dayIndex}: mean reward: {meanDayReward}.')
        print(f'Day {dayIndex}: mean efficiency: {meanEff}.')
        print(f'Day {dayIndex}: overdue rate: {overdueRate}.')
        dayIndex += 1

    dayRecorder.close()

    # 保存每天训练结果
    if PRIVACY_MODE_FEE or PRIVACY_MODE_TIME:
        suffix_parts = []
        if PRIVACY_MODE_FEE:
            suffix_parts.append(f"fee_{fee_noise_epsilon}")
        if PRIVACY_MODE_TIME:
            suffix_parts.append(f"time_{time_noise_epsilon}")
        suffix = "_".join(suffix_parts)
        result_file = f"results_experiment_noise_{suffix}.json"
    else:
        result_file = "results_experiment.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    # 保存模型
    if PRIVACY_MODE_FEE or PRIVACY_MODE_TIME:
        suffix_parts = []
        if PRIVACY_MODE_FEE:
            suffix_parts.append(f"fee_{fee_noise_epsilon}")
        if PRIVACY_MODE_TIME:
            suffix_parts.append(f"time_{time_noise_epsilon}")
        suffix = "_".join(suffix_parts)
        model_file = f"model/agent_noise_{suffix}.pth"
    else:
        model_file = "model/agent_wtnoise.pth"
    torch.save(agent, model_file)
    print(f"Training finished. Results saved to {result_file} and model saved to {model_file}.")


if config.PRIVACY_MODE_FEE or config.PRIVACY_MODE_TIME:
    # fee_noise_epsilons 取自 config.NOISE_EPSILON
    # time_noise_epsilons 取自 config.TIME_NOISE_EPSILON
    fee_epsilons = config.NOISE_EPSILON if isinstance(config.NOISE_EPSILON, list) else [config.NOISE_EPSILON]
    time_epsilons = config.TIME_NOISE_EPSILON if isinstance(config.TIME_NOISE_EPSILON, list) else [config.TIME_NOISE_EPSILON]

    for fee_epsilon in fee_epsilons:
        for time_epsilon in time_epsilons:
            print(f"Training model with fee NOISE_EPSILON = {fee_epsilon} and time NOISE_EPSILON = {time_epsilon}")
            train_model(fee_noise_epsilon=fee_epsilon, time_noise_epsilon=time_epsilon, privacy_mode=True, privacy_mode_time=True)
else:
    print("Training model without noise.")
    train_model(privacy_mode=False, privacy_mode_time=False)
