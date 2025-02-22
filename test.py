import numpy as np
import pandas as pd
import torch
import pickle
from simulator.envs import *
from route_planning.route import *
from algorithm.AC import *
from simulator.dispatch import *
from simulator.utility import cal_distance, cal_best_route, cal_route_dir, process_memory
import random
import json
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 固定随机种子
set_seed(42)

# 测试参数设置
maxDay = 10        # 测试天数
maxTime = 179     # 每天时间步数
matrixX = 10
matrixY = 10

# 读取数据（订单数据和骑手初始化信息）
realOrder = pd.read_pickle('data/OrderList.pickle')
courierInit = pd.read_pickle('data/courierInit.pickle')

# 待测试模型路径列表
model_paths = [
    # 'model/agent_noise_0.5.pth',
    'model/agent_noise_fee_1.pth',
    # 'model/agent_noise_2.pth',
    'model/agent_noise_fee_3.pth',
    'model/agent_noise_fee_5.pth',
    'model/agent_noise_fee_100.pth',

    'model/agent_wtnoise.pth'
]
def test_model(agent, model_name, realOrder, courierInit):
    """
    对单个模型进行测试，返回一个字典，包含每一天的指标
    """
    # 每次测试前重新创建环境实例，确保环境状态独立
    env = Region(courierInit, realOrder, matrixX, matrixY, maxDay, maxTime + 1)
    env.set_node_info()
    env.set_courier_info()

    model_results = {}
    # 针对每一天进行测试
    for dayIndex in range(maxDay):
        env.get_day_info(dayIndex)
        env.reset_clean()  # 重置当天状态
        dayRewards = []
        # 逐步模拟当天的过程
        for T in range(maxTime):
            dDict = {}
            # 对当前时隙内的所有订单进行派单
            for order in env.dayOrder[env.cityTime]:
                courierList = env.action_collect(order)
                # 计算各骑手状态及订单状态
                courierStateArray = env.courier_state_compute(courierList)
                orderStateArray = env.order_state_compute(courierList, order)
                supplydemandStateArray = env.sd_state_compute()
                # 拼接状态匹配数组
                stateArray = np.hstack((courierStateArray,
                                        supplydemandStateArray.reshape(1, 200).repeat(courierStateArray.shape[0], axis=0)))
                stateMatchArray = np.hstack((stateArray, orderStateArray))
                
                # 由 agent 根据状态选择动作（测试时不进行探索更新）
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
                # 分配订单
                courierSolution.add_new_order(order)
            # 更新环境状态，推进一个时间步
            env.step(dDict)
            # 累计本时隙的奖励
            for dispatch in dDict.values():
                _, _, r, _ = process_memory(dispatch)
                dayRewards.append(r)
        # 恢复原来代码的计算方式：遍历所有骑手，累加各自的overdueOrder
        courierAccEff = []
        for courier in env.courierList:
            courierAccEff.append(courier.accEfficiency)
            env.overdueOrder += courier.route.overdueOrder
        fullOrder = 0
        for slot in env.dayOrder:
            fullOrder += len(slot)
        meanDayReward = float(np.mean(dayRewards)) if dayRewards else 0.0
        meanEff = float(np.mean(courierAccEff)) if courierAccEff else 0.0
        varEff = float(np.std(courierAccEff)) if courierAccEff else 0.0
        overdueRate = round(env.overdueOrder / fullOrder, 4) if fullOrder > 0 else 0.0

        model_results[f"day{dayIndex}"] = {
            "meanReward": round(meanDayReward, 4),
            "meanEff": round(meanEff, 4),
            "varEff": round(varEff, 4),
            "overdueRate": overdueRate
        }
        print(f"[{model_name}] Day {dayIndex}: meanReward={meanDayReward:.4f}, meanEff={meanEff:.4f}, overdueRate={overdueRate}")
    return model_results


# 对每个模型进行测试，并将结果保存到单独的 JSON 文件中
for model_path in model_paths:
    model_name = model_path.split("/")[-1].split(".")[0]
    print(f"Testing model: {model_name}")
    agent = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_results = test_model(agent, model_name, realOrder, courierInit)
    
    # 保存测试结果到以模型名称命名的文件中
    output_file = f"test_results_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(model_results, f, indent=4)
    print(f"Test results for {model_name} saved to {output_file}")
