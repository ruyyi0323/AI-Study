def generate():
    '''
    generate initial status of chessplate
    :return: initial chessplate
    '''
    global n;
    next_status = [];  # 下一步要走的状态，在第一步时，为初始生成的棋盘状态
    for i in range(n):
        next_status.append(i)

    for i in range(n):
        r = random.randint(0, 10)
        r = r % n
        next_status[r], next_status[n - r - 1] = next_status[n - r - 1], next_status[r]

    return next_status


def getconflict(status):
    '''
    compute the conflict (also heuristic function value) of this chessplate
    :param status: the chessplate status that under computation
    :return: the number of confliction
    '''
    num = 0
    for i, j in [[i, j] for i in range(len(status)) for j in range(i+1,len(status))]:
        if status[i] == status[j]:  num += 1  # 列发生冲突
        if abs(status[i] - status[j]) == j - i: num += 1  # 发生冲突
    if num == 0:return num
    else:return num + 10


def prettyprint(solution):
    '''
    formalized print the status ,crossing represent queens, dots represent no-queen
    :param solution: input chessplate
    :return: nothing
    '''
    print(solution)

    def line(pos, length=len(solution)):
        return '. ' * (pos) + 'X ' + '. ' * (length - pos - 1)

    for pos in solution:
        print(line(pos))


def a_star():
    '''
    main procedure of a-star algorithm
    :return: the final status or f value valid path
    when find solution or openlist is empty, consider it as failure
    otherwise, continue search neighbors until the algorithm find a node whose heuristic value is 0
    '''
    global chess_status_count, success_time, all_count, n, final_status, init_status, final_f_value, effective_path
    chess_status_count = 0;  # 棋盘变更状态次数
    status = generate();
    init_status = list(status);
    openlist,closelist = [],[];
    # 将开始节点放入开放列表(开始节点的F和G值都视为0);
    openlist.append([status, getconflict(status), 0, getconflict(status), status])
    while len(openlist) != 0:
        if 0 in [x[1] for x in openlist]: break
        # 如果没有找到最优解(h_value)，且openlist不为空(len),则执行循环
        neighborlist = [];  # 当前搜索节点的近邻节点列表
        # 在开放列表中查找具有最小F值的节点, 并把查找到的节点作为当前节点;
        min_f_index = [x[3] for x in openlist].index(min([x[3] for x in openlist]))
        current_status = openlist[min_f_index][0]
        # 把当前节点从开放列表删除, 加入到封闭列表
        closelist.append(openlist[min_f_index])
        del (openlist[min_f_index])
        # 找到相邻节点
        for i in current_status:
            j = 0;
            neighbor = list(current_status)
            while j == i: j = random.randint(0, n - 1)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            # 随机交换棋盘中的两个皇后的列数
            if neighbor not in neighborlist: neighborlist.append(neighbor)

        # 循环结束条件:
        # 当终点节点被加入到开放列表作为待检验节点时, 表示路径被找到,此时应终止循环;
        # 或者当开放列表为空,表明已无可以添加的新节点,而已检验的节点中没有终点节点则意味着路径无法被找到,此时也结束循环;
        for neighbor in neighborlist:
            if neighbor not in [x[0] for x in closelist]:
                # 如果该相邻节点不可通行或者该相邻节点已经在封闭列表中,则什么操作也不执行,继续检验下一个节点
                if neighbor not in [x[0] for x in openlist]:
                    # 如果该相邻节点不在开放列表中, 则将该节点添加到开放列表中, 并将该相邻节点的父节点设为当前节点, 同时保存该相邻节点的G和F值
                    moving_cost = np.abs(list(map(operator.sub, neighbor, current_status)))  # 计算皇后移动的步数
                    parent_g_value = closelist[([x[0] for x in closelist].index(current_status))][2]
                    h_value = getconflict(neighbor)
                    g_value = 20 + pow(max(moving_cost), 2) * 2 + parent_g_value
                    # 当前移动的cost + 父节点的G值
                    f_value = h_value + g_value
                    openlist.append([neighbor, h_value, g_value, f_value, current_status])
                    if h_value == 0:
                        final_status = neighbor
                        final_f_value = f_value
                        temp_g_value = 9999;
                        effective_path = [neighbor];
                        prev_status = current_status
                        while temp_g_value > 0:
                            prev_status = closelist[([x[0] for x in closelist].index(prev_status))][4]
                            temp_g_value = closelist[([x[0] for x in closelist].index(prev_status))][2]
                            effective_path.append(prev_status)
                        break
                elif neighbor in [x[0] for x in openlist]:
                    # 如果该相邻节点在开放列表中, 则判断若经由当前节点到达该相邻节点的G值是否小于原来保存的G值, 若小于, 则将该相邻节点的父节点设为当前节点, 并重新设置该相邻节点的G和F值
                    moving_cost = np.abs(list(map(operator.sub, neighbor, current_status)))  # 计算皇后移动的步数
                    parent_g_value = closelist[([x[0] for x in closelist].index(current_status))][2]
                    g_value = 20 + pow(max(moving_cost), 2) * 2 + parent_g_value  # 计算经由当前节点到达该相邻节点的G值
                    previous_safe_g_value = openlist[[x[0] for x in openlist].index(neighbor)][2]
                    if g_value < previous_safe_g_value:
                        #判断若经由当前节点到达该相邻节点的G值是否小于原来保存的G值, 若小于, 则将该相邻节点的父节点设为当前节点, 并重新设置该相邻节点的G和F值
                        h_value = getconflict(neighbor)
                        g_value = 10 + pow(max(moving_cost), 2) + parent_g_value
                        openlist[[x[0] for x in openlist].index(neighbor)] = \
                            [neighbor, h_value, g_value, f_value, current_status]
                    if h_value == 0:
                        final_f_value = f_value
                        final_status = neighbor
                        break
    # 已确定的当前状态，在第一步时，初始为空

    success_time += 1
    all_count = len(openlist) + len(closelist)


if __name__ == '__main__':
    import datetime
    import random
    import numpy as np
    import operator

    all_count = 0;
    success_time = 0;
    n = int(input("Enter number of queens: "));
    final_status = [];
    init_status = [];
    final_f_value = 0;
    effective_path = [];

    success_time = 0;
    test_num = 1;

    time_start = datetime.datetime.now();
    for run in range(test_num):
        a_star();
    time_stop = datetime.datetime.now();

    print("The Start State")
    prettyprint(init_status)
    print("Number of nodes expanded: %d" % all_count)
    print("Effective Branching Factor: %d" % (all_count / len(effective_path)))
    print("Time solve the puzzle", (time_stop - time_start) / test_num)
    print("Search Cost(F_value): %d" % final_f_value)
    print("Sequence of moves (reverse order)",effective_path)

