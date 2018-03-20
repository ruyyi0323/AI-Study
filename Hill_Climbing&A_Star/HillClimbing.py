def generate():
    global N;
    current_chessplate = [];
    next_chessplate = [];  # 下一步要走的状态，在第一步时，为初始生成的棋盘状态
    # while len(next_chessplate) < 5:
    #     tempcoordinate = (random.randint(0, 4), random.randint(0, 4))
    # if tempcoordinate not in next_chessplate:
    #     next_chessplate.append(tempcoordinate)
    for row in range(N):
        next_chessplate.append((row, random.randint(0, 4)))
    return next_chessplate, current_chessplate


def hillclimbing():
    global all_count, success_time, all_count, chess_status_count
    chess_status_count = 0;  # 棋盘变更状态次数
    next_chessplate, current_chessplate = generate();
    while current_chessplate != next_chessplate:  # 在下一步棋盘状态改变时（找到了更好的近邻状态）进行下面步骤，反之（当前状态为最佳状态）计算冲突值
        current_chessplate = list(next_chessplate)  # 棋盘由当前状态变更为下一步的状态，在第一步时，将当前状态置为初始生成棋盘的状态
        next_chessplate = neighbormove(next_chessplate)  # 计算近邻状态，找出是否有冲突值更少的棋盘的下一步状态
    if getconflict(current_chessplate) == 0:
        success_time += 1
    else:
        all_count += chess_status_count
        hillclimbing();


def neighbormove(chessplate):
    #寻找有效的棋盘下一个状态（会使状态变得更小的下一步状态）
    global chess_status_count, N
    pos = [(row, col) for row in range(N) for col in range(N)]
    random.shuffle(pos)
    # rowlist = [row for (row,i) in chessplate]
    # collist = [col for (i,col) in chessplate]
    for tempcoordinate in pos:
        if tempcoordinate[1] != chessplate[tempcoordinate[0]][1]:  # 所选坐标的列与棋盘中同行皇后坐标的列不相等，则为近邻状态
            chess_status_count += 1
            chessplate_copy = list(chessplate)
            chessplate_copy[tempcoordinate[0]] = tempcoordinate
            if getconflict(chessplate_copy) < getconflict(chessplate):
                chessplate = chessplate_copy
                return chessplate
    return chessplate


def getconflict(chessplate):
    num = 0
    # rowlist = [row for (row, i) in chessplate]
    collist = [col for (i, col) in chessplate]
    # 计算列冲突，根据生成的部分，行不会有冲突
    for i in range(len(chessplate)):
        for j in range(i + 1, len(chessplate)):
            if chessplate[i][1] == chessplate[j][1]: num += 1
            if abs(chessplate[i][1] - chessplate[j][1]) == j - i: num += 1
    return num

if __name__ == '__main__':
    import random
    import numpy as np
    import datetime

    all_count = 0
    chess_status_count = 0;
    success_time = 0;
    test_num = 10;
    N = 22;

    all_count = 0;
    success_time = 0;

    time_start = datetime.datetime.now();
    for test in range(test_num):
        hillclimbing();
        all_count += chess_status_count
    time_stop = datetime.datetime.now();

    print("Test number %d, for %d Queens Problem" % (test_num, N))
    print("Average search cost: %.2f chessboards" % (all_count / (test_num + 0.0)))
    print("Percentage of solved problems: %.2f%%" % (success_time / (test_num + 0.0) * 100))
    print("Average time taken", (time_stop - time_start)/test_num)
