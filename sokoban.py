import sys
import collections
import heapq
import numpy as np
import time
from array import array
from collections import deque


class PriorityQueue:

    def  __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        ele = (priority, self.Count, item)
        heapq.heappush(self.Heap, ele)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

def t_game_state(grid):
    grid = [x.replace('\n','') for x in grid]
    grid = [','.join(grid[i]) for i in range(len(grid))]
    grid = [x.split(',') for x in grid]
    max_col = max([len(x) for x in grid])
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == ' ': grid[r][c] = 0
            elif grid[r][c] == '#': grid[r][c] = 1
            elif grid[r][c] == '&': grid[r][c] = 2
            elif grid[r][c] == 'B': grid[r][c] = 3
            elif grid[r][c] == '.': grid[r][c] = 4
            elif grid[r][c] == 'X': grid[r][c] = 5
        col_num = len(grid[r])
        if col_num < max_col:
            grid[r].extend([1 for _ in range(max_col-col_num)])
    return np.array(grid)

def Pos_Player(state):

    return tuple(np.argwhere(state == 2)[0])

def Pos_Box(state):

    return tuple(tuple(x) for x in np.argwhere((state == 3) | (state == 5)))

def Pos_Wall(state):

    return tuple(tuple(x) for x in np.argwhere(state == 1))

def Pos_Goal(state):

    return tuple(tuple(x) for x in np.argwhere((state == 4) | (state == 5)))

def is_End(box_pos):

    return sorted(box_pos) == sorted(posGoals)

def is_legal_act(action, player_pos, box_pos):


    x_player, y_player = player_pos

    if action[-1].isupper():
        x1, y1 = x_player + 2 * action[0], y_player + 2 * action[1]
    else:
        x1, y1 = x_player + action[0], y_player + action[1]
    return (x1, y1) not in box_pos + pos_walls

def legal_action(player_pos, box_pos):

    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    x_player, y_player = player_pos
    legalmove = []
    for action in allActions:
        x1, y1 = x_player + action[0], y_player + action[1]
        if (x1, y1) in box_pos:
            action.pop(2)
        else:
            action.pop(3)
        if is_legal_act(action, player_pos, box_pos):
            legalmove.append(action)
        else:
            continue
    return tuple(tuple(x) for x in legalmove)

def update_state(player_pos, box_pos, action):

    x_player, y_player = player_pos
    new_pos_player = [x_player + action[0], y_player + action[1]]
    box_pos = [list(x) for x in box_pos]
    if action[-1].isupper():
        box_pos.remove(new_pos_player)
        box_pos.append([x_player + 2 * action[0], y_player + 2 * action[1]])
    box_pos = tuple(tuple(x) for x in box_pos)
    new_pos_player = tuple(new_pos_player)
    return new_pos_player, box_pos

def is_failed(box_pos):

    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in box_pos:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                new_board = [board[i] for i in pattern]
                if new_board[1] in pos_walls and new_board[5] in pos_walls: return True
                elif new_board[1] in box_pos and new_board[2] in pos_walls and new_board[5] in pos_walls: return True
                elif new_board[1] in box_pos and new_board[2] in pos_walls and new_board[5] in box_pos: return True
                elif new_board[1] in box_pos and new_board[2] in box_pos and new_board[5] in box_pos: return True
                elif new_board[1] in box_pos and new_board[6] in box_pos and new_board[2] in pos_walls and new_board[3] in pos_walls and new_board[8] in pos_walls: return True
    return False



def bFS():

    lol = Pos_Box(gameState)
    begin_player = Pos_Player(gameState)

    startState = (begin_player, lol)
    checkingthevaluesf = collections.deque([[startState]])
    a = collections.deque([[0]])
    explored_set = set()
    while checkingthevaluesf:
        node = checkingthevaluesf.popleft()
        node_action = a.popleft()
        if is_End(node[-1][-1]):
            print(','.join(node_action[1:]).replace(',',''))
            break
        if node[-1] not in explored_set:
            explored_set.add(node[-1])
            for action in legal_action(node[-1][0], node[-1][1]):
                new_pos_player, new_pos_box = update_state(node[-1][0], node[-1][1], action)
                if is_failed(new_pos_box):
                    continue
                checkingthevaluesf.append(node + [(new_pos_player, new_pos_box)])
                a.append(node_action + [action[-1]])


def dFS():

    begin_box = Pos_Box(gameState)
    begin_player = Pos_Player(gameState)

    start_state = (begin_player, begin_box)
    front = collections.deque([[start_state]])
    explored_set = set()
    a = [[0]]
    while front:
        node = front.pop()
        node_action = a.pop()
        if is_End(node[-1][-1]):
            print(','.join(node_action[1:]).replace(',',''))
            break
        if node[-1] not in explored_set:
            explored_set.add(node[-1])
            for action in legal_action(node[-1][0], node[-1][1]):
                new_pos_player, new_pos_box = update_state(node[-1][0], node[-1][1], action)
                if is_failed(new_pos_box):
                    continue
                front.append(node + [(new_pos_player, new_pos_box)])
                a.append(node_action + [action[-1]])


def is_solved(data):
    for i in range(len(data)):
        if (sdata[i] == '.') != (data[i] == '*'):
            return False
    return True


def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('sokobanLevels/'+options.sokobanLevels,"r") as f:
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args


if __name__ == '__main__':
    print("Hello!!! It's loading.... Be patient")

    grid, method = readCommand(sys.argv[1:]).values()
    gameState = t_game_state(grid)
    pos_walls = Pos_Wall(gameState)
    posGoals = Pos_Goal(gameState)

    if method == 'bfs':
        bFS()

    if method == 'dfs':
        dFS()

    else:
        raise ValueError('Invalid method.')
