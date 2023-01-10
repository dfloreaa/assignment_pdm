'''
MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

import numpy as np
from random import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections  as mc
from collections import deque
import time

SAFETY_MARGIN = True

class Line():
    ''' Define line '''
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.p1 = np.array(p1)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn = self.dirn/self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn

class Obstacle():
    ''' Define an obstacle '''
    ''' x & y are the center point'''
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def corners(self, margin=1):
        if SAFETY_MARGIN:
            x1 = self.x - self.width/2 - margin/2
            y1 = self.y - self.height/2 - margin/2

            x2 = self.x + self.width/2 + margin/2
            y2 = self.y - self.height/2 - margin/2

            x3 = self.x + self.width/2 + margin/2
            y3 = self.y + self.height/2 + margin/2

            x4 = self.x - self.width/2 - margin/2
            y4 = self.y + self.height/2 + margin/2
        else:
            x1 = self.x - self.width/2
            y1 = self.y - self.height/2

            x2 = self.x + self.width/2
            y2 = self.y - self.height/2

            x3 = self.x + self.width/2
            y3 = self.y + self.height/2

            x4 = self.x - self.width/2
            y4 = self.y + self.height/2

        return np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

def IntersectLine(line1, line2):
    x1 = line1.p[0]
    y1 = line1.p[1]
    x2 = line1.p1[0]
    y2 = line1.p1[1]

    x3 = line2.p[0]
    y3 = line2.p[1]
    x4 = line2.p1[0]
    y4 = line2.p1[1]

    uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))

    if (uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1):
        return True
    else:
        return False

def Intersect(line, obstacle): 
    ''' Check line-obstacle intersection '''
    corners = obstacle.corners()

    for i in range(len(corners)):
        # For every edge of the obstacle, check if the line intersects that edge
        edge = Line(corners[i], corners[i-1])
        collision = IntersectLine(line, edge)
        if collision:
            break
    return collision




def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def isInObstacle(vex, obstacles, margin=1):
    for obs in obstacles:
        nodeX = vex[0]
        nodeY = vex[1]
        if SAFETY_MARGIN:
            if (obs.x - obs.width/2 - margin/2 <= nodeX and nodeX <= obs.x + obs.width/2 + margin/2 and
            obs.y - obs.height/2 - margin/2 <= nodeY and nodeY <= obs.y + obs.height/2 + margin/2):
                return True
        else:
            if (obs.x - obs.width/2 <= nodeX and nodeX <= obs.x + obs.width/2 and
            obs.y - obs.height/2 <= nodeY and nodeY <= obs.y + obs.height/2):
                return True
    return False


def isThruObstacle(line, obstacles):
    for obstacle in obstacles:
        if Intersect(line, obstacle):
            return True
    return False


def nearest(G, vex, obstacles):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isThruObstacle(line, obstacles):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min (stepSize, length)

    newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
    return newvex


def window(startpos, endpos):
    ''' Define seach window - 2 times of start to end rectangle'''
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height

def isInWindow(pos, winx, winy, width, height):
    ''' Restrict new vertex insides search window'''
    if winx < pos[0] < winx+width and \
        winy < pos[1] < winy+height:
        return True
    else:
        return False


class Graph:
    ''' Define graph '''
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]
    def add_vex(self, pos):
        try: # Check if index exists
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


    def randomPosition(self):
        rx = random()
        ry = random()

        posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        return posx, posy

def RRT(startpos, endpos, obstacles, n_iter, stepSize):
    ''' RRT algorithm '''
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            #print('success')
            # break
    return G

def RRT_star(startpos, endpos, obstacles, n_iter, stepSize, radius = 1):
    ''' RRT star algorithm '''
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        G.distances[newidx] = G.distances[nearidx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            line = Line(vex, newvex)
            if isThruObstacle(line, obstacles):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist
        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

            G.success = True
            #print('success')
            # break
    return G

def dijkstra(G):
    '''
    Dijkstra algorithm for finding shortest path from start position to end.
    '''
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    print(path)
    return list(path)

def plot(G, obstacles, path=None):
    '''
    Plot RRT, obstacles and shortest path
    '''
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    fig, ax = plt.subplots()

    for obstacle in obstacles:
        rect = patches.Rectangle((obstacle.x-obstacle.width/2, obstacle.y - obstacle.height/2), obstacle.width, obstacle.height, color='red')
        ax.add_artist(rect)

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        plot(G, obstacles, path)
        return path

# def gymObstacleToPlot(obstacle_coordinates, obstacle_dimensions):
#     '''
#     obstacle_coordinates = x, y, orientation
#     obstacle_dimensions = width, length, height
#         length >>>>>> width: this is independant of the orientation
#     '''
#     x = obstacle_coordinates[0]
#     y = obstacle_coordinates[1]
#     possible_orientations = 
#     if obstacle_coordinates[2]
#     width = 
#     height =




def pathComputation():
    path = None
    startpos = (-10., -10.)
    endpos = (10., 10.)
    boundaryObstacles = [Obstacle(0, -15, 1, 30), Obstacle(-15, 0, 30, 1), Obstacle(15, 0, 1, 30), Obstacle(0,15, 30,1)]
    obstacles = [Obstacle(15, 15, 1, 2), Obstacle(5, 8, 2, 5), Obstacle(17, 9, 1, 7)] 
    obstacles += boundaryObstacles
    n_iter = 2000
    stepSize = 0.7

    radius = 2 # New nodes will be accepted if they are inside this radius of a neighbouring node


    G = RRT_star(startpos, endpos, obstacles, n_iter, stepSize, radius)
    # G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        t0 = time.time()
        path = dijkstra(G)
        t1 = time.time()
        print("Time spent computing shortest path {}".format(t1-t0))
        plot(G, obstacles, path)
    else:
        plot(G, obstacles)
    return path


if __name__ == '__main__':
    path = None
    startpos = (-13., -13.)
    endpos = (10., 10.)
    boundaryObstacles = [Obstacle(0, -15, 30, 1), Obstacle(-15, 0, 1, 30), Obstacle(15, 0, 1, 30), Obstacle(0,15, 30,1)]
    obstacles = [Obstacle(0, 2, 1, 2), Obstacle(5, 8, 2, 5), Obstacle(-10, 8, 1, 7)] 
    obstacles += boundaryObstacles
    n_iter = 2000
    stepSize = 0.7

    radius = 2 # New nodes will be accepted if they are inside this radius of a neighbouring node


    G = RRT_star(startpos, endpos, obstacles, n_iter, stepSize, radius)
    # G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        t0 = time.time()
        path = dijkstra(G)
        t1 = time.time()
        print("Time spent computing shortest path {}".format(t1-t0))
        plot(G, obstacles, path)
    else:
        plot(G, obstacles)