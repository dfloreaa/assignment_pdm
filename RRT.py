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
import matplotlib.animation as animation
import matplotlib
import time
import os
import glob

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


def RRT_star(startpos, endpos, obstacles, n_iter, stepSize, radius = 1, make_animation=False):
    ''' RRT star algorithm '''
    fig = plt.figure()
    G = Graph(startpos, endpos)
    n_succes = np.inf
    n_ims = 0

    for i in range(n_iter):
        if G.success == True and i == 1.5*n_succes:
            break
            
        if i%50 == 0:
            print("at", i)
            if make_animation:
                intermediatePlot(G, obstacles, i/50)
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
            if G.success == False:
                n_succes = i
            G.success = True
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

def plot(G, obstacles, environment_id, path=None):
    '''
    Plot RRT, obstacles and shortest path
    '''
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    fig, ax = plt.subplots(figsize=(8,8))

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

    plt.axis([-15, 15, -15, 15])
    ax.margins(0.1)
    plt.savefig('graph{}.png'.format(environment_id))

def intermediatePlot(G, obstacles, i):
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    fig, ax = plt.subplots(figsize=(6,6))

    for obstacle in obstacles:
        rect = patches.Rectangle((obstacle.x-obstacle.width/2, obstacle.y - obstacle.height/2), obstacle.width, obstacle.height, color='red')
        ax.add_artist(rect)

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')
    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    plt.axis([-15, 15, -15, 15])
    ax.margins(0.1)
    plt.savefig('intermediate/intermediate{}.png'.format(int(i)))
    plt.close()



def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        plot(G, obstacles, path)
        return path

def gymObstacleToPlot(obstacle_coordinates, obstacle_dimensions):
    '''
    obstacle_coordinates = x, y, orientation
    obstacle_dimensions = width, length, height
    '''
    x = obstacle_coordinates[0]
    y = obstacle_coordinates[1]
    width = obstacle_dimensions[0]
    height = obstacle_dimensions[1]
    return Obstacle(x, y, width, height)
    # possible_orientations = [0, 0.5*np.pi, np.pi, 1.5*np.pi]
    # if obstacle_coordinates[2] in possible_orientations:
    #     if obstacle_coordinates[2] == 0 or obstacle_coordinates[2] == np.pi:
    
        # else:
        #     width = obstacle_dimensions[1]
        #     height = obstacle_dimensions[0]
    
    # else:
    #     print("Not supported obstacle orientation")



def makeAnimation(environment_id):
    _, _, files = next(os.walk("./intermediate/"))
    file_count = len(files)
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    plt.axis('off')
    #initialization of animation, plot array of zeros 
    def init():
        imobj.set_data(np.zeros((100, 100)))

        return  imobj,

    def animate(i):
        ## Read in picture
        fname = "./intermediate/intermediate%0d.png" % i 

        img = matplotlib.image.imread(fname)[-1::-1]
        imobj.set_data(img)

        return  imobj,


    ## create an AxesImage object
    imobj = ax.imshow(np.zeros((100, 100)), origin='lower', alpha=1.0, zorder=1, aspect=1 )
    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, repeat = True,
                                frames=range(0,file_count), interval=200, blit=True, repeat_delay=1000)
    f = r"./animation{}.gif".format(environment_id) 
    writergif = matplotlib.animation.FFMpegWriter(fps=8) 
    anim.save(f, writer=writergif)

    plt.show()

def pathComputation(obstacles_coordinates, obstacles_dimensions, environment_id,
                    startpos, endpos, n_iter, make_animation = False):
    path = None

    obstacles_coordinates = np.array(obstacles_coordinates)
    obstacles_dimensions = np.array(obstacles_dimensions)
    assert obstacles_coordinates.size == obstacles_dimensions.size

    obstacles = []
    for i in range(len(obstacles_coordinates)):
        obstacles.append(gymObstacleToPlot(obstacles_coordinates[i], obstacles_dimensions[i]))


    stepSize = 0.7

    radius = 2 # New nodes will be accepted if they are inside this radius of a neighbouring node
    
    if make_animation:
        files = glob.glob('./intermediate/*')
        for f in files:
            os.remove(f)

    t0 = time.time()
    G = RRT_star(startpos, endpos, obstacles, n_iter, stepSize, radius, make_animation=make_animation)
    t1 = time.time()

    if make_animation:
        makeAnimation( environment_id)
    

    print("Time spent creating graph: {}".format(t1-t0))
    if G.success:
        path = dijkstra(G)
        plot(G, obstacles, environment_id, path)
    else:
        plot(G, obstacles, environment_id)
    return path
