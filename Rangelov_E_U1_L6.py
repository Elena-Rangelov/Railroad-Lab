# Name: Elena Rangelov     Date: 10.23.2020

import random, pickle, math, time
from math import pi, acos, sin, cos
from tkinter import *

import PIL

import io, sys
import urllib.request


class HeapPriorityQueue():
    def __init__(self):
        self.queue = ["dummy"]  # we do not use index 0 for easy index calulation
        self.current = 1  # to make this object iterable

    def next(self):  # define what __next__ does
        if self.current >= len(self.queue):
            self.current = 1  # to restart iteration later
            raise StopIteration

        out = self.queue[self.current]
        self.current += 1

        return out

    def __iter__(self):
        return self

    __next__ = next

    def isEmpty(self):
        return len(self.queue) == 1  # b/c index 0 is dummy

    def swap(self, a, b):
        self.queue[a], self.queue[b] = self.queue[b], self.queue[a]

    # Add a value to the heap_pq
    def push(self, value):
        self.queue.append(value)
        self.heapUp(len(self.queue) - 1)
        # write more code here to keep the min-heap property

    # helper method for push
    def heapUp(self, k):
        parent = k // 2
        if parent > 0:
            if self.queue[parent] > self.queue[k]:
                self.swap(parent, k)
                self.heapUp(parent)

    # helper method for reheap and pop
    def heapDown(self, k, size):
        left, right = 2 * k, 2 * k + 1
        if left <= size:
            if right > size:
                min = left
            elif self.queue[left] < self.queue[right]:
                min = left
            else:
                min = right
            if self.queue[min] < self.queue[k]:
                self.swap(min, k)
                self.heapDown(min, size - 1)

    # make the queue as a min-heap
    def reheap(self):
        for i in range(len(self.queue) // 2, 0, -1):
            self.heapDown(i, len(self.queue) - 1)

    # remove the min value (root of the heap)
    # return the removed value
    def pop(self):
        x = self.queue.pop(1)
        self.reheap()
        return x  # change this

    # remove a value at the given index (assume index 0 is the root)
    # return the removed value
    def remove(self, index):
        x = self.queue.pop(index)
        self.reheap()
        return x  # change this


def calc_edge_cost(y1, x1, y2, x2):
    #
    # y1 = lat1, x1 = long1
    # y2 = lat2, x2 = long2
    # all assumed to be in decimal degrees

    # if (and only if) the input is strings
    # use the following conversions

    y1 = float(y1)
    x1 = float(x1)
    y2 = float(y2)
    x2 = float(x2)
    #
    R = 3958.76  # miles = 6371 km
    #
    y1 *= pi / 180.0
    x1 *= pi / 180.0
    y2 *= pi / 180.0
    x2 *= pi / 180.0
    #
    # approximate great circle distance with law of cosines
    #
    return acos(sin(y1) * sin(y2) + cos(y1) * cos(y2) * cos(x2 - x1)) * R
    #


# NodeLocations, NodeToCity, CityToNode, Neighbors, EdgeCost
# Node: (lat, long) or (y, x), node: city, city: node, node: neighbors, (n1, n2): cost
def make_graph(nodes="rrNodes.txt", node_city="rrNodeCity.txt", edges="rrEdges.txt"):
    nodeLoc, nodeToCity, cityToNode, neighbors, edgeCost = {}, {}, {}, {}, {}
    map = {}  # have screen coordinate for each node location

    with open(nodes, "r") as infile:

        for line in infile:
            words = line.strip().split(" ")
            nodeLoc[words[0]] = (words[1], words[2])

    with open(node_city, "r") as infile:
        for line in infile:
            words = line.strip().split(" ")
            n1, n2 = words[0], ' '.join(words[1:])
            nodeToCity[n1] = n2
            cityToNode[n2] = n1

    with open(edges, "r") as infile:
        for line in infile:
            words = line.strip().split(" ")
            n1 = words[0]
            n2 = words[1]
            if n1 in neighbors:
                neighbors[n1].add(n2)
            else:
                neighbors[n1] = {n2}
            if n2 in neighbors:
                neighbors[n2].add(n1)
            else:
                neighbors[n2] = {n1}
            cost = calc_edge_cost(*nodeLoc[n1], *nodeLoc[n2])
            edgeCost[(n1, n2)] = cost
            edgeCost[(n2, n1)] = cost

    infile.close()

    # ''' Un-comment after you fill the nodeLoc dictionary.
    for node in nodeLoc:  # checks each
        lat = float(nodeLoc[node][0])  # gets latitude
        long = float(nodeLoc[node][1])  # gets long
        modlat = (lat - 10) / 60  # scales to 0-1
        modlong = (long + 130) / 70  # scales to 0-1
        map[node] = [modlat * 800, modlong * 1200]  # scales to fit 800 1200
    # '''

    return [nodeLoc, nodeToCity, cityToNode, neighbors, edgeCost, map]


# Retuen the direct distance from node1 to node2
# Use calc_edge_cost function.
def dist_heuristic(n1, n2, graph):

    return calc_edge_cost(*graph[0][n1], *graph[0][n2])


# Create a city path. 
# Visit each node in the path. If the node has the city name, add the city name to the path.
# Example: ['Charlotte', 'Hermosillo', 'Mexicali', 'Los Angeles']
def display_path(path, graph):

    p = []
    for node in path:
        if node in graph[1]:
            p += [graph[1][node]]
    print(p)


# Using the explored, make a path by climbing up to "s"
# This method may be used in your BFS and Bi-BFS algorithms.
def generate_path(state, explored, graph):
    path = [state]
    cost = 0

    while explored[state] != "s":
        t = explored[state]
        path.append(t)
        cost += graph[4][(state, t)]
        state = t

    return path[::-1], cost


def drawLine(canvas, y1, x1, y2, x2, col):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    canvas.create_line(x1, 800 - y1, x2, 800 - y2, fill=col)


# Draw the final shortest path.
# Use drawLine function.
def draw_final_path(ROOT, canvas, path, graph, col='red'):
    for p in range(len(path) - 1):
        drawLine(canvas, *graph[5][path[p]], *graph[5][path[p + 1]], col)

    ROOT.update()


def draw_all_edges(ROOT, canvas, graph):
    ROOT.geometry("1200x800")  # sets geometry
    canvas.pack(fill=BOTH, expand=1)  # sets fill expand
    for n1, n2 in graph[4]:  # graph[4] keys are edge set
        drawLine(canvas, *graph[5][n1], *graph[5][n2], 'white')  # graph[5] is map dict
    ROOT.update()


def bfs(start, goal, graph, col):
    ROOT = Tk()  # creates new tkinter
    ROOT.title("BFS")
    canvas = Canvas(ROOT, background='black')  # sets background
    draw_all_edges(ROOT, canvas, graph)

    counter = 0
    frontier, explored = [], {start: "s"}
    frontier.append(start)
    while frontier:
        s = frontier.pop(0)
        if s == goal:
            path, cost = generate_path(s, explored, graph)
            draw_final_path(ROOT, canvas, path, graph)
            return path, cost
        for a in graph[3][s]:  # graph[3] is neighbors
            if a not in explored:
                explored[a] = s
                frontier.append(a)
                drawLine(canvas, *graph[5][s], *graph[5][a], col)
        counter += 1
        if counter % 1000 == 0: ROOT.update()
    return None


def bi_bfs(start, goal, graph, col):
    '''The idea of bi-directional search is to run two simultaneous searches--
       one forward from the initial state and the other backward from the goal--
       hoping that the two searches meet in the middle.
       '''
    ROOT = Tk()  # creates new tkinter
    ROOT.title("BFS")
    canvas = Canvas(ROOT, background='black')  # sets background
    draw_all_edges(ROOT, canvas, graph)
    counter = 0

    if start == goal: return []

    explored_start = {start: "s"}
    explored_end = {goal: "s"}
    q_start = [start]
    q_end = [goal]

    while q_start and q_end:
        s = q_start.pop(0)
        e = q_end.pop(0)
        if e in explored_start:
            path1, cost1 = generate_path(e, explored_start, graph)
            path2, cost2 = generate_path(e, explored_end, graph)
            path2.reverse()
            path = path1 + path2
            cost = cost1 + cost2
            draw_final_path(ROOT, canvas, path, graph)
            return path, cost
        if s in explored_end:
            path1, cost1 = generate_path(s, explored_start, graph)
            path2, cost2 = generate_path(s, explored_end, graph)
            path2.reverse()
            path = path1 + path2
            cost = cost1 + cost2
            draw_final_path(ROOT, canvas, path, graph)
            return path, cost
        for child in graph[3][s]:
            if not child in explored_start:
                q_start.append(child)
                explored_start[child] = s
                drawLine(canvas, *graph[5][s], *graph[5][child], col)
        for child in graph[3][e]:
            if not child in explored_end:
                q_end.append(child)
                explored_end[child] = e
                drawLine(canvas, *graph[5][e], *graph[5][child], col)
        counter += 1
        if counter % 1000 == 0: ROOT.update()

    return None

def a_star(start, goal, graph, col, heuristic=dist_heuristic):

    ROOT = Tk()  # creates new tkinter
    ROOT.title("BFS")
    canvas = Canvas(ROOT, background='black')  # sets background
    draw_all_edges(ROOT, canvas, graph)
    counter = 0

    h = heuristic(start, goal, graph)
    frontier = HeapPriorityQueue()
    explored = {start: h}
    if start == goal: return []
    frontier.push((h, start, [start], 0))
    while not frontier.isEmpty():
        q = frontier.pop()
        if q[1] == goal:
            draw_final_path(ROOT, canvas, q[2], graph)
            return q[2], q[3]
        for i in graph[3][q[1]]:
            h = heuristic(i, goal, graph)
            g = q[3] + graph[4][(q[1], i)]
            if i not in explored or explored[i] > g + h:
                frontier.push((h + g, i, q[2] + [i], g))
                explored[i] = h + g
                drawLine(canvas, *graph[5][i], *graph[5][q[1]], col)
        counter += 1
        if counter % 1000 == 0: ROOT.update()

    return None


def bi_a_star(start, goal, graph, col, heuristic=dist_heuristic):
    # Your code goes here

    return None


def tri_directional(city1, city2, city3, graph, col, heuristic=dist_heuristic):
    # Your code goes here

    return None


def main():
    start, goal = input("Start city: "), input("Goal city: ")
    # third = input("Third city for tri-directional: ")
    graph = make_graph("rrNodes.txt", "rrNodeCity.txt", "rrEdges.txt")  # Task 1

    cur_time = time.time()
    path, cost = bfs(graph[2][start], graph[2][goal], graph, 'yellow')  # graph[2] is city to node
    print("Length of the path: ", len(path))
    if path != None:
        display_path(path, graph)
    else:
        print("No Path Found.")
    print('BFS Path Cost:', cost)
    print('BFS duration:', (time.time() - cur_time))
    print()

    cur_time = time.time()
    path, cost = bi_bfs(graph[2][start], graph[2][goal], graph, 'green')
    print("Length of the path: ", len(path))
    if path != None:
        display_path(path, graph)
    else:
        print("No Path Found.")
    print('Bi-BFS Path Cost:', cost)
    print('Bi-BFS duration:', (time.time() - cur_time))
    print()

    cur_time = time.time()
    path, cost = a_star(graph[2][start], graph[2][goal], graph, 'blue')
    print("Length of the path: ", len(path))
    if path != None:
        display_path(path, graph)
    else:
        print("No Path Found.")
    print('A star Path Cost:', cost)
    print('A star duration:', (time.time() - cur_time))
    print()

    """
   cur_time = time.time()
   path, cost = bi_bfs(graph[2][start], graph[2][goal], graph, 'green')
   if path != None: display_path(path, graph)
   else: print ("No Path Found.")
   print ('Bi-BFS Path Cost:', cost)
   print ('Bi-BFS duration:', (time.time() - cur_time))
   print ()

   cur_time = time.time()
   path, cost = a_star(graph[2][start], graph[2][goal], graph, 'blue')
   if path != None: display_path(path, graph)
   else: print ("No Path Found.")
   print ('A star Path Cost:', cost)
   print ('A star duration:', (time.time() - cur_time))
   print ()

   cur_time = time.time()
   path, cost = bi_a_star(graph[2][start], graph[2][goal], graph, 'orange', ROOT, canvas)
   if path != None: display_path(path, graph)
   else: print ("No Path Found.")
   print ('Bi-A star Path Cost:', cost)
   print ("Bi-A star duration: ", (time.time() - cur_time))
   print ()

   print ("Tri-Search of ({}, {}, {})".format(start, goal, third))
   cur_time = time.time()
   path, cost = tri_directional(graph[2][start], graph[2][goal], graph[2][third], graph, 'pink', ROOT, canvas)
   if path != None: display_path(path, graph)
   else: print ("No Path Found.")
   print ('Tri-A star Path Cost:', cost)
   print ("Tri-directional search duration:", (time.time() - cur_time))
   """
    mainloop()  # Let TK windows stay still


if __name__ == '__main__':
    main()
