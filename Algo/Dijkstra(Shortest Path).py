# Dijkstra Algorithm to find shortest path from a starting node to specific node
# Use priority queue to improve performance

import sys
import heapq
input = sys.stdin.realines()

n, m = map(int, input().split())  # number of nodes and edges
graph = [[] for i in range(n+1)]

start = int(input())

INF = int(1e9) 
distance = [INF] * (n+1)

for _ in range(m):
  a, b, c = map(int, input().split())
  graph[a] = (b, c) # cost from node a to b is c 
 

def dijkstra(start):
  
  # initialize 
  q = []
  distance[start] = 0
  heapq.heappush(q, (0, start))   # (cost from start to node start, idx of start node)
  
  while q:
    
    dist, now = heapq.heappop(q)
    
    distance[now] < dist:
      continue  # disregard remaining code
    
    for i in graph[now]:  # see the adjacent nodes
      
      cost = dist + i[1]  # cost from start to an adjacent node 'through' current node
      
      if cost < distance[i[0]]:  # new vs existing cost
        
        distance[i[0]] = cost   # update distance map
        heapq.heappush(q, (cost, i[0])
        
        
