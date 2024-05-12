from typing import List
from collections import defaultdict
import heapq
import sys

class Solution:

    # 1. Najbliższego sąsiada
    def nearest_neighbor(self, edges, start_node):
        
        graph = defaultdict(list)

        for x, y, z in edges:
            graph[x].append([y, z])
            graph[y].append([x, z])

        for x in graph.keys():
            graph[x] = sorted(graph[x], key = lambda l: l[1])

        visited = set()
        visited.add(start_node)

        def dfs(node, path):

            if node == start_node and len(path) == len(graph):
                return path

            for nei, _ in graph[node]:
                
                if nei not in visited:
                    visited.add(nei)
                    path.append(nei)
                    dfs(nei, path)

            return path

        return dfs(start_node, [start_node]) + [start_node]


 # 2. Johnsona w wersji wykorzystującej algorytmy Dijkstry oraz Bellmana-Forda
    def johnson(self, edges):
        graph = defaultdict(list)

        for x, y, z in edges:
            graph[x].append((y, z))

        new_node = len(graph)
        for node in graph:
            graph[node].append((new_node, 0))
        graph[new_node] = []

        h_values = self.bellman_ford(graph, new_node)
        if h_values is None:
            return None  

        del graph[new_node]
        for node in graph:
            graph[node] = [edge for edge in graph[node] if edge[0] != new_node]

        for node in graph:
            for i, edge in enumerate(graph[node]):
                graph[node][i] = (edge[0], edge[1] + h_values[node] - h_values[edge[0]])
                
        shortest_paths = []
        for node in graph:
            shortest_paths.append(self.dijkstra(graph, node, h_values))

        return shortest_paths

    def bellman_ford(self, graph, source):
        dist = {node: float('inf') for node in graph}
        dist[source] = 0

        for _ in range(len(graph) - 1):
            for node in graph:
                for neighbor, weight in graph[node]:
                    if dist[node] != float('inf') and dist[node] + weight < dist[neighbor]:
                        dist[neighbor] = dist[node] + weight

        for node in graph:
            for neighbor, weight in graph[node]:
                if dist[node] != float('inf') and dist[node] + weight < dist[neighbor]:
                    return None 

        return dist

    def dijkstra(self, graph, source, h):
        dist = {node: float('inf') for node in graph}
        dist[source] = 0
        queue = [(0, source)]

        while queue:
            current_dist, current_node = heapq.heappop(queue)

            if current_dist > dist[current_node]:
                continue

            for neighbor, weight in graph[current_node]:
                distance = current_dist + weight + h[current_node] - h[neighbor]
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))

        return dist

    # 3. Algorytm Kruskala wyznaczania drzewa spinającego
    def kruskal(self, edges: List[str], n: int):

        par = [i for i in range(n)]
        rank = [1 for i in range(n)]


        def find(n1):
            res = n1
            while par[res]!=res:
                par[res] = par[par[res]]
                res = par[res]

            return res

        def union(n1,n2):

            p1,p2 = find(n1),find(n2)

            if p1 == p2:
                return False
            
            if rank[p2] > rank[p1]:
                par[p1] = p2
                rank[p2] += rank[p1]
            else:
                par[p2] = p1
                rank[p1] += rank[p2]

            return True
        
        mst = []
        res = 0
        
        for x,y,z in sorted(edges, key = lambda l : l[2]):
            if union(x, y):
                mst.append((x, y, z))
                res += z
                if len(mst) == n - 1:
                    break

        return mst, res

    # 4. Algorytm wyznaczanie drzewa spinającego
    def spinal_tree(self, edges, n):

        mst = []
        res = 0

        graph = defaultdict(list)

        for x, y, z in edges:
            graph[x].append([y, z])
            graph[y].append([x, z])

        for x in graph.keys():
            graph[x] = sorted(graph[x], key = lambda l: l[1])

        visited = set()

        def dfs(node):
            nonlocal res
    
            if node not in visited:
                visited.add(node)
                for nei, wei in graph[node]:
                    mst.append((node, nei, wei))
                    res += wei
                    visited.add(nei)
                    dfs(nei)
                    break


        for i in range(n):
            dfs(i)

        return mst, res
    
    # 5. Algorytm kolorowania wierzchołków LF (Largest First)
    def lf_coloring(self, edges, n):
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        degree = {node: len(adj) for node, adj in graph.items()}
        nodes_sorted_by_degree = sorted(degree, key=degree.get, reverse=True)

        color = {}
        
        for node in nodes_sorted_by_degree:
            adjacent_colors = {color[neighbor] for neighbor in graph[node] if neighbor in color}
            for c in range(n):
                if c not in adjacent_colors:
                    color[node] = c
                    break
        
        return color

    # 6. Algorytm kolorowania wierzchołków SL (Smallest Last)
    def sl_coloring(self, edges, n):
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        degree = {node: len(adj) for node, adj in graph.items()}

        heap = [(len(adj), node) for node, adj in graph.items()]
        heapq.heapify(heap)
        
        elimination_order = []
        while heap:
            _, node = heapq.heappop(heap)
            elimination_order.append(node)
            for neighbor in graph[node]:
                if neighbor in degree:
                    degree[neighbor] -= 1
                    heap = [(degree[n], n) for n in degree if n != node]
                    heapq.heapify(heap)
            del degree[node]

        color = {}
        for node in reversed(elimination_order):
            available_colors = {color[neighbor] for neighbor in graph[node] if neighbor in color}
            for c in range(n):
                if c not in available_colors:
                    color[node] = c
                    break
        
        return color

        
# 1.

edges = [(1, 2, 10), (2, 3, 20), (3, 4, 30), (4, 1, 40), (1, 3, 15), (2, 4, 25)]
start_node = 1

sol = Solution()
result = sol.nearest_neighbor(edges, start_node)
print(f"Nearest Neighbour path is {result}")

# 2.
edges = [(0, 1, 4), (1, 2, -2), (2, 0, 1), (1, 3, 2), (3, 4, -1), (4, 1, 1)]
sol = Solution()
result = sol.johnson(edges)
print(result)

# 3.
edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
n = 4

solution = Solution()
mst, total_weight = solution.kruskal(edges, n)

print("Minimum Spanning Tree:")
for edge in mst:
    print(edge)
print("Total Weight of MST:", total_weight)


# 4.
edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]

n = 4

solution = Solution()
mst, total_weight = solution.spinal_tree(edges, n)

print("Minimum Spanning Tree (Spinal Tree):")
for edge in mst:
    print(edge)
print("Total Weight of Spinal Tree:", total_weight)

# 5 i 6
edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D')]
n = 4
sol = Solution()
lf_colors = sol.lf_coloring(edges, n)
sl_colors = sol.sl_coloring(edges, n)

print("LF Coloring:", lf_colors)
print("SL Coloring:", sl_colors)


