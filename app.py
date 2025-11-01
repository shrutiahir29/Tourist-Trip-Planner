from flask import Flask, render_template, request
import random
import io
import base64
import matplotlib.pyplot as plt
import networkx as nx

app = Flask(__name__)

# ---------- Algorithms ----------

def generate_random_graph(places):
    n = len(places)
    graph = [[0 if i == j else random.randint(10, 100) for j in range(n)] for i in range(n)]
    return graph

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = [False] * n

    for _ in range(n):
        u = min(range(n), key=lambda i: dist[i] if not visited[i] else float('inf'))
        visited[u] = True
        for v in range(n):
            if graph[u][v] and not visited[v]:
                if dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]
    return dist

def kruskal(graph, places):
    edges = []
    n = len(graph)
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((graph[i][j], i, j))
    edges.sort()

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    mst = []
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            mst.append((places[u], places[v], w))
    return mst

def tsp(graph):
    n = len(graph)
    best_cost = float('inf')
    best_path = []

    def backtrack(path, cost, visited):
        nonlocal best_cost, best_path
        if len(path) == n:
            cost += graph[path[-1]][path[0]]
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]
            return

        for nxt in range(n):
            if not visited[nxt]:
                visited[nxt] = True
                backtrack(path + [nxt], cost + graph[path[-1]][nxt], visited)
                visited[nxt] = False

    visited = [False] * n
    visited[0] = True
    backtrack([0], 0, visited)
    return best_path, best_cost

# ---------- Visualization ----------

def visualize(places, graph, path):
    G = nx.Graph()
    for i, place in enumerate(places):
        G.add_node(place)
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            G.add_edge(places[i], places[j], weight=graph[i][j])

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_color="#ffcc80", node_size=1200, edgecolors="#d84315")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="#4e342e")
    nx.draw_networkx_edges(G, pos, edge_color="#ccc", width=1)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(places[i], places[j]): graph[i][j] for i in range(len(graph)) for j in range(i + 1, len(graph))}, font_color="#6d4c41")

    # Highlight the best path in red with thicker lines
    path_edges = [(places[path[i]], places[path[i + 1]]) for i in range(len(path) - 1)] + [(places[path[-1]], places[path[0]])]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.5, edge_color="#ff7043", style="solid")

    plt.title("Optimized Tourist Route", fontsize=14, fontweight="bold", color="#6d4c41")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#fff3e0")
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return image_data

# ---------- Flask Route ----------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        places = [p.strip() for p in request.form["places"].split(",") if p.strip()]
        if len(places) < 3:
            result = {"error": "Please enter at least 3 places."}
        else:
            graph = generate_random_graph(places)
            dijkstra_result = dijkstra(graph, 0)
            mst = kruskal(graph, places)
            best_path, best_cost = tsp(graph)
            image_data = visualize(places, graph, best_path)

            result = {
                "places": places,
                "graph": graph,
                "dijkstra": dijkstra_result,
                "mst": mst,
                "tsp": {"path": [places[i] for i in best_path], "cost": best_cost},
                "image": image_data
            }

    return render_template("index.html", result=result, zip=zip)


if __name__ == "__main__":
    app.run(debug=True)
