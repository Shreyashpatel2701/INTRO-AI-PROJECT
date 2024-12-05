import osmnx as ox
import networkx as nx
import tkinter as tk
from tkinter import scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import heapq
import random


def init_map():
    """Fetch a small road network near Boyd Center, Athens, GA."""
    location_point = (33.938, -83.374)  # Latitude, Longitude of Boyd Center
    G = ox.graph_from_point(location_point, dist=700, dist_type='bbox', network_type='drive')
    G = ox.project_graph(G)

    # Add weights to edges based on their length
    for u, v, key, data in G.edges(keys=True, data=True):
        data['weight'] = data.get('length', 1.0)  # Default to 1.0 if 'length' not available
        data['cost'] = data['weight'] * random.uniform(0.8, 1.5)  # Add cost factor
        data['time'] = data['weight'] / random.uniform(20, 50)  # Simulate travel time
    return G


def plot_map(G, start_node=None, stop_node=None, waypoints=None, path_edges=None):
    """Visualize the map with start, stop, waypoints, and optional path."""
    node_colors = []
    edge_colors = ['gray'] * len(G.edges())
    edge_widths = [1] * len(G.edges())

    for node in G.nodes():
        if node == start_node:
            node_colors.append('green')  # Highlight the start node
        elif node == stop_node:
            node_colors.append('blue')  # Highlight the stop node
        elif waypoints and node in waypoints:
            node_colors.append('orange')  # Highlight waypoints
        else:
            node_colors.append('white')  # Default node color

    if path_edges:
        for u, v in path_edges:
            try:
                edge_idx = list(G.edges).index((u, v, 0))
                edge_colors[edge_idx] = 'red'
                edge_widths[edge_idx] = 3
            except ValueError:
                pass

    fig, ax = ox.plot_graph(
        G, node_color=node_colors, node_size=50, edge_color=edge_colors, edge_linewidth=edge_widths,
        figsize=(10, 8), show=False, close=False, bgcolor="black"
    )
    return fig, ax


def bfs(graph, start, goal):
    """Breadth-First Search to find a path."""
    queue = [(start, [start])]
    visited = set()
    while queue:
        node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None


def dfs(graph, start, goal):
    """Depth-First Search to find a path."""
    stack = [(start, [start])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None


def dijkstra(graph, start, goal, weight="weight"):
    """Dijkstra's algorithm to find the shortest path."""
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == goal:
            return path
        for neighbor, data in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + data.get(weight, 1.0), neighbor, path))
    return None


class MapApp:
    def __init__(self, root, G):
        self.root = root
        self.G = G
        self.start_node = None
        self.stop_node = None
        self.waypoints = []
        self.algorithm = 'Dijkstra'  # Default algorithm
        self.heuristic = 'weight'
        self.fig, self.ax = plot_map(G)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.setup_ui()

    def setup_ui(self):
        sidebar = tk.Frame(self.root, width=300, bg="#f0f0f0")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.start_button = tk.Button(sidebar, text="Set Start Node", command=self.enable_start_node_selection, bg="lightblue")
        self.start_button.pack(pady=10, padx=10, fill=tk.X)

        self.stop_button = tk.Button(sidebar, text="Set Stop Node", command=self.enable_stop_node_selection, bg="lightblue")
        self.stop_button.pack(pady=10, padx=10, fill=tk.X)

        self.waypoint_button = tk.Button(sidebar, text="Add Waypoint", command=self.enable_waypoint_selection, bg="lightgreen")
        self.waypoint_button.pack(pady=10, padx=10, fill=tk.X)

        self.algorithm_option = ttk.Combobox(sidebar, values=["Dijkstra", "BFS", "DFS"])
        self.algorithm_option.set("Dijkstra")
        self.algorithm_option.pack(pady=10, padx=10, fill=tk.X)

        self.heuristic_option = ttk.Combobox(sidebar, values=["Shortest Distance", "Least Cost", "Shortest Time"])
        self.heuristic_option.set("Shortest Distance")
        self.heuristic_option.pack(pady=10, padx=10, fill=tk.X)

        self.route_button = tk.Button(sidebar, text="Find Route", command=self.find_route, bg="purple")
        self.route_button.pack(pady=10, padx=10, fill=tk.X)

        self.clear_button = tk.Button(sidebar, text="Clear Map", command=self.clear_map, bg="lightcoral")
        self.clear_button.pack(pady=10, padx=10, fill=tk.X)

        self.log_text = scrolledtext.ScrolledText(sidebar, wrap=tk.WORD, width=30, height=15)
        self.log_text.pack(pady=10, padx=10)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def enable_start_node_selection(self):
        self.canvas.mpl_connect('button_press_event', self.set_start_node)
        self.log_message("Click on the map to set the start node.")

    def enable_stop_node_selection(self):
        self.canvas.mpl_connect('button_press_event', self.set_stop_node)
        self.log_message("Click on the map to set the stop node.")

    def enable_waypoint_selection(self):
        self.canvas.mpl_connect('button_press_event', self.add_waypoint)
        self.log_message("Click on the map to add a waypoint.")

    def set_start_node(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            # Convert clicked coordinates to the nearest node in the graph
            self.start_node = ox.distance.nearest_nodes(self.G, X=x, Y=y, return_dist=False)
            self.log_message(f"Start node set to {self.start_node}.")
            self.update_map()

    def set_stop_node(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            # Convert clicked coordinates to the nearest node in the graph
            self.stop_node = ox.distance.nearest_nodes(self.G, X=x, Y=y, return_dist=False)
            self.log_message(f"Stop node set to {self.stop_node}.")
            self.update_map()

    def add_waypoint(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            # Convert clicked coordinates to the nearest node in the graph
            waypoint = ox.distance.nearest_nodes(self.G, X=x, Y=y, return_dist=False)
            self.waypoints.append(waypoint)
            self.log_message(f"Waypoint added: {waypoint}.")
            self.update_map()

    def find_route(self):
        if not self.start_node or not self.stop_node:
            self.log_message("Both start and stop nodes must be set.")
            return

        algorithm = self.algorithm_option.get()
        heuristic_map = {"Shortest Distance": "weight", "Least Cost": "cost", "Shortest Time": "time"}
        self.heuristic = heuristic_map[self.heuristic_option.get()]

        path = [self.start_node]  # Start the path with the start node
        nodes_to_visit = self.waypoints + [self.stop_node]  # Waypoints and the final stop node

        for next_node in nodes_to_visit:
            sub_path = None
            if algorithm == "Dijkstra":
                sub_path = dijkstra(self.G, path[-1], next_node, weight=self.heuristic)
            elif algorithm == "BFS":
                sub_path = bfs(self.G, path[-1], next_node)
            elif algorithm == "DFS":
                sub_path = dfs(self.G, path[-1], next_node)

            if sub_path:
                path.extend(sub_path[1:])  # Add sub-path excluding the start of the sub-path to avoid duplication
            else:
                self.log_message(f"No path found to {next_node} using {algorithm}.")
                return

        # Convert the full path to edge pairs for visualization
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        self.fig, self.ax = plot_map(self.G, self.start_node, self.stop_node, self.waypoints, path_edges)
        self.canvas.figure = self.fig
        self.canvas.draw()
        self.log_message(f"{algorithm} route computed: {path}.")

    def clear_map(self):
        self.start_node = None
        self.stop_node = None
        self.waypoints = []
        self.update_map()
        self.log_message("Map cleared.")

    def update_map(self):
        self.fig, self.ax = plot_map(self.G, self.start_node, self.stop_node, self.waypoints)
        self.canvas.figure = self.fig
        self.canvas.draw()


def main():
    G = init_map()
    root = tk.Tk()
    root.title("Interactive Traffic Routing System with Multiple Algorithms")
    app = MapApp(root, G)
    root.mainloop()


if __name__ == "__main__":
    main()
