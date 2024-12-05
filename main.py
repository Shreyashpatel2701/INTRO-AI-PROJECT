import osmnx as ox
import networkx as nx
import tkinter as tk
from tkinter import scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import random
import time
import heapq
import itertools
import math


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


def plot_map(G, start_node=None, stop_node=None, waypoints=None, path_edges=None, lpa_edges=None):
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
        for i, (u, v) in enumerate(path_edges):
            try:
                edge_idx = list(G.edges).index((u, v, 0))
                edge_colors[edge_idx] = 'red' if i < len(path_edges) - 1 else 'purple'  # Differentiate final edges
                edge_widths[edge_idx] = 3
            except ValueError:
                pass

    if lpa_edges:
        for u, v in lpa_edges:
            try:
                edge_idx = list(G.edges).index((u, v, 0))
                edge_colors[edge_idx] = 'cyan'
                edge_widths[edge_idx] = 2
            except ValueError:
                pass

    fig, ax = ox.plot_graph(
        G, node_color=node_colors, node_size=50, edge_color=edge_colors, edge_linewidth=edge_widths,
        figsize=(10, 8), show=False, close=False, bgcolor="black"
    )
    return fig, ax



class LPAStar:
    def __init__(self, graph, start, goal, heuristic='weight'):
        self.G = graph
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic
        self.priority_queue = []
        self.g = {node: float('inf') for node in self.G.nodes}
        self.rhs = {node: float('inf') for node in self.G.nodes}
        self.rhs[self.goal] = 0
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
        self.add_task(self.goal, self.calculate_key(self.goal))

    def add_task(self, node, priority):
        if node in self.entry_finder:
            self.remove_task(node)
        count = next(self.counter)
        entry = [priority, count, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.priority_queue, entry)

    def remove_task(self, node):
        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED

    def pop_task(self):
        while self.priority_queue:
            priority, count, node = heapq.heappop(self.priority_queue)
            if node is not self.REMOVED:
                del self.entry_finder[node]
                return node
        raise KeyError('pop from an empty priority queue')

    def calculate_key(self, node):
        return (min(self.g[node], self.rhs[node]) + self.heuristic(node, self.goal), min(self.g[node], self.rhs[node]))

    def heuristic(self, node1, node2):
        y1, x1 = self.G.nodes[node1]['y'], self.G.nodes[node1]['x']
        y2, x2 = self.G.nodes[node2]['y'], self.G.nodes[node2]['x']
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min([self.G[u][v].get(self.heuristic_type, 1.0) + self.g[v] for v in self.G.neighbors(u)])
        if u in self.entry_finder:
            self.remove_task(u)
        if self.g[u] != self.rhs[u]:
            self.add_task(u, self.calculate_key(u))

    def compute_shortest_path(self):
        while self.priority_queue and (
                self.g[self.start] != self.rhs[self.start] or self.priority_queue[0][0] < self.calculate_key(self.start)
        ):
            u = self.pop_task()
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = float('inf')
            self.update_vertex(u)
            for v in self.G.neighbors(u):
                self.update_vertex(v)

    def get_shortest_path(self):
        path = [self.start]
        node = self.start
        while node != self.goal:
            node = min(
                (self.G[node][v].get(self.heuristic_type, 1.0) + self.g[v], v)
                for v in self.G.neighbors(node)
            )[1]
            path.append(node)
        return path


class MapApp:
    def __init__(self, root, G):
        self.root = root
        self.G = G
        self.start_node = None
        self.stop_node = None
        self.waypoints = []
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

        self.start_button = tk.Button(sidebar, text="Set Start Node", command=self.enable_start_node_selection,
                                      bg="lightblue")
        self.start_button.pack(pady=10, padx=10, fill=tk.X)

        self.stop_button = tk.Button(sidebar, text="Set Stop Node", command=self.enable_stop_node_selection,
                                     bg="lightblue")
        self.stop_button.pack(pady=10, padx=10, fill=tk.X)

        self.waypoint_button = tk.Button(sidebar, text="Add Waypoint", command=self.enable_waypoint_selection,
                                         bg="lightgreen")
        self.waypoint_button.pack(pady=10, padx=10, fill=tk.X)

        self.route_lpa_button = tk.Button(sidebar, text="Find Route with LPA*", command=self.prepare_route_lpa,
                                          bg="purple")
        self.route_lpa_button.pack(pady=10, padx=10, fill=tk.X)

        self.clear_button = tk.Button(sidebar, text="Clear Map", command=self.clear_map, bg="lightcoral")
        self.clear_button.pack(pady=10, padx=10, fill=tk.X)

        self.heuristic_option = ttk.Combobox(sidebar, values=["Shortest Distance", "Least Cost", "Shortest Time"])
        self.heuristic_option.set("Shortest Distance")
        self.heuristic_option.pack(pady=10, padx=10, fill=tk.X)

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

    def add_waypoint(self, event):
        if event.inaxes:
            lon, lat = event.xdata, event.ydata
            waypoint = ox.nearest_nodes(self.G, lon, lat)
            self.waypoints.append(waypoint)
            self.log_message(f"Waypoint added: {waypoint}.")
            self.prepare_route_lpa()

    def prepare_route_lpa(self):
        """Prepare and compute the route using LPA*."""
        if not self.start_node or not self.stop_node:
            self.log_message("Both start and stop nodes must be set.")
            return

        heuristic_map = {
            "Shortest Distance": "weight",
            "Least Cost": "cost",
            "Shortest Time": "time"
        }
        self.heuristic = heuristic_map[self.heuristic_option.get()]

        lpa = LPAStar(self.G, self.start_node, self.stop_node, heuristic=self.heuristic)
        lpa.compute_shortest_path()
        path = lpa.get_shortest_path()
        lpa_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        if self.waypoints:
            path = [self.start_node]
            for waypoint in self.waypoints:
                sub_path = nx.shortest_path(self.G, path[-1], waypoint, weight=self.heuristic)
                path.extend(sub_path[1:])
            final_path = nx.shortest_path(self.G, path[-1], self.stop_node, weight=self.heuristic)
            path.extend(final_path[1:])

        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        self.fig, self.ax = plot_map(self.G, self.start_node, self.stop_node, self.waypoints, path_edges, lpa_edges)
        self.canvas.figure = self.fig
        self.canvas.draw()
        self.log_message(f"LPA* route computed: {path}.")

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
    root.title("Interactive Traffic Routing System with LPA*")
    app = MapApp(root, G)
    root.mainloop()


if __name__ == "__main__":
    main()
