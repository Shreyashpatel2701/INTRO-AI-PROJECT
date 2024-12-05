import osmnx as ox
import networkx as nx
import tkinter as tk
from tkinter import scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import random
import time
import itertools
import math
import heapq  # Added missing import

def init_map():
    """Fetch a small road network near Boyd Center, Athens, GA."""
    location_point = (33.938, -83.374)  # Latitude, Longitude of Boyd Center
    G = ox.graph_from_point(location_point, dist=700, dist_type='bbox', network_type='drive')
    G = ox.project_graph(G)

    # Add weights to edges based on their length
    for u, v, key, data in G.edges(keys=True, data=True):
        data['weight'] = data.get('length', 1.0)  # Default to 1.0 if 'length' not available
        data['original_weight'] = data['weight']  # Store original weight
        data['cost'] = data['weight'] * random.uniform(0.8, 1.5)  # Add cost factor
        data['time'] = data['weight'] / random.uniform(20, 50)  # Simulate travel time
    return G

class TrafficSimulator:
    """Class to handle traffic simulation."""
    def __init__(self, G, interval=5):
        self.G = G
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update_traffic, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def update_traffic(self):
        while self.running:
            for u, v, key, data in self.G.edges(keys=True, data=True):
                traffic_factor = random.uniform(0.8, 2.0)  # Simulate traffic (weights vary)
                data['weight'] = data['original_weight'] * traffic_factor  # Use original weight
                data['cost'] = data['weight'] * random.uniform(0.8, 3)
                data['time'] = data['weight'] / random.uniform(20, 75)
            time.sleep(self.interval)

def plot_map_with_closures(G, start_node=None, stop_node=None, waypoints=None, path_edges=None, closed_edges=None):
    """Visualize the map, highlighting closed edges."""
    node_colors = []
    edge_colors = ['gray'] * len(G.edges())
    edge_widths = [1] * len(G.edges())
    edge_tuples = list(G.edges(keys=True))

    for node in G.nodes():
        if node == start_node:
            node_colors.append('green')
        elif node == stop_node:
            node_colors.append('blue')
        elif waypoints and node in waypoints:
            node_colors.append('orange')
        else:
            node_colors.append('white')

    if path_edges:
        for u, v in path_edges:
            for key in G[u][v]:
                try:
                    edge_idx = edge_tuples.index((u, v, key))
                    edge_colors[edge_idx] = 'red'
                    edge_widths[edge_idx] = 3
                except ValueError:
                    pass

    if closed_edges:
        for u, v, key in closed_edges:
            try:
                edge_idx = edge_tuples.index((u, v, key))
                edge_colors[edge_idx] = 'cyan'  # Closed edges in cyan
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
        k1 = min(self.g[node], self.rhs[node]) + self.heuristic(self.start, node)
        k2 = min(self.g[node], self.rhs[node])
        return (k1, k2)

    def heuristic(self, node1, node2):
        if self.heuristic_type == 'weight':
            y1, x1 = self.G.nodes[node1]['y'], self.G.nodes[node1]['x']
            y2, x2 = self.G.nodes[node2]['y'], self.G.nodes[node2]['x']
            return math.hypot(x1 - x2, y1 - y2)
        else:
            return 0  # Zero heuristic for time and cost

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min([
                min([
                    edge_data.get(self.heuristic_type, 1.0) + self.g[v]
                    for edge_data in self.G.get_edge_data(u, v).values()
                ])
                for v in self.G.successors(u)
            ], default=float('inf'))
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
                for pred in self.G.predecessors(u):
                    self.update_vertex(pred)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for pred in self.G.predecessors(u):
                    self.update_vertex(pred)

    def get_shortest_path(self):
        path = [self.start]
        node = self.start
        while node != self.goal:
            successors = list(self.G.successors(node))
            if not successors:
                return []  # No path found
            min_cost, min_node = min(
                (
                    min([
                        edge_data.get(self.heuristic_type, 1.0)
                        for edge_data in self.G.get_edge_data(node, v).values()
                    ]) + self.g[v], v
                )
                for v in successors
            )
            node = min_node
            path.append(node)
        return path

class MapApp:
    def __init__(self, root, G):
        self.root = root
        self.G = G
        self.start_node = None
        self.stop_node = None
        self.waypoints = []
        self.closed_edges = set()
        self.temp_node = None  # Temporary node for road closure selection
        self.heuristic = 'weight'
        self.cid = None  # For event handling
        self.fig, self.ax = plot_map_with_closures(G)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.traffic_simulator = TrafficSimulator(self.G)  # Initialize traffic simulator
        self.setup_ui()
        self.start_map_update_loop()  # Start updating the map periodically

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

        self.close_road_button = tk.Button(sidebar, text="Close Road", command=self.enable_road_closure,
                                           bg="orange")
        self.close_road_button.pack(pady=10, padx=10, fill=tk.X)

        self.simulate_traffic_button = tk.Button(sidebar, text="Simulate Traffic", command=self.toggle_traffic_simulation,
                                                 bg="yellow")
        self.simulate_traffic_button.pack(pady=10, padx=10, fill=tk.X)

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
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', self.set_start_node)
        self.log_message("Click on the map to set the start node.")

    def enable_stop_node_selection(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', self.set_stop_node)
        self.log_message("Click on the map to set the stop node.")

    def enable_waypoint_selection(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', self.add_waypoint)
        self.log_message("Click on the map to add a waypoint.")

    def enable_road_closure(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', self.close_road)
        self.log_message("Click on two nodes to close a road (edge).")

    def toggle_traffic_simulation(self):
        if self.traffic_simulator.running:
            self.traffic_simulator.stop()
            self.log_message("Traffic simulation stopped.")
            self.simulate_traffic_button.config(text="Simulate Traffic")
        else:
            self.traffic_simulator.start()
            self.log_message("Traffic simulation started.")
            self.simulate_traffic_button.config(text="Stop Traffic Simulation")

    def start_map_update_loop(self):
        def update():
            if self.traffic_simulator.running:
                self.update_map()
                self.compute_route()
            self.root.after(5000, update)  # Update every 5 seconds
        update()

    def set_start_node(self, event):
        if event.inaxes:
            lon, lat = event.xdata, event.ydata
            self.start_node = ox.nearest_nodes(self.G, lon, lat)
            self.log_message(f"Start node set to {self.start_node}.")
            self.compute_route()
            if self.cid is not None:
                self.canvas.mpl_disconnect(self.cid)
                self.cid = None

    def set_stop_node(self, event):
        if event.inaxes:
            lon, lat = event.xdata, event.ydata
            self.stop_node = ox.nearest_nodes(self.G, lon, lat)
            self.log_message(f"Stop node set to {self.stop_node}.")
            self.compute_route()
            if self.cid is not None:
                self.canvas.mpl_disconnect(self.cid)
                self.cid = None

    def add_waypoint(self, event):
        if event.inaxes:
            lon, lat = event.xdata, event.ydata
            waypoint = ox.nearest_nodes(self.G, lon, lat)
            self.waypoints.append(waypoint)
            self.log_message(f"Waypoint added: {waypoint}.")
            self.compute_route()
            if self.cid is not None:
                self.canvas.mpl_disconnect(self.cid)
                self.cid = None

    def close_road(self, event):
        if event.inaxes:
            lon, lat = event.xdata, event.ydata
            node = ox.nearest_nodes(self.G, lon, lat)
            if not self.temp_node:
                self.temp_node = node
                self.log_message(f"Selected node {node} for road closure.")
            else:
                if self.G.has_edge(self.temp_node, node):
                    edge_keys = list(self.G[self.temp_node][node].keys())
                    for key in edge_keys:
                        self.G.remove_edge(self.temp_node, node, key=key)
                        self.closed_edges.add((self.temp_node, node, key))
                    self.log_message(f"Road between {self.temp_node} and {node} closed.")
                else:
                    self.log_message(f"No edge between {self.temp_node} and {node} to close.")
                self.temp_node = None
                self.compute_route()
            if self.cid is not None:
                self.canvas.mpl_disconnect(self.cid)
                self.cid = None

    def compute_route(self):
        if not self.start_node or not self.stop_node:
            self.log_message("Both start and stop nodes must be set.")
            return

        heuristic_map = {
            "Shortest Distance": "weight",
            "Least Cost": "cost",
            "Shortest Time": "time"
        }
        self.heuristic = heuristic_map[self.heuristic_option.get()]

        # Compute route with waypoints using LPA*
        all_nodes = [self.start_node] + self.waypoints + [self.stop_node]
        path = []
        try:
            for i in range(len(all_nodes) - 1):
                lpa = LPAStar(self.G, all_nodes[i], all_nodes[i + 1], heuristic=self.heuristic)
                lpa.compute_shortest_path()
                sub_path = lpa.get_shortest_path()
                if not sub_path:
                    raise Exception(f"No path found between {all_nodes[i]} and {all_nodes[i+1]}")
                if i > 0:  # Avoid duplicating nodes
                    sub_path = sub_path[1:]
                path.extend(sub_path)
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        except Exception as e:
            self.log_message(f"Error computing route: {e}")
            path_edges = []
            path = []

        self.fig, self.ax = plot_map_with_closures(self.G, self.start_node, self.stop_node, self.waypoints, path_edges, self.closed_edges)
        self.canvas.figure = self.fig
        self.canvas.draw()
        self.log_message(f"Route computed: {path}.")

    def clear_map(self):
        self.start_node = None
        self.stop_node = None
        self.waypoints = []
        self.closed_edges = set()
        self.temp_node = None
        self.update_map()
        self.log_message("Map cleared.")

    def update_map(self):
        self.fig, self.ax = plot_map_with_closures(self.G, self.start_node, self.stop_node, self.waypoints, closed_edges=self.closed_edges)
        self.canvas.figure = self.fig
        self.canvas.draw()

def main():
    G = init_map()
    root = tk.Tk()
    root.title("Interactive Map Routing System with LPA*")
    app = MapApp(root, G)
    root.mainloop()

if __name__ == "__main__":
    main()
