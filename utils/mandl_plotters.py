import Mandl

import math
from collections import defaultdict

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


DEFAULT_COLORS = ['red', 'darkorange', 'gold', 'yellow', 'lime',
          'green', 'cyan', 'dodgerblue', 'blue', 'darkviolet',
          'magenta', 'deeppink', 'coral', 'grey', 'black',
          'purple', 'maroon', 'saddlebrown', 'olive', 'teal',
          'salmon', 'peru', 'tan', 'palegreen', 'aquamarine',
          'powderblue', 'royalblue', 'indigo', 'navy', 'plum']

DEFAULT_COLORS_SHORT = ['red', 'darkorange', 'gold', 'lime',
                'green', 'cyan', 'dodgerblue', 'blue', 'darkviolet',
                'magenta']


class MandlPlotter():
    def __init__(self) -> None:
        self.network = Mandl.MandlNetwork()
        self.graph = self.network.graph
        self.pos = {i: (self.graph.nodes[i]["x"], self.graph.nodes[i]["y"])
                    for i in range(1, len(self.graph.nodes)+1)}

    def plot_mandl_graph(self, color='blue'):
        nx.draw(self.graph, self.pos, with_labels=True,
                node_color=color, font_color='white')

        edge_labels = {(u, v): d['weight']
                       for u, v, d
                       in self.graph.edges(data=True)}

        nx.draw_networkx_edge_labels(self.graph,
                                     self.pos,
                                     edge_labels=edge_labels)
        plt.show()

    def plot_colored_mandl(self, values, cmap='viridis', font_color='white', vmin=1.0, vmax=2.0, figsize=(8, 8)):
        _, ax = plt.subplots(figsize=figsize)

        nx.draw(
            self.graph,
            self.pos,
            ax=ax,
            with_labels=True,
            node_color=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            font_color=font_color
        )

        edge_labels = {
            (u, v): d['weight']
            for u, v, d in self.graph.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            self.graph,
            self.pos,
            ax=ax,
            edge_labels=edge_labels
        )

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)
        plt.show()

    def plot_od_matrix(self, cmap=plt.cm.viridis, figsize=(8, 8)):  # type: ignore
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.network.od_matrix,
            annot=True,
            fmt="d",
            cmap=cmap,
            linewidths=0.5,
            square=True
        )

        plt.xlabel('to')
        plt.ylabel('from')
        plt.show()

    def plot_od_on_graph(self,
                         min_width=0.5,
                         max_width=6,
                         node_color="lightgray",
                         base_edge_color="lightgray",
                         base_edge_width=0.5,
                         cmap=plt.get_cmap("viridis"),
                         symmetric=True
                         ):

        od_matrix = self.network.od_matrix

        if hasattr(od_matrix, "values"):
            values = od_matrix.values
            nodes = od_matrix.index.to_list()
        else:
            values = od_matrix
            nodes = list(self.pos.keys())

        lines = []
        widths = []
        colors = []
        demands = []

        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u == v:
                    continue

                d = values[i, j]

                if symmetric and j <= i:
                    continue
                if u not in self.pos or v not in self.pos:
                    continue

                demands.append(d)
                lines.append([self.pos[u], self.pos[v]])
                colors.append(d)

        if not demands:
            print("No OD demand to visualize")
            return

        demands = np.array(demands)
        d_min, d_max = demands.min(), demands.max()

        def scale_width(d):
            if d_min == d_max:
                return (min_width + max_width) / 2
            return min_width + (d - d_min) / (d_max - d_min) * (max_width - min_width)

        widths = [scale_width(d) for d in demands]

        order = np.argsort(demands)

        lines = [lines[i] for i in order]
        widths = [widths[i] for i in order]
        demands = demands[order]

        _, ax = plt.subplots(figsize=(8, 6))

        nx.draw_networkx_nodes(
            self.graph, self.pos,
            node_size=30,
            node_color=node_color,
            ax=ax
        )

        nx.draw_networkx_edges(
            self.graph, self.pos,
            edge_color=base_edge_color,
            width=base_edge_width,
            ax=ax
        )

        lc = LineCollection(
            lines,
            linewidths=widths,
            array=demands,
            cmap=cmap,
            alpha=1
        )

        ax.add_collection(lc)

        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=d_min, vmax=d_max)  # type: ignore
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="OD demand")

        ax.set_title("OD demand (direct connections)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


class NetworkOnMandlPlotter(MandlPlotter):
    def __init__(self) -> None:
        super().__init__()

    def plot_network(self, network_routes, title=None, figsize=(10, 8), with_node_labels=False, cmap=plt.get_cmap("Set3")):
        plt.figure(figsize=figsize)

        nx.draw(
            self.graph, self.pos,
            node_size=200,
            node_color="lightgray",
            edge_color="lightgray",
            with_labels=with_node_labels
        )

        colors = [cmap(i) for i in range(len(network_routes))]

        edge_usage = defaultdict(list)

        for i, route in enumerate(network_routes):
            nodes = route['path']
            for u, v in [(int(nodes[i]), int(nodes[i+1])) for i in range(0, len(nodes) - 1)]:
                edge = tuple(sorted((u, v)))
                edge_usage[edge].append(i)

        for edge, route_ids in edge_usage.items():
            u, v = edge

            for k, rid in enumerate(route_ids):
                color = colors[rid % len(colors)]

                rad = 0.15 * (k - len(route_ids)/2)
                if len(route_ids) == 1:
                    rad = 0

                nx.draw_networkx_edges(
                    self.graph, self.pos,
                    edgelist=[(u, v)],
                    width=3,
                    edge_color=[color],
                    alpha=0.9,
                    arrows=True,
                    connectionstyle=f"arc3,rad={rad}"
                )

        for i, route in enumerate(network_routes):
            nodes = route['path']
            color = colors[i % len(colors)]
            nx.draw_networkx_nodes(
                self.graph, self.pos,
                nodelist=nodes,
                node_color=[color],
                node_size=300,
                alpha=0.9
            )

        if title:
            plt.title(title)

        plt.axis("off")
        plt.show()

    def plot_routes(self, routes, route_width=3, cell_size=5, with_node_labels=False, cmap=plt.get_cmap("tab20")):
        n = len(routes)

        if n == 1:
            print(1)

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        _, axes = plt.subplots(rows, cols, figsize=(
            cell_size * cols, cell_size * rows))
        axes = axes.flatten()

        colors = [cmap(i) for i in range(len(routes))]

        for i, route in enumerate(routes):
            ax = axes[i]
            ax.set_title(f"Route {i + 1}")

            nx.draw(
                self.graph,
                self.pos,
                ax=ax,
                node_size=30,
                node_color="lightgray",
                edge_color="lightgray",
                with_labels=with_node_labels
            )

            route_edges = list(zip(route[:-1], route[1:]))
            
            nx.draw_networkx_nodes(
                self.graph,
                self.pos,
                nodelist=route,
                node_color=[colors[i % len(colors)]],
                node_size=60,
                ax=ax
            )

            nx.draw_networkx_edges(
                self.graph,
                self.pos,
                edgelist=route_edges,
                edge_color=[colors[i % len(colors)]],
                width=route_width,
                ax=ax
            )

            ax.axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_edge_usage(self,
                        routes,
                        min_width=1,
                        max_width=6,
                        cmap=plt.get_cmap("viridis"),
                        base_edge_color="lightgray",
                        base_edge_width=0.5):

        edge_usage = defaultdict(int)

        for route in routes:
            path = route["path"]
            for u, v in zip(path[:-1], path[1:]):
                edge = tuple(sorted((u, v)))
                edge_usage[edge] += 1

        _, ax = plt.subplots(figsize=(8, 6))

        nx.draw_networkx_nodes(
            self.graph, self.pos,
            node_size=30,
            node_color="lightgray",
            ax=ax
        )

        nx.draw_networkx_edges(
            self.graph, self.pos,
            edge_color=base_edge_color,
            width=base_edge_width,
            ax=ax
        )

        if not edge_usage:
            ax.set_title("Graph (no routes)")
            ax.axis("off")
            plt.show()
            return

        counts = np.array(list(edge_usage.values()))
        min_c, max_c = counts.min(), counts.max()

        def scale_width(c):
            if min_c == max_c:
                return (min_width + max_width) / 2
            return min_width + (c - min_c) / (max_c - min_c) * (max_width - min_width)

        used_edges = list(edge_usage.keys())
        widths = [scale_width(edge_usage[e]) for e in used_edges]
        colors = [edge_usage[e] for e in used_edges]

        nx.draw_networkx_edges(
            self.graph, self.pos,
            edgelist=used_edges,
            width=widths,
            edge_color=colors,  # type: ignore[arg-type]
            edge_cmap=cmap,
            ax=ax
        )

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=min_c, vmax=max_c)  # type: ignore
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Number of routes")

        ax.set_title("Edge usage by routes")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
