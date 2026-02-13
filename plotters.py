import Mandl

import math
from collections import defaultdict

import numpy as np
import networkx as nx
import geopandas as gpd
import contextily as ctx
import osmnx as ox
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


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
        
    def plot_colored_mandl(self, values, cmap='viridis', font_color='white', vmin = 1.0, vmax = 2.0):        
        _, ax = plt.subplots()
        
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
        sm.set_array([])  # обязательно, иначе будет warning
        plt.colorbar(sm, ax=ax)
        plt.show()
        

    def plot_od_matrix(self, cmap=plt.cm.viridis, figsize=(8, 8)):  # type: ignore
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.network.od_matrix,
            annot=True,       # Показывать числа в ячейках
            fmt="d",          # Целочисленный формат
            cmap=cmap,        # Цветовая схема
            linewidths=0.5,   # Разделительные линии
            square=True,      # Квадратные ячейки
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

        # масштаб толщины
        def scale_width(d):
            if d_min == d_max:
                return (min_width + max_width) / 2
            return min_width + (d - d_min) / (d_max - d_min) * (max_width - min_width)

        widths = [scale_width(d) for d in demands]

        order = np.argsort(demands)

        lines = [lines[i] for i in order]
        widths = [widths[i] for i in order]
        demands = demands[order]

        # отрисовка
        fig, ax = plt.subplots(figsize=(8, 6))

        # базовый граф
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

        # OD-линии
        lc = LineCollection(
            lines,
            linewidths=widths,
            array=demands,
            cmap=cmap,
            alpha=1
        )

        ax.add_collection(lc)

        # colorbar
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

    def plot_network(self, network_routes, title=None, figsize=(10, 8), with_node_labels=False, cmap = plt.get_cmap("Set3")):
        plt.figure(figsize=figsize)

        # базовый граф
        nx.draw(
            self.graph, self.pos,
            node_size=200,
            node_color="lightgray",
            edge_color="lightgray",
            with_labels=with_node_labels
        )

        colors = [cmap(i) for i in range(len(network_routes))]

        # считаем, сколько раз каждое ребро используется
        edge_usage = defaultdict(list)

        for i, route in enumerate(network_routes):
            nodes = route['path']
            for u, v in [(int(nodes[i]), int(nodes[i+1])) for i in range(0, len(nodes) - 1)]:
                edge = tuple(sorted((u, v)))
                edge_usage[edge].append(i)

        # рисуем маршруты
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

        # вершины маршрутов
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

    def plot_routes(self, routes, route_width=3, cell_size=5, with_node_labels=False, cmap = plt.get_cmap("tab20")):
        n = len(routes)

        if n == 1:
            print(1)

        # размеры сетки
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(
            cell_size * cols, cell_size * rows))
        axes = axes.flatten()

        colors = [cmap(i) for i in range(len(routes))]

        for i, route in enumerate(routes):
            ax = axes[i]
            ax.set_title(f"Route {i + 1}")

            path = route["path"]

            # 1. базовый граф
            nx.draw(
                self.graph,
                self.pos,
                ax=ax,
                node_size=30,
                node_color="lightgray",
                edge_color="lightgray",
                with_labels=with_node_labels
            )

            # 2. рёбра маршрута
            route_edges = list(zip(path[:-1], path[1:]))

            # 3. выделение маршрута
            nx.draw_networkx_nodes(
                self.graph,
                self.pos,
                nodelist=path,
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

        # скрываем пустые оси
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
                edge = tuple(sorted((u, v)))  # неориентированный граф
                edge_usage[edge] += 1

        # 2. базовая отрисовка графа
        fig, ax = plt.subplots(figsize=(8, 6))

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

        # 3. нормализация значений для визуализации
        counts = np.array(list(edge_usage.values()))
        min_c, max_c = counts.min(), counts.max()

        def scale_width(c):
            if min_c == max_c:
                return (min_width + max_width) / 2
            return min_width + (c - min_c) / (max_c - min_c) * (max_width - min_width)

        # 4. данные для рёбер маршрутов
        used_edges = list(edge_usage.keys())
        widths = [scale_width(edge_usage[e]) for e in used_edges]
        colors = [edge_usage[e] for e in used_edges]

        # 5. отрисовка используемых рёбер поверх
        edges_drawn = nx.draw_networkx_edges(  
            self.graph, self.pos,
            edgelist=used_edges,
            width=widths,
            edge_color=colors,  # type: ignore[arg-type]
            edge_cmap=cmap,
            ax=ax
        )

        # 6. colorbar
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


class CityPlotter():
    def plot_boundary(self, gdf):
        gdf.to_crs(epsg=3857).plot(figsize=(8, 8),
                                   #    edgecolor='red',
                                   #    color='none',
                                   linewidth=3)

    def plot_boundary_on_map(self, gdf):
        ax = (gdf.to_crs(epsg=3857)
              .plot(figsize=(8, 8),
                    edgecolor='red',
                    color='none',
                    linewidth=3))
        ctx.add_basemap(ax)

    def plot_streets_graph(self, graph, ax=None, figsize=(8,8)):
        nodes, edges = ox.graph_to_gdfs(graph)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        edges.plot(ax=ax, linewidth=0.5, color="gray")
        nodes.plot(ax=ax, color="blue", markersize=2)

    def plot_streets_graph_on_map(self, graph, boundary_gdf=None, ax=None, figsize=(8,8)):
        nodes, edges = ox.graph_to_gdfs(graph)
        nodes = nodes.to_crs(epsg=3857)
        edges = edges.to_crs(epsg=3857)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if boundary_gdf is not None:
            boundary_gdf.to_crs(epsg=3857).plot(ax=ax,
                                                figsize=(8, 8),
                                                edgecolor='red',
                                                color='none',
                                                linewidth=3)

        edges.plot(ax=ax, linewidth=0.5, color="gray")
        nodes.plot(ax=ax, color="blue", markersize=2)

        ctx.add_basemap(ax)
        ax.set_axis_off()
        plt.show()
