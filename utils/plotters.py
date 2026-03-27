from shapely.geometry import LineString, Point

import networkx as nx
import geopandas as gpd
import contextily as ctx
import osmnx as ox
import matplotlib.pyplot as plt


DEFAULT_COLORS = ['red', 'darkorange', 'gold', 'yellow', 'lime',
                  'green', 'cyan', 'dodgerblue', 'blue', 'darkviolet',
                  'magenta', 'deeppink', 'coral', 'grey', 'black',
                  'purple', 'maroon', 'saddlebrown', 'olive', 'teal',
                  'salmon', 'peru', 'tan', 'palegreen', 'aquamarine',
                  'powderblue', 'royalblue', 'indigo', 'navy', 'plum']

DEFAULT_COLORS_SHORT = ['red', 'darkorange', 'gold', 'lime',
                        'green', 'cyan', 'dodgerblue', 'blue', 'darkviolet',
                        'magenta']


def get_route_geometries(G, route):
    geometries = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data_dict = G.get_edge_data(u, v)
        if edge_data_dict is None:
            continue

        key = list(edge_data_dict.keys())[0]
        geom = edge_data_dict[key].get("geometry")

        if geom is None:
            geom = LineString([
                Point(G.nodes[u]['x'], G.nodes[u]['y']),
                Point(G.nodes[v]['x'], G.nodes[v]['y'])
            ])

        geometries.append(geom)

    return geometries


class CityPlotter():
    def plot_boundary(self, gdf):
        gdf.to_crs(epsg=3857).plot(figsize=(8, 8),
                                   linewidth=3)

    def plot_boundary_on_map(self, gdf, figsize=(8, 8), axis=False):
        ax = (gdf.to_crs(epsg=3857)
              .plot(figsize=figsize,
                    edgecolor='red',
                    color='none',
                    linewidth=3))
        if not axis:
            ax.set_axis_off()
        ctx.add_basemap(ax)
        return ax

    def plot_base_graph(self, graph, ax):
        nodes, edges = ox.graph_to_gdfs(nx.MultiDiGraph(graph))
        nodes = nodes.to_crs(epsg=3857)
        edges = edges.to_crs(epsg=3857)
        edges.plot(ax=ax, linewidth=0.5, color="lightgray")
        nodes.plot(ax=ax, color="blue", markersize=2)

    def plot_streets_graph(self,
                           graph,
                           ax=None,
                           with_basemap=False,
                           boundary_gdf=None
                           ):

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        if boundary_gdf is not None:
            boundary_gdf.to_crs(epsg=3857).plot(
                ax=ax,
                edgecolor='red',
                color='none',
                linewidth=3
            )

        self.plot_base_graph(graph, ax)

        if with_basemap:
            ctx.add_basemap(ax)

        ax.set_axis_off()
        return ax

    def plot_graph_highlight_nodes(self, graph, ax=None, highlight_nodes=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))

        self.plot_base_graph(graph, ax)

        nodes = ox.graph_to_gdfs(nx.MultiDiGraph(graph), edges=False)
        nodes = nodes.to_crs(epsg=3857)

        if highlight_nodes:
            highlight_nodes_gdf = nodes.loc[highlight_nodes]
            highlight_nodes_gdf.plot(
                ax=ax,
                color="red",
                markersize=5,
                zorder=5
            )

    def plot_routes(self,
                    graph,
                    routes,
                    ax=None,
                    colors=DEFAULT_COLORS,
                    show_graph=True
                    ):
        # plot_routes(G, [route])           for a single route
        # plot_routes(G, routes)            for multiple routes
        # plot_routes(G, network.routes)    for network

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        if show_graph:
            self.plot_base_graph(graph, ax)

        G = nx.MultiGraph(graph)

        for i, route in enumerate(routes):
            color = colors[i % len(colors)]

            geoms = get_route_geometries(G, route)
            gpd.GeoSeries(geoms).plot(ax=ax, linewidth=2, color=color)

            coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in route]
            gpd.GeoDataFrame(
                geometry=[Point(xy) for xy in coords]
            ).plot(ax=ax, color=color, markersize=20)

        ax.axis("off")
        return ax

    def plot_network_with_offset(self, graph, network, ax=None, colors=DEFAULT_COLORS, offset_step=20, plot_nodes=True):
        routes = network.routes

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))

        self.plot_base_graph(graph, ax)

        G = nx.MultiGraph(graph)

        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            center = (len(routes) - 1) / 2
            offset = (i - center) * offset_step

            geoms = get_route_geometries(G, route)

            route_edges = []
            for geom in geoms:
                if offset != 0:
                    try:
                        geom = geom.parallel_offset(
                            abs(offset),
                            "right" if offset > 0 else "left"
                        )
                    except Exception:
                        pass

                route_edges.append(geom)

            gpd.GeoSeries(route_edges).plot(ax=ax, linewidth=2, color=color)

            if plot_nodes:
                route_nodes_coords = [(G.nodes[n]['x'], G.nodes[n]['y'])
                                      for n in route]
                route_nodes_gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in route_nodes_coords]
                )
                route_nodes_gdf.plot(ax=ax, color=color, markersize=20)

        ax.axis("off")
