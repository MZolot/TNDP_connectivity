import pandas as pd
import networkx as nx


class MandlNetwork:
    def __init__(self) -> None:
        self._init_graph()
        self._init_od_matrix()
        self._init_pedestrian_graph()

    def _init_graph(self):
        nodes_df = pd.read_csv("data/mandl/mandl1_nodes.csv")
        edges_df = pd.read_csv("data/mandl/mandl1_links.csv")

        self.graph = nx.Graph()

        for _, row in nodes_df.iterrows():
            node_id = int(row["id"])
            x = row["lon"]
            y = row["lat"]
            self.graph.add_node(node_id, x=x, y=y)

        for _, row in edges_df.iterrows():
            source = row["from"]
            target = row["to"]
            weight = row["travel_time"]
            self.graph.add_edge(source, target, weight=weight)

        # Vertexes positions for visualisation
        self.vertex_positions = {i: (self.graph.nodes[i]["x"], self.graph.nodes[i]["y"]) for i in range(
            1, len(self.graph.nodes)+1)}

    def _init_od_matrix(self):
        od_df = pd.read_csv("data/mandl/mandl1_demand.csv")
        self.od_matrix = od_df.pivot(
            index='from', columns='to', values='demand').fillna(0)
        all_nodes = range(1, 16)
        self.od_matrix = self.od_matrix.reindex(
            index=all_nodes, columns=all_nodes, fill_value=0)
        self.od_matrix = self.od_matrix.astype(int)

    def _init_pedestrian_graph(self):
        self.graph_pedestrian = self.graph.copy()
        for _, _, data in self.graph_pedestrian.edges(data=True):
            if 'weight' in data:
                data['weight'] *= 4
