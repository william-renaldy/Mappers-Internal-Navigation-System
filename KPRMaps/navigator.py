import networkx as nx
# from networkx.drawing.layout import rotate_layout
import matplotlib.pyplot as plt
from haversine import haversine, Unit
import numpy as np
from KPRMaps.utility import LOCATIONS, CONNECTIONS
import plotly.graph_objs as go
from plotly.offline import plot
from KPRMaps.speech import SpeechToText, TextToSpeech
from threading import Thread


class Grapher:
        
    def __init__(self) -> None:
        self.locations = LOCATIONS
        self.connections = CONNECTIONS

        self.G = nx.Graph()
        self.graph = {}




    def create(self):
        self.G.add_nodes_from(self.locations.keys())

        for conn in self.connections:
            dist = haversine(self.locations[conn[0]], self.locations[conn[1]], unit=Unit.METERS)

            self.G.add_edge(conn[0], conn[1], weight = dist)

            gg = nx.to_dict_of_dicts(self.G)

            for node in gg:
                temp = []

                for j in gg[node]:
                    temp.append((j,gg[node][j]["weight"]))

                self.graph[node] = temp

        return self.graph


    def load_path(self,path,cost):
        res = nx.Graph()


        for i in range(1, len(path)):
            dist = haversine(self.locations[path[i-1]],self.locations[path[i]],unit= Unit.METERS)
            res.add_edge(path[i-1], path[i], weight = str(round(dist,2))+" m")


        pos = {loc: (self.locations[loc][1], self.locations[loc][0]) for loc in path}

        start = [path[0], (self.locations[path[0]][1], self.locations[path[0]][0])]



        print(start)


        # while(any([(start[0] != key1  and (start[1][1] > value1[1])) for key1,value1 in pos.items()])):
        #         print("Rotated")
        #         pos_array = np.array(list(pos.values()))
        #         pos_array = np.array([-pos_array[:,1], pos_array[:,0]]).T
        #         pos = dict(zip(pos.keys(), pos_array))

        


        # create the edge_trace object with weights as text annotations
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines',
            textposition='middle center',
            textfont=dict(color='black', size=10),
            text=[edge[2]['weight'] for edge in res.edges(data=True)]
        )

        for edge in res.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])


        # create the node_trace object with node labels as text annotations
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[node for node in res.nodes()],
            mode='markers+text',
            hoverinfo='text',
            textposition='top center',
            marker=dict(
                color='rgba(50, 50, 50, 0.5)',
                size=10,
                line=dict(width=2,color='white')
            )
        )

        # add the node coordinates to the node_trace object
        for node in res.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        # create the figure object and add the edge_trace and node_trace to it
        fig = go.Figure(data=[edge_trace, node_trace])

        # update the layout of the figure
        fig.update_layout(
            title='<br>Total Distance = ' + str(round(cost,2)) + " m",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 )],
            height=600
        )
        

        # display the figure
        Thread(target = self.audio_output, args = (path[0], path[-1], cost)).start()
        
        fig.show()
        return fig
    
        """
        pos = {loc: (self.locations[loc][1], self.locations[loc][0]) for loc in path}

        start = [path[0], (self.locations[path[0]][1], self.locations[path[0]][0])]



        print(start)


        while(any([(start[0] != key1  and (start[1][1] > value1[1])) for key1,value1 in pos.items()])):
                print("Rotated")
                pos_array = np.array(list(pos.values()))
                pos_array = np.array([-pos_array[:,1], pos_array[:,0]]).T
                pos = dict(zip(pos.keys(), pos_array))









        for edge in res.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        for node in res.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_info = 'Node: '+str(node)
            node_trace['text']+=tuple([node_info])
        
        # Color node points by the number of connections
        for node, adjacencies in enumerate(res.adjacency()):
            node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        
        # Create the layout
        layout = go.Layout(
            title='My Networkx Plot',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        # Create the figure
        fig = go.Figure(layout=layout)
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)
        node_hover_text = []
        for node, adjacencies in enumerate(res.adjacency()):
            node_hover_text.append(
                f"{node}: {len(adjacencies[1])} connections")

        # Update the hover text of the nodes in the plotly graph object
        fig.data[1].hovertext = node_hover_text

        # Define the edge labels
        edge_labels = []
        for edge in res.edges():
            edge_labels.append(res.get_edge_data(edge[0], edge[1])['weight']+" m")

        # Add the edge labels to the plotly graph object
        edge_trace['text'] = edge_labels
        fig.add_trace(edge_trace)

        # Update the title of the plotly graph object
        fig.update_layout(title_text="My Networkx Graph")

        # Return the plotly graph object
        return fig
        """


    def audio_output(self,start,end,cost):
        TextToSpeech().Text_to_speech(f"""
        The total distance from {start} to {end} is {round(cost,2)} meters
        """)



        



class Navigator:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        # H = {
        #     'Water Purifier': 1,
        #     '2nd Year Classroom': 1,
        #     'HOD Office': 1,
        #     '3rd Year Class': 1,
        #     'Steps': 1,
        #     'Round Table': 1,
        #     '4th Year Class': 1,
        #     'AI Lab': 1,
        #     'Staff Room': 1,
        #     'My Location' : 1
        # }

        return 1

    def a_star_algorithm(self, start_node, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node
        # print(open_list)

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                # print(v)
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                cost = g[n]
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path, cost

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            # print(open_list)
            # print(closed_list)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None
    

class AudioController:



    def _find_similarity(self,word):

        similist = []
        wordlist = []
        
        for location in LOCATIONS:

            n = 2

            set1 = set(word[i:i+n] for i in range(len(word)-n+1))
            set2 = set(location[i:i+n] for i in range(len(location)-n+1))

            similarity = len(set1.intersection(set2)) / len(set1.union(set2))

            similist.append(similarity)
            wordlist.append(location)

        max_similar = max(similist)
        print(max_similar)
        if(max_similar < 0.25):
            return None
        return wordlist[similist.index(max_similar)]


    def audio_input(self):
        text = SpeechToText().Speech_to_text()
        if(text != False):
            text_list = text.split()
        else:
            print(text)
            return
        
        print(text_list)
        loc = []
        for word in text_list:
            res = self._find_similarity(word)
            if(res is not None):
                loc.append(res)

        if(len(loc) >= 2):
            start,end = loc[0],loc[1]
            g = Grapher()

            graph = g.create()

            x = Navigator(graph)
            path,cost = x.a_star_algorithm(start,end)
            g.load_path(path,cost)




        







if __name__ == "__main__":
    g = Grapher()
    graph = g.create()
    
    x = Navigator(graph)
    path,cost = x.a_star_algorithm("E-Gate","E-Gate")

    g.load_path(path,cost)
    