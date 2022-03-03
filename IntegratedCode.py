from geopy.geocoders import Nominatim
import googlemaps
from collections import defaultdict
from geopy import distance
from gmplot import gmplot
import os
import plotly.graph_objects as go
import re
from igraph import *
import cairo

# Stop Total : LRT KLCC, KL SENTRAL, KLIA, Subang Airport, Masjid Jamek, BTS, TBS, Perhentian Bas Bentayan ,
# Skudai Parade, JB Larkin Terminal, Senai International Airport, Singapore Airport,
#  JDT Stadium,
#
# KLCC => JDT Stadium
# Option 1
# Take KLCC LRT to KL Sentral => Take KL Sentral Train to Bandar Tasik Selatan => Walk to TBS =>
# Take a bus from TBS to JB Larkin Terminal => Take a taxi to JDT Stadium
#
# Option 2
# Take KLCC LRT to Masjid Jamek => Take Masjid Jamek Train to Bandar Tasik Selatan => WalktoTBS => Take a bus from TBS to Muar
# => Muar bus stop to JB larkin terminal => Take a taxi to JDT Stadium
#
# Option 3
# KLCC LRT to KL sentral => KL Sentral train to KLIA => Flight to Senai International Airport =>
# TAke taxi to JDT Stadium
#
# Option 4
# KLCC LRT to KL sentral => KL Sentral train to KLIA => Flight to Senai International Airport =>
# Take bus P403(Senai) to Skudai Parade => Skudai Parade bus to Johor Larkin => Walk to JDT Stadium
#
# Option 5
# KLCC LRT to KL sentral => KL Sentral train to Subang Airport => Flight to Singapore Seletar Airport
# => Take Taxi to JDT Stadium
#
# Option 6
# KLCC LRT to KL Sentral => KL Sentral train to Subang Airport => Flight to Senai Airport =>
# Take Taxi to JDT STADIUM

#initializor
gmap = googlemaps.Client(key='AIzaSyAmomq-Z8ALrMwa2JI6hKRL6FDSEmk_1Vk')
geolocator = Nominatim(user_agent="geoAPIExercise")

#Question 1
#Address of Stops
LRTKLCC = geolocator.geocode("KLCC")
KLSentral = geolocator.geocode("KL Sentral")
KLIA = geolocator.geocode("KLIA")
SubangAirport = geolocator.geocode("Sultan Abdul Aziz Shah Airport")
MasjidJamek = geolocator.geocode("Masjid Jamek")
BTS = geolocator.geocode("Bandar Tasik Selatan")
TBS = geolocator.geocode("Terminal Bersepadu Selatan")
Bentayan = geolocator.geocode("Terminal Bentayan Muar")
SkudaiParade = geolocator.geocode("Skudai Parade")
JBLarkinTerminal = geolocator.geocode("Larkin Sentral")
SenaiAirport = geolocator.geocode("Senai International Airport")
SingaporeAirport = geolocator.geocode("Seletar Airport")
LarkinStadium = geolocator.geocode("Larkin Stadium")

# print(LRTKLCC)
# print(KLSentral)
# print(KLIA)

#Question 2
#Distance between stops
#Option 1 (Take KLCC LRT to KL Sentral => Take KL Sentral Train to Bandar Tasik Selatan => Walk to TBS => Take a bus from TBS to JB Larkin Terminal => Take a taxi to JDT Stadium)
KLCCtoKLSentral_Transit = gmap.distance_matrix((LRTKLCC.latitude,LRTKLCC.longitude), (KLSentral.latitude,KLSentral.longitude), mode="transit")
KLSentraltoBTS_Transit = gmap.distance_matrix((KLSentral.latitude,KLSentral.longitude),(BTS.latitude,BTS.longitude),mode ="transit")
BTStoTBS_Walk = gmap.distance_matrix((BTS.latitude,BTS.longitude),(TBS.latitude,TBS.longitude),mode ='walking')
TBStoJBLarkinTerminal_Bus = gmap.distance_matrix((TBS.latitude,TBS.longitude),(JBLarkinTerminal.latitude,JBLarkinTerminal.longitude),mode ="transit",transit_mode ='bus')
JBLarkinTerminaltoJDTStadium = gmap.distance_matrix((JBLarkinTerminal.latitude,JBLarkinTerminal.longitude),(LarkinStadium.latitude,LarkinStadium.longitude),mode ='driving')

#Option 2 (Take KLCC LRT to Masjid Jamek => Take Masjid Jamek Train to Bandar Tasik Selatan => WalktoTBS => Take a bus from TBS to Muar => Muar bus stop to JB larkin terminal => Take a taxi to JDT Stadium)
KLCCtoMasjidJamek_Transit = gmap.distance_matrix((LRTKLCC.latitude,LRTKLCC.longitude),(MasjidJamek.latitude,MasjidJamek.longitude), mode ="transit")
MasjidJamektoBTS_Transit = gmap.distance_matrix((MasjidJamek.latitude,MasjidJamek.longitude),(BTS.latitude,BTS.longitude), mode = 'transit')
#WalkBTStoTBS
TBStoBentayan_Bus = gmap.distance_matrix((TBS.latitude,TBS.longitude),(Bentayan.latitude,Bentayan.longitude), mode ='transit', transit_mode = 'bus')
BentayantoJBLarkinTerminal_Bus = gmap.distance_matrix((Bentayan.latitude,Bentayan.longitude),(JBLarkinTerminal.latitude,JBLarkinTerminal.longitude), mode ='transit',transit_mode='bus')
#JBLarkinTerminaltoJDTStadium

#Option 3 (KLCC LRT to KL sentral => KL Sentral train to KLIA => Flight to Senai International Airport => TAke taxi to Larkin Stadium)
#KLCCtoKLSentral_Transit
KLSentraltoKLIA = gmap.distance_matrix((KLSentral.latitude,KLSentral.longitude),(KLIA.latitude,KLIA.longitude), mode = "transit")
KLIAtoSenaiFlight = distance.distance((KLIA.latitude,KLIA.longitude),(SenaiAirport.latitude,SenaiAirport.longitude)).m
SenaitoJDTStadium = gmap.distance_matrix((SenaiAirport.latitude,SenaiAirport.longitude),(LarkinStadium.latitude,LarkinStadium.longitude), mode = "driving")

#Option 4 (KLCC LRT to KL sentral => KL Sentral train to KLIA => Flight to Senai International Airport => Take bus P403 to Skudai Parade => Skudai Parade bus to Larkin Terminal => Walk to Larkin Stadium)
#KLCCtoKLSentral_Transit
# KLSentraltoKLIA
# KLIAtoSenaiFlight
SenaitoSkudaiParade = gmap.distance_matrix((SenaiAirport.latitude,SenaiAirport.longitude),(SkudaiParade.latitude,SkudaiParade.longitude), mode="driving")
SkudaiParadetoLarkinTerminal= gmap.distance_matrix((SkudaiParade.latitude,SkudaiParade.longitude),((JBLarkinTerminal.latitude,JBLarkinTerminal.longitude)), mode = "transit", transit_mode = 'bus')
LarkinTerminaltoJDTStadium_Walk = gmap.distance_matrix ((JBLarkinTerminal.latitude,JBLarkinTerminal.longitude),(LarkinStadium.latitude,LarkinStadium.longitude),mode ='walking')

#Option 5 (KLCC LRT to KL sentral => KL Sentrail train to Subang Airport => Flight to Singapore Seletar Airport => Take Taxi to Larkin Stadium)
#KLCCtoKLSentral_Transit
KLSentraltoSubang = gmap.distance_matrix((KLSentral.latitude,KLSentral.longitude), (SubangAirport.latitude,SubangAirport.longitude), mode="transit")
SubangAirportToSingapore = distance.distance((SubangAirport.latitude,SubangAirport.longitude),(SingaporeAirport.latitude,SingaporeAirport.longitude)).m
SingaporetoJDTStadium = gmap.distance_matrix((SingaporeAirport.latitude,SingaporeAirport.longitude),(LarkinStadium.latitude,LarkinStadium.longitude), mode = "driving")

#Option 6 (KLCC LRT to KL Sentral => KL Sentral train to Subang Airport => Flight to Senai Airport => Take Taxi to JDTStadium)
#KLCCtoKLSentral_Transit
# KLSentraltoSubang
SubangAirportToSenai = distance.distance((SubangAirport.latitude,SubangAirport.longitude),(SenaiAirport.latitude,SenaiAirport.longitude)).m
# SenaitoJDTStadium

#Question 3
g = Graph(directed = True)
# print("Is the graph directed:", g.is_directed())
g.add_vertices(13)

listofstops = ["KLCC", "KL Sentral", "KLIA", "Sultan Abdul Aziz Shah Airport", "Masjid Jamek", "Bandar Tasik Selatan",
               "Terminal Bersepadu Selatan", "Terminal Bentayan Muar",
               "Skudai Parade", "Larkin Sentral", "Senai International Airport", "Seletar Airport","Larkin Stadium"]

# Add ids and labels to vertices
for i in range(len(g.vs)):
    g.vs[i]["id"]= i
    g.vs[i]["label"] = listofstops[i]
    g.vs[i]["stops"] = listofstops[i]
    # Add edges

g.add_edges([(0,1),(0,4),(1,3),(1,2),(1,5),(2,10),(3,10),(3,11),(4,5),(5,6),(6,9),(6,7),(7,9),(8,9),(9,12),(10,8),(10,12),(11,12)])
# Add weights and edge labels
weights = [KLCCtoKLSentral_Transit['rows'][0]['elements'][0]['distance']['value'],
           KLCCtoMasjidJamek_Transit['rows'][0]['elements'][0]['distance']['value'],
           KLSentraltoSubang['rows'][0]['elements'][0]['distance']['value'],
           KLSentraltoKLIA['rows'][0]['elements'][0]['distance']['value'],
           KLSentraltoBTS_Transit['rows'][0]['elements'][0]['distance']['value'],
           KLIAtoSenaiFlight,SubangAirportToSenai,SubangAirportToSingapore,
           MasjidJamektoBTS_Transit['rows'][0]['elements'][0]['distance']['value'],BTStoTBS_Walk['rows'][0]['elements'][0]['distance']['value'],
           TBStoJBLarkinTerminal_Bus['rows'][0]['elements'][0]['distance']['value'],TBStoBentayan_Bus['rows'][0]['elements'][0]['distance']['value'],
           BentayantoJBLarkinTerminal_Bus['rows'][0]['elements'][0]['distance']['value'],SkudaiParadetoLarkinTerminal['rows'][0]['elements'][0]['distance']['value'],
           JBLarkinTerminaltoJDTStadium['rows'][0]['elements'][0]['distance']['value'],SenaitoSkudaiParade['rows'][0]['elements'][0]['distance']['value'],
           SenaitoJDTStadium['rows'][0]['elements'][0]['distance']['value'],SingaporetoJDTStadium['rows'][0]['elements'][0]['distance']['value']]
g.es['weight'] = weights
g.es['label'] = weights

visual_style = {}
out_name = "graph.png"
visual_style["bbox"] = (1920,1080)
visual_style["margin"] = 20
# visual_style["vertex_color"] = 'blue'
g.vs["color"] = ["red", "green", "blue", "yellow", "orange", "gold", "purple", "pink","violet","plum","maroon"]
# Set vertex size
visual_style["vertex_size"] = 50# Set vertex lable size
visual_style["vertex_label_size"] = 15
# Don't curve the edges
visual_style["edge_curved"] = False
# Set the layout
my_layout = g.layout_lgl()
visual_style["layout"] = my_layout# Plot the graph
plot(g, out_name, **visual_style)

heuristics = {}
heuristics[g.vs[0]] = distance.distance((LRTKLCC.latitude, LRTKLCC.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[1]] = distance.distance((KLSentral.latitude, KLSentral.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[2]] = distance.distance((KLIA.latitude, KLIA.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[3]] = distance.distance((SubangAirport.latitude, SubangAirport.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[4]] = distance.distance((MasjidJamek.latitude, MasjidJamek.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[5]] = distance.distance((BTS.latitude, BTS.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[6]] = distance.distance((TBS.latitude, TBS.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[7]] = distance.distance((Bentayan.latitude, Bentayan.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[8]] = distance.distance((SkudaiParade.latitude, SkudaiParade.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[9]] = distance.distance((JBLarkinTerminal.latitude, JBLarkinTerminal.longitude),
                                        (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[10]] = distance.distance((SenaiAirport.latitude, SenaiAirport.longitude),
                                         (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[11]] = distance.distance((SingaporeAirport.latitude, SingaporeAirport.longitude),
                                         (LarkinStadium.latitude, LarkinStadium.longitude)).m
heuristics[g.vs[12]] = distance.distance((LarkinStadium.latitude, LarkinStadium.longitude),
                                         (LarkinStadium.latitude, LarkinStadium.longitude)).m

# This class represent a node for the purpose of A* Algorithm
class Node:

    # Initialize the class
    def __init__(self, name: str, parent: str):
        self.name = name
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # Sort nodes
    def __lt__(self, other):
        return self.f < other.f

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))


# A* search
def astar_search(graph, heuristics, start, end):
    # Create lists for open nodes and closed nodes
    open = []
    closed = []
    path = []
    # Create a start node and an goal node
    start_node = Node(start, None)
    goal_node = Node(end, None)

    # Add the start node
    open.append(start_node)

    # Loop until the open list is empty
    while len(open) > 0:

        # Sort the open list to get the node with the lowest cost first
        open.sort()

        # Get the node with the lowest cost
        current_node = open.pop(0)
        # Add the current node to the closed list
        closed.append(current_node)

        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            while current_node != start_node:
                path.append(current_node.name['id'])
                current_node = current_node.parent
            path.append(start_node.name['id'])
            # Return reversed path
            return path[::-1]

        # Get neighbours
        neighbors = graph.neighbors(current_node.name['id'],mode=OUT)

        # Loop neighbors
        for index in neighbors:
            # Create a neighbor node
            neighbor = Node(graph.vs[index], current_node)
            # print(neighbor.name)
            # Check if the neighbor is in the closed list
            if (neighbor in closed):
                continue

            # Calculate full path cost
            toint = (graph.es.select(_source = current_node.name['id'], _target = neighbor.name['id'])['weight'])[0]
            neighbor.g = current_node.g + toint
            neighbor.h = heuristics.get(neighbor.name)
            neighbor.f = neighbor.g + neighbor.h

            # Check if neighbor is in open list and if it has a lower f value
            if (add_to_open(open, neighbor) == True):
                # Everything is green, add neighbor to open list
                open.append(neighbor)

    # Return None, no path is found
    return path
# Check if a neighbor should be added to open list
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f > node.f):
            return False
    return True


def path_cost(graph, path, weights=None):
    pathcost = 0
    for i in range(len(path)):
        if i > 0:
            edge=graph.es.find(_source=path[i-1], _target=path[i])
            if weights != None:
                pathcost += edge[weights]
            else:
                #just count the number of edges
                pathcost += 1
    return pathcost

def yen_igraph(graph, source, target, num_k, weights,heuristics):
    import queue
    #Shortest path from the source to the target
    A = [astar_search(graph,heuristics,source,target)]
    A_costs = [path_cost(graph, A[0], weights)]
    #Initialize the heap to store the potential kth shortest path
    B = queue.PriorityQueue()
    for k in range(1, num_k):
        #The spur node ranges from the first node to the next to last node in the shortest path
        for i in range(len(A[k-1])-1):
            #Spur node is retrieved from the previous k-shortest path, k − 1
            spurNode = A[k-1][i]
            #The sequence of nodes from the source to the spur node of the previous k-shortest path
            rootPath = A[k-1][:i]

            #We store the removed edges
            removed_edges = []

            for path in A:
                if len(path) - 1 > i and rootPath == path[:i]:
                    #Remove the links that are part of the previous shortest paths which share the same root path
                    edge = graph.es.select(_source=path[i], _target=path[i+1])
                    if len(edge) == 0:
                        continue #edge already deleted
                    edge = edge[0]
                    removed_edges.append((path[i], path[i+1], edge.attributes()))
                    edge.delete()

            #Calculate the spur path from the spur node to the target
            spurPath = astar_search(graph, heuristics,graph.vs[spurNode],target)
            if len(spurPath) > 0:
                #Entire path is made up of the root path and spur path
                totalPath = rootPath + spurPath
                totalPathCost = path_cost(graph, totalPath, weights)
                #Add the potential k-shortest path to the heap
                B.put((totalPathCost, totalPath))

            #Add back the edges that were removed from the graph
            for removed_edge in removed_edges:
                node_start, node_end, cost = removed_edge
                graph.add_edge(node_start, node_end)
                edge = graph.es.select(_source=node_start, _target=node_end)[0]
                edge.update_attributes(cost)

        #Sort the potential k-shortest paths by cost
        #B is already sorted (Since we are using .priorityQueue() = it will retrieve the lowest first))
        #Add the lowest cost path becomes the k-shortest path.
        while True:
            cost_, path_ = B.get()
            if path_ not in A:
                #We found a new path to add
                A.append(path_)
                A_costs.append(cost_)
                break

    return A, A_costs

sourcevertice = g.vs[0]
targetvertice = g.vs[12]
indexofshortestpaths = yen_igraph(g,sourcevertice,targetvertice,6,None,heuristics)[0]
print(indexofshortestpaths)

def getPathName(index,array):
    arraypaths = []
    for i in range(len(array[index])):
        arraypaths.append(g.vs[array[index][i]]["stops"])
    return arraypaths


allpathname = getPathName(0,indexofshortestpaths)
shortest = "The shortest distance possible calculated by A* Algorithm & Yen's Algorithm is "
for i in allpathname:
    if (i == allpathname[len(allpathname)-1]):
        shortest += i
    else:
        shortest += i + " => "

print(shortest)

#Question 4
thelatnodes = []
thelongnodes = []

def GetLatPath(path):
    thelatnodes = []
    for destination in path:
        pathcurrent = geolocator.geocode(destination)
        thelatnodes.append(pathcurrent.latitude)
    return thelatnodes

def GetLongPath(path):
    thelongnodes = []
    for destination in path:
        pathcurrent = geolocator.geocode(destination)
        thelongnodes.append(pathcurrent.longitude)
    return thelongnodes

Center = geolocator.geocode("Segamat")
gmapone = gmplot.GoogleMapPlotter(Center.latitude,Center.longitude,9, apikey='AIzaSyDyva-TKgS2ex0Z0JsdP9UDMo4sf5x15UI')
# gmapone.marker(GetLatPath(arraypaths),GetLongPath(arraypaths),title="Test")
gmapone.coloricon = "http://www.googlemapsmarkers.com/v1/%s/"
gmapone.marker(Center.latitude,Center.longitude, "#f0dd92", title="The Center")
gmapone.scatter(GetLatPath(getPathName(0,indexofshortestpaths)),GetLongPath(getPathName(0,indexofshortestpaths)),'purple',size = 1300, marker = False)
gmapone.plot(GetLatPath(getPathName(0,indexofshortestpaths)),GetLongPath(getPathName(0,indexofshortestpaths)),'green',edge_width = 5)
gmapone.draw("BestPathSelected.html")


# Question 5

class WebText:

    def __init__(self, filedata, placeName):
        with open('stopword.txt', 'r') as r, open('PositiveWords.txt', encoding='utf-8') as p, open('NegativeWords.txt', encoding='utf-8') as n:
            stopdata = r.read()
            positive = p.read()
            negative = n.read()

        self.filedata = filedata.lower()
        self.stopdata = stopdata.lower().split()
        self.positive = positive.split(', ')
        self.negative = negative.split(', ')
        self.placeName = placeName

    def do_all(self, display):

        clean_art = self.filter_stop(display)
        sentiment = self.sentiment(clean_art, display)

        return  sentiment


#Question 5
    def filter_stop(self, display):
        # remove regex from file data
        self.file = re.sub(r"[^A-Za-z0-9 '\n]+", '', self.filedata)

        # filter stop word
        self.filecheck = self.file.splitlines()
        self.stopcount = []
        for i in range(len(self.filecheck)):
            # skip empty line
            fileline = self.filecheck[i]
            if not fileline:
                continue

            fileline = ' ' + fileline + ' '
            for stopcheck in self.stopdata:
                stopcheck = ' ' + stopcheck + ' '
                if len(stopcheck) <= len(fileline):
                    result = self.rabin_karp(fileline, stopcheck)
                if result:
                    temp = 0
                    for j in result:
                        j = j - temp
                        self.stopcount.append(stopcheck)
                        fileline = fileline[:j+1] + fileline[j+len(stopcheck):]
                        temp += len(stopcheck) - 1
            self.filecheck[i] = fileline

        if (display == True):
            # count word frequency in article
            self.words = self.file.split()
            self.wordcount = {}
            for i in self.words:
                self.wordcount[i] = self.words.count(i)
            # print('Word Count of '+ self.placeName + ': ', self.wordcount, '\n\n')
            # stop word frequency
            self.stopfreq = {}
            for i in self.stopcount:
                self.stopfreq[i] = self.stopcount.count(i)
            # print('Stop Word Frequency of '+ self.placeName +': ', self.stopfreq, '\n')
            # print('-----------------------------------------------------------------------------------------------------------')
            # print(self.filecheck, '\n\n')

            # plot stopword graph
            self.plot_stop(self.stopfreq, self.placeName)
            self.plot_word(self.wordcount, self.placeName)

        return self.filecheck
    # !uncomment fig.show after this
#Question 6
    def plot_word(self, wordcount, placeName):
        x_value = []
        y_value = []
        fig = go.Figure()

        for i, j in wordcount.items():
            x_value.append(i)
            y_value.append(j)

        fig.add_trace(go.Scatter(x=x_value, y=y_value,
                                 mode='markers', ))
        title = "Word Count Frequencies of " + placeName

        fig.update_layout(title=title,
                          xaxis_title='Word',
                          yaxis_title='Frequencies')

        fig.show()

    def plot_stop(self, stopfreq, placeName):
        x_value = []
        y_value = []
        fig = go.Figure()

        for i,j in stopfreq.items():
            x_value.append(i)
            y_value.append(j)

        fig.add_trace(go.Scatter(x = x_value, y = y_value,
                                 mode = 'markers',))
        title = "Stop Word Frequencies of " + placeName

        fig.update_layout(title = title,
                          xaxis_title = 'Stop Word',
                          yaxis_title = 'Frequencies')

        fig.show()

    def rabin_karp(self, word, pattern):
        wordlen = len(word)
        patlen = len(pattern)
        h = 1
        p = 0
        w = 0
        result = []
        d = 256
        q = 101

        for i in range(patlen - 1):
            h = (h*d) % q

        for i in range(patlen):
            p = (d*p + ord(pattern[i])) % q
            w = (d*w + ord(word[i])) % q

        for s in range(wordlen - patlen + 1):
            if p == w:
                match = True
                for i in range(patlen):
                    if pattern[i] != word[s+i]:
                        match = False
                        break
                if match:
                    result.append(s)
            if s < wordlen - patlen:
                w = (w-h * ord(word[s])) % q  #rmove letter s
                w = (w*d + ord(word[s + patlen])) % q # remover letter s+pl
                w = (w+q) % q  # make sure t>=0
        return result
#Question 7
    def BoyerMooreHorspool(self,pattern, text):
        m = len(pattern)
        n = len(text)
        skip = []
        result = []
        if m > n: return result #kalau m > n , maksudnya tak betul, so return result kosong
        for k in range(256): skip.append(m) #setiap 256 word ada value m
        for k in range(m - 1): #setiap word pattern ada value m - k - 1 dekat dalam array skip.
            skip[ord(pattern[k])] = m - k - 1
        skip = tuple(skip)
        k = m - 1 #first window k
        while k < n: #gerak sampai okay
            j = m - 1; # untuk track length pattern
            i = k #untuk track kita dekat mane
            while j >= 0 and text[i] == pattern[j]: #j >=0 tu sebab kita tak nak check sampai lebih dari pattern.
                j -= 1; # - sampai habis pattern punya word.
                i -= 1 # - sampai habis word punya length.

            if j == -1: #maksudnya betol so, kita masukkan
                result.append(i + 1)

            k += skip[ord(text[k])] #tambah skip value dengan wordtext[k] (nak letak dekat index mane)
        return result

    def sentiment(self, article, display):
        result = []
        self.posicount = []
        self.negacount = []
        for line in article:

            if not line:
                continue

            for positive in self.positive:
                # positive = ' ' + positive + ' '
                if len(positive) <= len(line):
                    result = self.BoyerMooreHorspool(positive.strip(), line.strip())
                if result:
                    for i in range(len(result)):
                        self.posicount.append(positive)

            for negative in self.negative:
                # negative = ' ' + negative + ' '
                if len(negative) <= len(line):
                    result = self.BoyerMooreHorspool(negative.strip(), line.strip())
                if result:
                    for i in range(len(result)):
                        self.negacount.append(negative)

        # print("The positive word count for " + self.placeName  + " is " + str(len(self.posicount)))
        # print("The negative word count for " + self.placeName  + " is " + str(len(self.negacount)))
        self.p = {}
        self.n = {}
        for i in self.posicount:
            self.p[i] = self.posicount.count(i)
        for i in self.negacount:
            self.n[i] = self.negacount.count(i)
        # print('Positive Word Frequency of ' + self.placeName + ': ', self.p, '\n')
        # print('Negative Word Frequency of ' + self.placeName + ': ', self.n, '\n')
        # print("============================================================================================================================")
    #Question 8
        if (display == True):
            titleFile = "Positive and Negative Words Found from " + self.placeName
            x = ["Positive", "Negative"]
            y = [len(self.posicount), len(self.negacount)]  ##Gantikan dgn variable count

            fig2 = go.Figure(layout=dict(
                title= titleFile,
                xaxis=dict(title="Word Category"),
                yaxis=dict(title="Frequency")
            ))

            fig2.add_trace(go.Histogram(histfunc="sum",
                                        y=y,
                                        x=x,
                                        ),
                           )

            fig2.show()
#Question 9
        if (len(self.posicount) > len(self.negacount)):
            return 'positive'
        elif (len(self.negacount) > len(self.posicount)):
            return 'negative'
        else:
            return 'neutral'

#####################################################################################
sentimentStop = []
with open('masjidJamek.txt', encoding='utf-8') as jamek, open('klSentral.txt', encoding='utf-8') as klSentral, open('lrtKLCC.txt', encoding='utf-8') as lrtKl, open('KLIA.txt', encoding='utf-8') as klia, open('subang.txt', encoding='utf-8') as subang:
    fileJamek = WebText(jamek.read(), 'Masjid Jamek')
    fileKlSentral = WebText(klSentral.read(), 'KL Sentral')
    fileLrtKl = WebText(lrtKl.read(), 'LRT KLCC')
    fileKlia = WebText(klia.read(), 'KLIA')
    fileSubang = WebText(subang.read(), 'Subang Airport')

with open('bts.txt', encoding='utf-8') as bts, open('tbs.txt', encoding='utf-8') as tbs, open('basBentayan.txt', encoding='utf-8') as bentayan, open('skudai.txt', encoding='utf-8') as skudai, open('jb.txt', encoding='utf-8') as jb:
    fileBts = WebText(bts.read(), 'BTS')
    fileTbs = WebText(tbs.read(), 'TBS')
    fileBentayan = WebText(bentayan.read(), 'Perhentian Bas Bentayan')
    fileSkudai = WebText(skudai.read(), 'Skudai Parade')
    fileJb = WebText(jb.read(), 'JB Larkin Terminal')


with open('senai.txt', encoding='utf-8') as senai, open('singapore.txt', encoding='utf-8') as singapore, open('jdt.txt', encoding='utf-8') as jdt:
    fileSenai = WebText(senai.read(), 'Senai International Airport')
    fileSingapore = WebText(singapore.read(), 'Singapore Airport')
    fileJdt = WebText(jdt.read(), 'JDT Stadium')

sentimentStop.append(fileLrtKl.do_all(True))
sentimentStop.append(fileKlSentral.do_all(True))
sentimentStop.append(fileKlia.do_all(True))
sentimentStop.append(fileSubang.do_all(True))
sentimentStop.append(fileJamek.do_all(True))
sentimentStop.append(fileBts.do_all(True))
sentimentStop.append(fileTbs.do_all(True))
sentimentStop.append(fileBentayan.do_all(True))
sentimentStop.append(fileSkudai.do_all(True))
sentimentStop.append(fileJb.do_all(True))
sentimentStop.append(fileSenai.do_all(True))
sentimentStop.append(fileSingapore.do_all(True))
sentimentStop.append(fileJdt.do_all(True))
print(sentimentStop)

# Question 10
# 10.Develop an algorithm to automatically generate summary of the 5-10 options
# based on their respective distance and sentiment. Then, the algorithm should be able to recommend the best
# option based on the distance AND the sentiment for the passenger to travel from one destination to another destination.
#pseudocode dia
#initialize recommend shortest & recommend sentiment
recommendshortest = 50
routewithrecommend = [[None for x in range(2)] for y in range(len(indexofshortestpaths))]
# Get Shortest Route (kalau index 0 -> recommend = 50, index 1 -> recommend - recommend/totaloption, index2 etc..)
for i in range(len(indexofshortestpaths)):
    recommendsentiment = 50
    routewithrecommend[i][0] = i
    if (i == 0 ):
        routewithrecommend[i][1] = recommendshortest
    else:
        recommendshortest -= (50/len(indexofshortestpaths))
        routewithrecommend[i][1] = recommendshortest
    # Get Sentiment Route (total jugak 50 , for each sentiment, kalau positive +(50/totaloption), kalau negative -(50/totaloption)
    # print("route" + str(i))
    for j in range(len(indexofshortestpaths[i])):
        if (sentimentStop[indexofshortestpaths[i][j]] == 'positive'):
            recommendsentiment = recommendsentiment + (50/len(indexofshortestpaths[i]))
            if (j == len(indexofshortestpaths[i])-1):
                routewithrecommend[i][1] = routewithrecommend[i][1] + recommendsentiment
        elif (sentimentStop[indexofshortestpaths[i][j]] == 'negative'):
            recommendsentiment = recommendsentiment - (50/len(indexofshortestpaths[i]))
            if ((j == len(indexofshortestpaths[i])-1)):
                routewithrecommend[i][1] = routewithrecommend[i][1] + recommendsentiment
        else:
            if ((j == len(indexofshortestpaths[i]) - 1)):
                routewithrecommend[i][1] = routewithrecommend[i][1] + recommendsentiment

#      then we rank the route from least recommended to most optimized route
rankedrecommend = sorted(routewithrecommend,key=lambda x: x[1])
optimized = getPathName(rankedrecommend[1][0],indexofshortestpaths)
print(rankedrecommend)
print("The best most optimized route will be " + ', '.join(optimized) + " with a recommend mark " + str(rankedrecommend[len(rankedrecommend)-1][1]))

