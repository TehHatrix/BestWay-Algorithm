# Bestway Navigator Algorithm
The navigation system is intended for suggesting the best optimized route when travelling to a specific destination. In an everyday-life situation, many people use the public transportation service to travel to places, either for business, travelling, visiting someone. The system plays an important role to make our life easier.

This system is made with that idea in mind. The main function of this system is to calculate the best optimized route based on the distance of the starting stop to the destination and the sentiment analysis. 

First, the system will make the graph with the stops acting as vertices and distance acts as edges, then the system will choose the shortest route among them. This system will also identify the sentiment each stops in the route based on latest articles, reviews and it will extract the positive and negative words, then count the frequency of those words. It will also make a histogram graph for each stop to increase readability.
Next, the system will choose the best route from the combination of the shortest
distance and the current sentiment for each route. Lastly, the system will rank from least recommended to best recommended route and will display the best recommended (most optimized) route.

## Problem to Solve
Problem 1: Malaysia has developed an integrated public transportation network which provides multiple options to passenger. Passenger may take the furthest route due to the lack of information on determining the combination of public transportation to get to their destination.

Problem 2: Even the shortest path able to be determine, passengers are still facing daily problem due to the long waiting time. The unusual and unexpected condition during the journey such as unexpected traffic congestion, unexpected delay, randomness in passengers’ demands OR weather changes need to be considered before making suggestion.

Problem 3: The hassle question faced by passengers to decide which options should be taken depending on the time and situation. For example, “Should I wait for the bus at this time, or should I walk a few miles to the train station, or should I just take a grab as the traffic at this time may not be congested”

## Algorithms used
A* Search Algorithm\
Yen's K-Shortest Algorithm\
Boyer Moore Horspool String Matching Algorithm\
*Our Custom Algorithm for Problem 3*

## API used
Google Distance Matrix API

## Output
![output](https://github.com/TehHatrix/BestWay-Algorithm/blob/main/README/Output.png?raw=true)
