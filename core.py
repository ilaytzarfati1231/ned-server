import hashlib
from pyformlang.regular_expression import PythonRegex
from pyformlang.finite_automaton import *
from pyformlang.regular_expression import *
import graphviz
import logging
import os



logger = logging.getLogger(__name__)


def hash_regex_string(regex_string):
    return hashlib.md5(regex_string.encode()).hexdigest()


def generate_image_from_automata(automata, output_path):
    current_path = os.environ.get("PATH", "")
    # Add the Graphviz bin directory to the PATH
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    logger.info(f'Generating image at: {output_path}')
    
    graph = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})

    # Add nodes
    for state in automata.states:
        if state in automata.final_states:
            graph.node(str(state.value), shape="doublecircle")
        elif isinstance(automata, DeterministicFiniteAutomaton) and state == automata.start_state:
            graph.node(str(state.value), shape="circle", style="bold")
        elif state in automata.start_states:
            graph.node(str(state.value), shape="circle", style="bold")
        else:
            graph.node(str(state.value), shape="circle")

    # Add transitions
    transitions = automata.to_dict()
    for from_state, symbols in transitions.items():
        for symbol, to_states in symbols.items():
            if not isinstance(to_states, set):
                to_states = {to_states}
            for to_state in to_states:
                graph.edge(str(from_state.value), str(to_state.value), label=str(symbol))

    graph.render(output_path, format='png', cleanup=True)
    logger.info(f'Image generated at: {output_path}.png')



def parse_regex_string(regex_string):
    logger.info(f'Parsing regex string: {regex_string}')
    regex = PythonRegex(regex_string.replace(" ", ""))
    automata = regex.to_epsilon_nfa().minimize()
    
    logger.info(f'Regex parsed and automata generated')
    return automata

def parse_dict_to_automata(Q, Sigma, Delta, q0, F):
    states = [State(state) for state in Q]
    symbols = [Symbol(symbol) for symbol in Sigma]
    transitions = NondeterministicTransitionFunction()
    for transition in Delta:
        state, symbol, next_state = transition.split(":")
        if (State(state) not in states):
            raise ValueError(f"State {state} not in Q")
        if (State(next_state) not in states):
            raise ValueError(f"State {next_state} not in Q")
        if (Symbol(symbol) not in symbols):
            raise ValueError(f"Symbol {symbol} not in Sigma")
        transitions.add_transition(State(state), Symbol(symbol), State(next_state))
    if State(q0) not in states:
        raise ValueError(f"Initial state {q0} not in Q")
    initial_states = [State(q0)]
    if not all([State(state) in states for state in F]):
        raise ValueError(f"Final states {F} not in Q")
    final_states = [State(state) for state in F]
    automata = NondeterministicFiniteAutomaton(states, symbols, transitions, initial_states, final_states)
    return automata



def automata_edit_distance_graph(automata1, automata2):
    # States of the product automata are tuples of states from the input automatas
    product_states = [(q1, q2) for q1 in automata1.states for q2 in automata2.states]

    # Alphabet is the intersection of the alphabets of the input automatas4
    product_automata_input_symbols = [(a, b) for a in automata1.symbols for b in automata2.symbols] +[(a,"epsilon") for a in automata1.symbols] +[("epsilon",b) for b in automata2.symbols]


    # Initial state of the product automata is the pair of initial states of the input automatas
    start_states_in_automata1 = set()
    start_states_in_automata2 = set()
    if isinstance(automata1, DeterministicFiniteAutomaton):
        start_states_in_automata1.add(automata1.start_state)
    else:
        start_states_in_automata1 = automata1.start_states
    if isinstance(automata2, DeterministicFiniteAutomaton):
        start_states_in_automata2.add(automata2.start_state)
    else:
        start_states_in_automata2 = automata2.start_states

    product_automata_initial_states = [State("" + str(q1.value) + "," + str(q2.value)) for q1 in
                                       start_states_in_automata1 for q2 in start_states_in_automata2]

    # Final states of the product automata are pairs of states where both input automatas are in final states
    product_automata_final_states = [State("" + str(q1.value) + "," + str(q2.value)) for q1 in automata1.final_states
                                     for q2 in automata2.final_states]

    # Transition function of the product automata
    product_transitions = NondeterministicTransitionFunction()
    for q1, q2 in product_states:
        for symbol in product_automata_input_symbols:
            next_q1_set = automata1.to_dict().get(q1, {}).get(symbol[0], None)
            next_q2_set = automata2.to_dict().get(q2, {}).get(symbol[1], None)
            if not isinstance(next_q1_set, set):
                next_q1_set = {next_q1_set}
            if not isinstance(next_q2_set , set):
                next_q2_set = {next_q2_set}
            for next_q1 in next_q1_set:
                for next_q2 in next_q2_set:
                    if next_q1 is not None and next_q2 is not None:
                        product_transitions.add_transition(State("" + str(q1.value) + "," + str(q2.value)), symbol,
                                                           State("" + str(next_q1.value) + "," + str(next_q2.value)))
                    if next_q1 is not None:
                        product_transitions.add_transition(State("" + str(q1.value) + "," + str(q2.value)), (symbol[0], "epsilon"),
                                                           State("" + str(next_q1.value) + "," + str(q2.value)))
                    if next_q2 is not None:
                        product_transitions.add_transition(State("" + str(q1.value) + "," + str(q2.value)), ("epsilon", symbol[1]),
                                                           State("" + str(q1.value) + "," + str(next_q2.value)))
    # Create the product automata
    product_states = [State("" + str(q1.value) + "," + str(q2.value)) for q1 in automata1.states for q2 in automata2.states]
    product_automata = NondeterministicFiniteAutomaton(product_states, product_automata_input_symbols,
                                                       product_transitions, product_automata_initial_states,
                                                       product_automata_final_states)
    return product_automata


def BFS_on_automata(automata):
    stateNumbers = {}
    i = 0
    queue = []
    if isinstance(automata, DeterministicFiniteAutomaton):
        queue.append(automata.start_state)
    else:
        for start_state in automata.start_states:
            queue.append(start_state)
    while queue:
        s = queue.pop(0)
        stateNumbers[s.value] = i
        i += 1
        for symbol in automata.symbols:
            next_state = automata.to_dict().get(s, {}).get(symbol, None)
            if next_state is not None:
                if isinstance(next_state, set):
                    for state in next_state:
                        if state not in stateNumbers and state not in queue:
                            queue.append(state)
                elif next_state not in stateNumbers and next_state not in queue:
                    queue.append(next_state)
    return stateNumbers

def _check_if_automata_is_product(automata):
    for state in automata.states:
        if isinstance(state.value, str):
            if "," in state.value:
                return True
    return False


def make_from_automata_graph(automata, state_numbers):
    nodes = []
    source_nodes = []
    edges = {}
    target_nodes = []

    for state in automata.states:
        if state.value in state_numbers:
            nodes.append(state_numbers[state.value])
        if state in automata.final_states and state.value in state_numbers:
            target_nodes.append(state_numbers[state.value])

    if isinstance(automata, DeterministicFiniteAutomaton):
        source_nodes.append(state_numbers[automata.start_state.value])
    else:
        for start_state in automata.start_states:
            source_nodes.append(state_numbers[start_state.value])

    for state in automata.states:
        for symbol in automata.symbols:
            next_state = automata.to_dict().get(state, {}).get(symbol, None)
            if next_state is not None:
                if not isinstance(next_state, set):
                    next_state = {next_state}
                for ns in next_state:
                    if ns.value in state_numbers and state.value in state_numbers:
                        if (state_numbers[state.value], state_numbers[ns.value]) not in edges:
                            edges[(state_numbers[state.value], state_numbers[ns.value])] = weightFunction(symbol)
                        else:
                            edges[(state_numbers[state.value], state_numbers[ns.value])] = min(
                                weightFunction(symbol),
                                edges[(state_numbers[state.value], state_numbers[ns.value])])
    return nodes, edges, source_nodes, target_nodes


def weightFunction(symbol):
    if symbol.value[0] == symbol.value[1]:
        return 0
    return 1


def convertToGraph(automata):
    if _check_if_automata_is_product(automata):
        stateNumbers = BFS_on_automata(automata)
        print(stateNumbers)
        V, edges,source_nodes,target_nodes = make_from_automata_graph(automata, stateNumbers)
        print(source_nodes)
        print(target_nodes)
        return V, edges, source_nodes, stateNumbers,target_nodes
    else:
        return None, None, None, None, None


def karp_mean_cycle(V,edges,source_nodes):
    F = {}
    for v in V:
        if v in source_nodes:
            F[(v,0)] = 0
        else:
            F[(v,0)] = float("inf")

    for k in range(1,len(V)+1):
        for v in V:
            for u in V:
                if (u,v) in edges:
                    if (u,k-1) in F:
                        if (v,k) not in F:
                            F[(v,k)] = F[(u,k-1)] + edges[(u,v)]
                        else:
                            F[(v,k)] = min(F[(u,k-1)] + edges[(u,v)],F[(v,k)])
    min_cycle = float("inf")
    for v in V:
        max_cycle = -1
        for k in range(len(V)):
            if (v,k) in F and (v,len(V)) in F:
                n = (F[(v,len(V))] - F[(v,k)])
                d = len(V) - k
                max_cycle = max(max_cycle,n/d)
        if max_cycle != -1:
            min_cycle = min(min_cycle,max_cycle)
    return min_cycle


def shortest_path_with_k_edges(nodes, edges, sources, targets, k):
    D= {}
    for node in nodes:
        D[(node,0)] = float("inf")
        if node in sources:
            D[(node,0)] = 0
    for i in range(1,k+1):
        for node in nodes:
            D[(node,i)] = float("inf")
            for neighbor in nodes:
                if (neighbor,node) in edges:
                    D[(node,i)] = min(D[(node,i)],D[(neighbor,i-1)] + edges[(neighbor,node)])

    min_cost = float("inf")
    for target in targets:
        min_cost = min(min_cost, D[(target,k)] / k)
    if min_cost == float("inf"):
        return -1
    return min_cost


def minimum_mean_path_value(V,edges,source_nodes,target_nodes):
    min_cost = float("inf")
    for i in range(1,len(V)+1):
        min_cost_i = shortest_path_with_k_edges(V,edges,source_nodes,target_nodes,i)
        if min_cost_i != -1:
            min_cost = min(min_cost_i ,min_cost)
    return min_cost


def inf_inf(automata):
    V, edges, source_nodes, state_numbers,target_nodes = convertToGraph(automata)
    if V is None:
        return "ERROR"
    intersection = set(source_nodes).intersection(set(target_nodes))
    if len(intersection) > 0:
        return 0.0
    mean_cycle = karp_mean_cycle(V,edges,source_nodes)
    mean_path = minimum_mean_path_value(V,edges,source_nodes,target_nodes)
    if mean_cycle < 0 or mean_cycle == float("inf") or mean_cycle > mean_path:
        return mean_path
    return mean_cycle

def dijkstra(V,edges,source_nodes,target_nodes):
    D = {}
    for v in V:
        D[v] = float("inf")
    for source in source_nodes:
        D[source] = 0
    Q = set(V)
    while Q:
        u = min(Q,key = lambda x: D[x])
        Q.remove(u)
        for v in V:
            if (u,v) in edges:
                if D[v] > D[u] + edges[(u,v)]:
                    D[v] = D[u] + edges[(u,v)]
    min_cost = float("inf")
    for target in target_nodes:
        min_cost = min(min_cost,D[target])
    return min_cost


def sum_inf_inf(automata):
    V, edges, source_nodes, state_numbers,target_nodes = convertToGraph(automata)
    if V is None:
        return "ERROR"
    intersection = set(source_nodes).intersection(set(target_nodes))
    if len(intersection) > 0:
        return 0.0
    return dijkstra(V,edges,source_nodes,target_nodes)

