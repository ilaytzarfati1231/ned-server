import hashlib
from pyformlang.regular_expression import PythonRegex
from pyformlang.finite_automaton import *
from pyformlang.regular_expression import *
import graphviz
import logging
import os



logger = logging.getLogger(__name__)


WeightFunctions = []

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

    # Add an invisible starting point with an arrow to the initial state
    if isinstance(automata, DeterministicFiniteAutomaton):
        initial_state = automata.start_state
    else:
        initial_state = next(iter(automata.start_states))

    graph.node('start', shape="point", style="invisible")
    graph.edge('start', str(initial_state.value), style="bold")

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

def generate_image_from_balanced_graph(V, E,init, output_path):
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    logger.info(f'Generating image at: {output_path}')
    
    graph = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})

    # Add nodes
    for vertex in V:
        if vertex in init:
            graph.node(str(vertex), shape="circle", style="bold")
        else:
            graph.node(str(vertex))
    # Add edges
    for (from_vertex,to_vertex), weight in E.items():
        label = f'{weight} ({from_vertex[2],to_vertex[2]})'
        graph.edge(str(from_vertex), str(to_vertex), label=str(label))

    graph.render(output_path, format='png', cleanup=True)
    logger.info(f'Image generated at: {output_path}.png')


def generate_image_from_arena(arena, output_path):
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    logger.info(f'Generating image at: {output_path}')
    
    graph = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})

    # Add Adam's vertices (squares)
    for vertex in arena.adam_vertices:
        graph.node(str(vertex), shape="square", style="bold")

    # Add Eve's vertices (diamonds)
    for vertex in arena.eve_vertices:
        graph.node(str(vertex), shape="diamond", style="dotted")

    # Add an invisible starting point with an arrow to the initial vertex
    graph.node('start', shape="point", style="invisible")
    graph.edge('start', str(arena.initial_vertex), style="bold")

    # Add edges
    for from_vertex, edges in arena.edges.items():
        for cost, to_vertex in edges:
            graph.edge(str(from_vertex), str(to_vertex), label=str(cost))

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
    save_for_later = {}

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
                            save_for_later[(state_numbers[state.value], state_numbers[ns.value])] = symbol
                        else:
                            if weightFunction(symbol) < edges[(state_numbers[state.value], state_numbers[ns.value])]:
                                edges[(state_numbers[state.value], state_numbers[ns.value])] = weightFunction(symbol)
                                save_for_later[(state_numbers[state.value], state_numbers[ns.value])] = symbol
    
    return nodes, edges, source_nodes, target_nodes, save_for_later


def weightFunction(symbol):
    if symbol.value[0] == symbol.value[1]:
        return 0
    return 1

def addWeightFunction(dict):


def labelFunction(symbol):
    if symbol.value[0] == "epsilon":
        return 1
    elif symbol.value[1] == "epsilon":
        return -1
    return 0


def convertToGraph(automata):
    if _check_if_automata_is_product(automata):
        stateNumbers = BFS_on_automata(automata)
        print(stateNumbers)
        V, edges,source_nodes,target_nodes,save_for_later = make_from_automata_graph(automata, stateNumbers)
        print(source_nodes)
        print(target_nodes)
        return V, edges, source_nodes, stateNumbers,target_nodes,save_for_later
    else:
        return None, None, None, None, None,None



def karp_mean_cycle(V, edges, source_nodes):
    logger.info('Computing Karp mean cycle')
    
    F = {}
    predecessor = {}
    
    for v in V:
        if v in source_nodes:
            F[(v, 0)] = 0
        else:
            F[(v, 0)] = float("inf")
        predecessor[(v, 0)] = None
        
    logger.info('Entering the min loop')
    for k in range(1, len(V) + 1):
        if k % 100 == 0:
            logger.info(f'loop number = {k} loop remaining = {len(V) - k}')
        for v in V:
            for u in V:
                if (u, v) in edges:
                    if (u, k - 1) in F:
                        if (v, k) not in F:
                            F[(v, k)] = F[(u, k - 1)] + edges[(u, v)]
                            predecessor[(v, k)] = u
                        else:
                            if F[(u, k - 1)] + edges[(u, v)] < F[(v, k)]:
                                F[(v, k)] = F[(u, k - 1)] + edges[(u, v)]
                                predecessor[(v, k)] = u
    
    min_cycle = float("inf")
    cycle_start = None
    cycle_length = None
    logger.info('Entering the second loop')
    
    for v in V:
        max_cycle = -1
        local_start = None
        local_length = None
        
        for k in range(len(V)):
            if (v, k) in F and (v, len(V)) in F:
                n = (F[(v, len(V))] - F[(v, k)])
                d = len(V) - k
                cycle_cost = n / d
                
                if cycle_cost > max_cycle:
                    max_cycle = cycle_cost
                    local_start = v
                    local_length = d
        
        if max_cycle != -1 and max_cycle < min_cycle:
            min_cycle = max_cycle
            cycle_start = local_start
            cycle_length = local_length
    
    if cycle_start is None:
        return float("inf"), []
    
    path = []
    current = cycle_start
    k = len(V)
    while k > 0 and cycle_length > 0:
        path.append(current)
        next_node = predecessor[(current, k)]
        current = next_node
        k -= 1
        cycle_length -= 1
        
    path.reverse()
    print(path)
    return min_cycle, path




def shortest_path_with_k_edges(nodes, edges, sources, targets, k):
    D = {}
    predecessor = {}

    for node in nodes:
        D[(node, 0)] = float("inf")
        if node in sources:
            D[(node, 0)] = 0
        predecessor[(node, 0)] = None

    for i in range(1, k + 1):
        for node in nodes:
            D[(node, i)] = float("inf")
            for neighbor in nodes:
                if (neighbor, node) in edges:
                    if D[(neighbor, i - 1)] + edges[(neighbor, node)] < D[(node, i)]:
                        D[(node, i)] = D[(neighbor, i - 1)] + edges[(neighbor, node)]
                        predecessor[(node, i)] = neighbor

    min_cost = float("inf")
    best_target = None

    for target in targets:
        if D[(target, k)] / k < min_cost:
            min_cost = D[(target, k)] / k
            best_target = target

    if min_cost == float("inf"):
        return -1, []

    # Reconstruct the path
    path = []
    current = best_target
    current_k = k

    while current_k > 0:
        path.append(current)
        current = predecessor[(current, current_k)]
        current_k -= 1

    path.reverse()

    return min_cost, path


def minimum_mean_path_value(V,edges,source_nodes,target_nodes):
    min_path = None
    min_cost = float("inf")
    for i in range(1,len(V)+1):
        min_cost_i,min_path_i = shortest_path_with_k_edges(V,edges,source_nodes,target_nodes,i)
        if min_cost_i != -1:
            min_cost = min(min_cost_i ,min_cost)
            if min_cost_i == min_cost:
                min_path = min_path_i
    return min_cost,min_path

def inf_inf(automata):
    V, edges, source_nodes, state_numbers,target_nodes,save_for_reconstruct = convertToGraph(automata)
    if V is None:
        return "ERROR"
    intersection = set(source_nodes).intersection(set(target_nodes))
    if len(intersection) > 0:
        return 0.0
    mean_cycle,karp_path = karp_mean_cycle(V,edges,source_nodes)
    mean_path_value,dijsktra_path = minimum_mean_path_value(V,edges,source_nodes,target_nodes)
    if mean_cycle < 0 or mean_cycle == float("inf") or mean_cycle > mean_path_value:
        return mean_path_value,dijsktra_path, reconstruct_words(dijsktra_path,save_for_reconstruct)
    return mean_cycle,karp_path, reconstruct_words(karp_path,save_for_reconstruct)


def dijkstra(V, edges, source_nodes, target_nodes):
    D = {}
    predecessor = {}

    for v in V:
        D[v] = float("inf")
        predecessor[v] = None

    for source in source_nodes:
        D[source] = 0

    Q = set(V)

    while Q:
        u = min(Q, key=lambda x: D[x])
        Q.remove(u)

        for v in V:
            if (u, v) in edges:
                if D[v] > D[u] + edges[(u, v)]:
                    D[v] = D[u] + edges[(u, v)]
                    predecessor[v] = u

    min_cost = float("inf")
    best_target = None

    for target in target_nodes:
        if D[target] < min_cost:
            min_cost = D[target]
            best_target = target

    if min_cost == float("inf"):
        return float("inf"), []

    # Reconstruct the path
    path = []
    current = best_target

    while current is not None:
        path.append(current)
        current = predecessor[current]

    path.reverse()

    return min_cost, path

def sum_inf_inf(automata):
    V, edges, source_nodes, state_numbers,target_nodes,save_for_reconstruct = convertToGraph(automata)
    if V is None:
        return "ERROR"
    intersection = set(source_nodes).intersection(set(target_nodes))
    if len(intersection) > 0:
        return 0.0
    return dijkstra(V,edges,source_nodes,target_nodes),save_for_reconstruct

def compute_pairs_with_balance(b, range_min, range_max):
    pairs = []
    for i in range(range_min, range_max + 1):
        j = i + b
        if range_min <= j <= range_max:
            pairs.append((j, i))
    return pairs
    
def save_only_reachable(V, E, initial_states):
    reachable = set()
    to_explore = set(state for state in initial_states)
    
    while to_explore:
        current = to_explore.pop()
        if current not in reachable:
            reachable.add(current)
            for (src, dest), weight in E.items():
                if src == current and dest not in reachable:
                    to_explore.add(dest)
    
    V = [v for v in V if v in reachable]
    E = {(src, dest): weight for (src, dest), weight in E.items() if src in reachable and dest in reachable}
    
    return V, E

def build_balanced_graph(automata, state_numbers):
    for i in automata.start_states:
        print(i.value)
    print(automata.start_states)
    V = []
    E = {}
    start = []
    save_for_later = {}
    if isinstance(automata, DeterministicFiniteAutomaton):
        start = [state_numbers[automata.start_state.value]]
    else:
        for start_state in automata.start_states:
            start.append(state_numbers[start_state.value])

    initial_states = [(state, state, 0) for state in start]
    for state in automata.states:
        if state.value in state_numbers:
            for other_state in automata.states:
                for i in range(-len(state_numbers), len(state_numbers) + 1):
                    V.append((state_numbers[state.value], state_numbers[other_state.value], i))
    for state in automata.states:
        for symbol in automata.symbols:
            next_state = automata.to_dict().get(state, {}).get(symbol, None)
            if next_state is not None:
                if not isinstance(next_state, set):
                    next_state = {next_state}
                for ns in next_state:
                    if ns.value in state_numbers and state.value in state_numbers:
                        label_value = labelFunction(symbol)
                        weight_value = weightFunction(symbol)
                        pairs = compute_pairs_with_balance(label_value, -len(state_numbers), len(state_numbers))
                        for pair in pairs:
                            for other_state in automata.states:
                                first =(state_numbers[state.value], state_numbers[other_state.value], pair[1])
                                second = (state_numbers[ns.value], state_numbers[other_state.value], pair[0])
                                first_second =(state_numbers[other_state.value], state_numbers[state.value], pair[1])
                                second_second = (state_numbers[other_state.value], state_numbers[ns.value], pair[0])
                                if first in V and second in V:
                                    saved = E.get((first, second), None)
                                    if saved is None or saved < weight_value:
                                        E[(first, second)] = weight_value
                                        save_for_later[(first, second)] = symbol
                                elif first_second in V and second_second in V:
                                    saved = E.get((first_second, second_second), None)
                                    if saved is None or saved < weight_value:
                                        E[(first_second, second_second)] = weight_value
                                        save_for_later[(first_second, second_second)] = symbol
    print(len(V))
    print(len(E))
    print(initial_states)
    V, E = save_only_reachable(V, E, initial_states)
    print(len(V))
    print(len(E))
    return V, E, initial_states, save_for_later

def omega_graph(product):
    if product is None:
        return float("inf")
    print(product.states)
    state = BFS_on_automata(product)
    V,E,initial_states,save_for_later = build_balanced_graph(product,state)
    logger.info(f'{len(V)} vertices and {len(E)} edges')
    return V,E,initial_states,state,save_for_later

def reconstruct_words(path, save_for_later):
    words = []
    first_word = ""
    second_word = ""
    if len(path) == 1:
        words.append(save_for_later[(path[0], path[0])])
    else:
        for i in range(len(path) - 1):
            words.append(save_for_later[(path[i], path[i + 1])])
    print(f"Path: {path}, save_for_later: {save_for_later}")
    for symbol in words:
        print (symbol)
        print(type(symbol.value[0]))
        if symbol.value[0] != "epsilon":
            print(type(symbol.value[0]))
            first_word += str(symbol.value[0])
        if symbol.value[1] != "epsilon":
            second_word+=str(symbol.value[1])
    
    return [compress_string(first_word), compress_string(second_word)]

def compress_string(s):
    n = len(s)
    
    def find_optimal_repeated_substring(s):
        for length in range(1, n // 2 + 1):
            substring = s[:length]
            repeated_count = n // length
            if substring * repeated_count == s[:length * repeated_count]:
                remainder = s[length * repeated_count:]
                return substring, repeated_count, remainder
        return s, 1, ""

    def compress_part(s):
        repeated_substring, count, remainder = find_optimal_repeated_substring(s)
        compressed_part = f"({repeated_substring})^{count}"
        return compressed_part + remainder

    # Try compressing the whole string
    compressed_full = compress_part(s)
    
    # Try compressing with splits
    best_compressed = compressed_full
    for i in range(1, n):
        left_part = compress_part(s[:i])
        right_part = compress_part(s[i:])
        combined_compressed = left_part + right_part
        if len(combined_compressed) < len(best_compressed):
            best_compressed = combined_compressed

    return best_compressed


def value_of_arena(arena):
    def evaluate_adam_vertex(vertex, eve_values):
        return min(eve_values.get(edge[1], float("inf")) + edge[0] for edge in arena.edges.get(vertex, []))

    def evaluate_eve_vertex(vertex, adam_values):
        return max(adam_values.get(edge[1], float("-inf")) + edge[0] for edge in arena.edges.get(vertex, []))

    def is_adam_vertex(vertex):
        return vertex in arena.adam_vertices

    def is_eve_vertex(vertex):
        return vertex in arena.eve_vertices

    adam_values = {v: 0 for v in arena.adam_vertices}
    eve_values = {v: 0 for v in arena.eve_vertices}

    for _ in range(len(arena.adam_vertices) + len(arena.eve_vertices)):
        for vertex in arena.adam_vertices:
            adam_values[vertex] = evaluate_adam_vertex(vertex, eve_values)

        for vertex in arena.eve_vertices:
            eve_values[vertex] = evaluate_eve_vertex(vertex, adam_values)

    return adam_values[arena.initial_vertex]

class Arena:
    def __init__(self):
        self.adam_vertices = set()
        self.eve_vertices = set()
        self.edges = {}
        self.initial_vertex = None

    def add_adam_vertex(self, vertex):
        self.adam_vertices.add(vertex)

    def add_eve_vertex(self, vertex):
        self.eve_vertices.add(vertex)

    def add_edge(self, from_vertex, cost, to_vertex):
        if from_vertex not in self.edges:
            self.edges[from_vertex] = []
        self.edges[from_vertex].append((cost, to_vertex))

    def set_initial_vertex(self, vertex):
        self.initial_vertex = vertex

def edit_distance(a, b):
    return 1
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(a)][len(b)]

def build_arena(automata1, automata2):
    Σ = set(symbol.value for symbol in automata1.symbols) | set(symbol.value for symbol in automata2.symbols)
    Q = automata1.states
    δ = automata1.to_dict()
    q0 = automata1.start_states if not isinstance(automata1, DeterministicFiniteAutomaton) else {automata1.start_state}

    Q_prime = automata2.states
    δ_prime = automata2.to_dict()
    q0_prime = automata2.start_states if not isinstance(automata2, DeterministicFiniteAutomaton) else {automata2.start_state}

    arena = Arena()
    arena.set_initial_vertex((next(iter(q0)).value, next(iter(q0_prime)).value))

    for q in Q:
        for q_prime in Q_prime:
            adam_vertex = (q.value, q_prime.value)
            arena.add_adam_vertex(adam_vertex)
            for a in Σ:
                eve_vertex = (q.value, q_prime.value, a)
                arena.add_eve_vertex(eve_vertex)

    for q in Q:
        for q_prime in Q_prime:
            adam_vertex = (q.value, q_prime.value)
            for a in Σ:
                if q in δ and Symbol(a) in δ[q]:
                    p_set = δ[q][Symbol(a)]
                    for p in p_set if isinstance(p_set, set) else {p_set}:
                        eve_vertex = (p.value, q_prime.value, a)
                        arena.add_edge(adam_vertex, 0, eve_vertex)

                for p_prime in Q_prime:
                    if p_prime in δ_prime and Symbol(a) in δ_prime[p_prime]:
                        if isinstance(δ_prime[p_prime][Symbol(a)], set):
                            c = min(edit_distance(a, v.value) for v in δ_prime[p_prime][Symbol(a)])
                        else:
                            c = edit_distance(a, δ_prime[p_prime][Symbol(a)].value)
                        eve_vertex = (q.value, q_prime.value, a)
                        arena.add_edge(eve_vertex, c, (q.value, p_prime.value))

    return arena


